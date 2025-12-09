"""Algorithms."""

import copy as cp
from collections.abc import Sequence

import corrct as cct
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn as nn
from autoden.algorithms.denoiser import DataScaleBias, Denoiser, get_normalization_range
from autoden.models.config import create_optimizer
from autoden.models.param_utils import fix_invalid_gradient_values, get_num_parameters
from numpy.typing import NDArray
from tqdm.auto import trange

from noise2ghost.models.config import NetworkParamsINR
from noise2ghost.models.inr import SIREN, PositionalEncoder


def _compute_num_chunks(epochs: int, num_inps: int, size_cap: int = 8) -> NDArray:
    num_chunks = np.ceil(np.log2(epochs / np.fmax(np.arange(epochs), 1)))
    return np.fmin(num_chunks, min(num_inps, size_cap)).astype(int)


class DatasetSplit:
    """Store the dataset split indices."""

    trn_inds: NDArray[np.integer]
    tst_inds: NDArray[np.integer]

    def __init__(self, trn_inds: NDArray, tst_inds: NDArray) -> None:
        self.trn_inds = trn_inds
        self.tst_inds = tst_inds


def split_realizations(
    masks: NDArray,
    buckets: NDArray,
    num_splits: int | None = None,
    num_perms: int = 1,
    tst_fraction: float = 0.1,
    cv_fraction: float = 0.1,
    pre_permute: bool = True,
) -> Sequence[tuple[NDArray, NDArray]]:
    """Partition the set of realizations in multiple sub-sets.

    Parameters
    ----------
    masks : NDArray
        The full set of masks
    buckets : NDArray
        The full set of buckets
    num_splits : int | None
        Number of splits of the full realization set
    num_perms : int, optional
        Number of permutations of the full realization set, by default 1
    tst_fraction : float, optional
        Fraction of the realizations in the test set, by default 0.1
    cv_fraction : float, optional
        Fraction of the realizations in the cross-validation, by default 0.1

    Returns
    -------
    Sequence[tuple[NDArray, NDArray]]
        The list of partitions
    """
    tot_realizations = len(buckets)
    print(f"Total number of realizations: {tot_realizations}, split as:")

    if num_splits is not None:
        split_size = int((tot_realizations * (1 - cv_fraction - tst_fraction)) // num_splits)
        trn_size = split_size * num_splits
        print(
            f"- A set of {num_splits} splits of {split_size} buckets/masks each,"
            f" resulting in training set size: {trn_size} ({1 - tst_fraction - cv_fraction:%})"
        )
        # Slices are within the set
        slices_trn = [slice(ii * split_size, (ii + 1) * split_size) for ii in range(num_splits)]
    else:
        trn_size = int(tot_realizations * (1 - cv_fraction - tst_fraction))
        print(f"- Training set size: {trn_size} ({1 - tst_fraction - cv_fraction:%})")
        slices_trn = []

    tst_size = int(tot_realizations * tst_fraction)
    cv_size = tot_realizations - trn_size - tst_size
    print(f"- Test set size: {tst_size} ({tst_fraction:%})\n" f"- Cross-validation set size: {cv_size} ({cv_fraction:%})")

    masks_flat = masks.reshape([-1, *masks.shape[-2:]])
    # Think of randomizing masks and buckets, in order to avoid bias towards last realizations
    if pre_permute:
        rnd_perm = np.random.permutation(tot_realizations)
        masks_flat = masks_flat[rnd_perm]
        buckets = buckets[rnd_perm]

    masks_trn = masks_flat[:trn_size]
    buckets_trn = buckets[:trn_size]

    masks_tst = masks_flat[trn_size : trn_size + tst_size]
    buckets_tst = buckets[trn_size : trn_size + tst_size]

    masks_cv = masks_flat[trn_size + tst_size :]
    buckets_cv = buckets[trn_size + tst_size :]

    inds_trn_inp = []
    inds_trn_tgt = []

    if num_splits is not None:
        for _ in trange(num_perms, desc="Computing permutation tuples"):
            rnd_perm = np.random.permutation(trn_size)

            for s in slices_trn:
                inds_inp_s = np.zeros(len(rnd_perm), dtype=bool)
                inds_inp_s[rnd_perm[s]] = True
                inds_trn_inp.append(inds_inp_s)
                inds_trn_tgt.append(np.logical_not(inds_inp_s))

        inds_trn_inp = np.stack(inds_trn_inp, axis=0)
        inds_trn_tgt = np.stack(inds_trn_tgt, axis=0)
    else:
        inds_trn_inp = np.ones([1, len(buckets_trn)], dtype=bool)
        inds_trn_tgt = inds_trn_inp

    data_split_inds = (inds_trn_inp, inds_trn_tgt)
    data_trn = (masks_trn, buckets_trn)
    data_tst = (masks_tst, buckets_tst)
    data_cv = (masks_cv, buckets_cv)

    return data_split_inds, data_trn, data_tst, data_cv


def _gi_fwd(masks: pt.Tensor, image: pt.Tensor) -> pt.Tensor:
    new_masks_shape = [*masks.shape[:-2], int(np.prod(np.array(masks.shape[-2:])))]
    new_image_shape = [*image.shape[:-2], int(np.prod(np.array(image.shape[-2:]))), 1]
    masks = masks.reshape(new_masks_shape)
    image = image.reshape(new_image_shape)

    out = masks.matmul(image)
    return out.squeeze(dim=-1)


def compute_scaling_ghost_imaging(
    masks: NDArray, buckets: NDArray, adjust_scaling: bool = False, verbose: bool = True
) -> DataScaleBias:
    """
    Compute input, output, and target data scaling and bias for the ghost imaging reconstruction.

    Parameters
    ----------
    masks : NDArray
        Stack of masks.
    buckets : NDArray
        List of buckets, corresponding to the masks.
    adjust_scaling : bool, optional
        If True, adjust the scaling factors of the output data to the expected perceived range (wrong though!).
        Default is False.
    verbose : bool, optional
        If True, print detailed information about the scaling and bias computation process.
        Default is True.

    Returns
    -------
    DataScaleBias
        An instance of DataScaleBias containing the computed scaling and bias values.
    """
    data_sb = DataScaleBias()
    print("Computing least-squares reconstruction for normalization:")
    mc = cct.struct_illum.MaskCollection(masks)
    p = cct.struct_illum.ProjectorGhostImaging(mc)
    rec_ls = p.fbp(buckets, adjust_scaling=adjust_scaling)

    stats_det = get_normalization_range(buckets, percentile=0.01)
    stats_rec = get_normalization_range(rec_ls, percentile=0.01)
    if verbose:
        print("Input ranges:")
        print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
        print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

    data_scaling_recs = 1 / (stats_rec[1] - stats_rec[0])
    data_bias_recs = stats_rec[2] * data_scaling_recs

    masks = masks / data_scaling_recs

    mc = cct.struct_illum.MaskCollection(masks)
    p = cct.struct_illum.ProjectorGhostImaging(mc)

    data_bias_tgt = p(np.ones_like(rec_ls) * data_bias_recs)
    buckets = buckets - data_bias_tgt
    rec_ls = p.fbp(buckets, adjust_scaling=adjust_scaling)

    stats_det = get_normalization_range(buckets, percentile=0.01)
    stats_rec = get_normalization_range(rec_ls, percentile=0.01)
    if verbose:
        print("Input ranges:")
        print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
        print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

    refined_data_bias_recs = stats_rec[2]

    data_sb.scale_out = data_scaling_recs
    data_sb.bias_out = data_bias_recs + refined_data_bias_recs

    refined_data_bias_tgt = p(np.ones_like(rec_ls) * refined_data_bias_recs)
    buckets = buckets - refined_data_bias_tgt
    rec_ls = p.fbp(buckets, adjust_scaling=adjust_scaling)

    stats_det = get_normalization_range(buckets, percentile=0.01)
    stats_rec = get_normalization_range(rec_ls, percentile=0.01)
    if verbose:
        print("Input ranges:")
        print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
        print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

    data_scaling_tgt = 1 / (stats_det[1] - stats_det[0])

    data_sb.scale_tgt = data_scaling_tgt
    data_sb.bias_tgt = data_bias_tgt + refined_data_bias_tgt

    buckets = buckets * data_scaling_tgt
    masks = masks * data_scaling_tgt

    mc = cct.struct_illum.MaskCollection(masks)
    p = cct.struct_illum.ProjectorGhostImaging(mc)

    rec_ls = p.fbp(buckets, adjust_scaling=adjust_scaling)

    stats_det = get_normalization_range(buckets, percentile=0.01)
    stats_rec = get_normalization_range(rec_ls, percentile=0.01)
    if verbose:
        print("Input ranges:")
        print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
        print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

    return data_sb


class N2G(Denoiser):
    """Perform self-supervised reconstruction of GI."""

    def prepare_data(
        self,
        inp_masks: NDArray,
        inp_buckets: NDArray,
        num_splits: int | None = 4,
        num_perms: int = 1,
        tst_fraction: float = 0.1,
        cv_fraction: float = 0.1,
        force_scaling: bool = True,
    ) -> tuple[NDArray, tuple[NDArray, NDArray], tuple[NDArray, NDArray], tuple[NDArray, NDArray], NDArray]:
        adjust_scaling = False

        if self.data_sb is None or force_scaling:
            self.data_sb = compute_scaling_ghost_imaging(inp_masks, inp_buckets, adjust_scaling=adjust_scaling)

        inp_buckets = inp_buckets - self.data_sb.bias_tgt

        inp_buckets = inp_buckets * self.data_sb.scale_tgt
        inp_masks = inp_masks * self.data_sb.scale_tgt / self.data_sb.scale_out

        if self.verbose and not (self.data_sb is None or force_scaling):
            mc = cct.struct_illum.MaskCollection(inp_masks)
            p = cct.struct_illum.ProjectorGhostImaging(mc)

            rec_ls = p.fbp(inp_buckets, adjust_scaling=adjust_scaling)

            stats_det = get_normalization_range(inp_buckets, percentile=0.01)
            stats_rec = get_normalization_range(rec_ls, percentile=0.01)
            print("Input ranges:")
            print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
            print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

        data_trn_split, data_trn_tgt, data_tst_tgt, data_cv_tgt = split_realizations(
            masks=inp_masks,
            buckets=inp_buckets,
            num_splits=num_splits,
            num_perms=num_perms,
            tst_fraction=tst_fraction,
            cv_fraction=cv_fraction,
        )
        data_trn_m = data_trn_tgt[0]
        data_trn_b = data_trn_tgt[1]
        inds_trn_inp = data_trn_split[0]
        inds_trn_tgt = data_trn_split[1]

        recs_inp = []
        for ii in trange(len(inds_trn_inp), desc="Computing reconstructions", disable=(not self.verbose)):
            ms = data_trn_m[inds_trn_inp[ii]]
            bs = data_trn_b[inds_trn_inp[ii]]
            mc = cct.struct_illum.MaskCollection(ms)
            p = cct.struct_illum.ProjectorGhostImaging(mc)

            rec_ls = p.fbp(bs, adjust_scaling=adjust_scaling)
            recs_inp.append(rec_ls)
        recs_inp = np.stack(recs_inp, axis=0)

        return recs_inp, data_trn_tgt, data_tst_tgt, data_cv_tgt, inds_trn_tgt

    def train(
        self,
        inp_trn_r: NDArray,
        tgt_trn_mb: tuple[NDArray, NDArray],
        tgt_trn_inds: NDArray | None,
        tgt_tst_mb: tuple[NDArray, NDArray],
        epochs: int,
        algo: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        lower_limit: float | NDArray | None = None,
        accum_grads: bool = False,
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses = dict(trn=[], tst=[], tst_sbi=[])
        loss_track_type = "tst"

        loss_data_fn = nn.MSELoss(reduction="sum")
        loss_reg_fn = self._get_regularization()
        optim = create_optimizer(self.model, algo=algo, learning_rate=learning_rate, weight_decay=weight_decay)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_out - self.data_sb.bias_out

        best_epoch = -1
        best_loss = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        if tgt_trn_inds is not None:
            tgt_trn_b = np.stack([tgt_trn_mb[1][inds] for inds in tgt_trn_inds], axis=0)
        else:
            tgt_trn_b = tgt_trn_mb[1][None, ...]

        tgt_trn_m_t = pt.tensor(tgt_trn_mb[0].astype(np.float32), device=self.device, requires_grad=True)
        tgt_trn_b_t = pt.tensor(tgt_trn_b.astype(np.float32), device=self.device)
        inp_trn_r_t = pt.tensor(inp_trn_r[:, None, ...].astype(np.float32), device=self.device)

        tgt_tst_m_t = pt.tensor(tgt_tst_mb[0].astype(np.float32), device=self.device)
        tgt_tst_b_t = pt.tensor(tgt_tst_mb[1].astype(np.float32), device=self.device)
        tgt_tst_b_t_sbi = (tgt_tst_b_t - tgt_tst_b_t.mean()) / (tgt_tst_b_t.std() + 1e-5)

        num_inp_recs = inp_trn_r_t.shape[0]
        all_num_chunks = _compute_num_chunks(epochs, num_inps=num_inp_recs)
        previous_chunks = all_num_chunks[0] * 2

        num_tgt_trn_b = tgt_trn_b_t.numel()
        num_tgt_tst_b = tgt_tst_b_t.numel()

        for epoch in trange(epochs, desc=f"Training {algo.upper()}", disable=(not self.verbose)):
            # Train
            self.model.train()

            num_chunks = int(all_num_chunks[epoch])
            if previous_chunks != num_chunks:
                optim = create_optimizer(self.model, algo=algo, learning_rate=learning_rate, weight_decay=weight_decay)
                previous_chunks = num_chunks

            loss_trn_val = 0

            for chunk in range(num_chunks):
                inp_trn_r_ep_ch = inp_trn_r_t[chunk::num_chunks]

                optim.zero_grad()

                if accum_grads:
                    for ii in range(inp_trn_r_ep_ch.shape[0]):
                        tmp_trn_i = self.model(inp_trn_r_ep_ch[ii : ii + 1 :])
                        tmp_trn_b = _gi_fwd(tgt_trn_m_t, tmp_trn_i[0, 0, :, :])

                        if tgt_trn_inds is not None:
                            tmp_trn_b = tmp_trn_b[tgt_trn_inds[chunk + ii * num_chunks]]
                            tgt_tmp_b = tgt_trn_b_t[chunk + ii * num_chunks]
                        else:
                            tgt_tmp_b = tgt_trn_b_t

                        loss_trn = loss_data_fn(tmp_trn_b, tgt_tmp_b) / num_tgt_trn_b
                        if loss_reg_fn is not None:
                            loss_trn += loss_reg_fn(tmp_trn_i)
                        if lower_limit is not None:
                            loss_trn += nn.ReLU(inplace=False)(-tmp_trn_i.flatten() + lower_limit).mean()

                        loss_trn.backward()
                        loss_trn_val += loss_trn.item()
                else:
                    # Compute network's output
                    tmp_trn_i = self.model(inp_trn_r_ep_ch)
                    # Compute residual on target
                    tmp_trn_b = _gi_fwd(tgt_trn_m_t, tmp_trn_i[..., 0, :, :])

                    if tgt_trn_inds is not None:
                        tmp_trn_b = pt.stack(
                            [tmp_trn_b[ii][inds] for ii, inds in enumerate(tgt_trn_inds[chunk::num_chunks])], dim=0
                        )
                        tgt_tmp_b = tgt_trn_b_t[chunk::num_chunks]
                    else:
                        tgt_tmp_b = tgt_trn_b_t

                    if inp_trn_r_t.shape[0] > 1:
                        tmp_trn_i = pt.concatenate((tmp_trn_i, tmp_trn_i.mean(dim=0, keepdim=True)), dim=0)

                    loss_trn = loss_data_fn(tmp_trn_b, tgt_tmp_b) / num_tgt_trn_b
                    if loss_reg_fn is not None:
                        loss_trn += loss_reg_fn(tmp_trn_i)
                    if lower_limit is not None:
                        loss_trn += nn.ReLU(inplace=False)(-tmp_trn_i.flatten() + lower_limit).mean()

                    loss_trn.backward()
                    loss_trn_val += loss_trn.item()

                fix_invalid_gradient_values(self.model)
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                optim.step()

            losses["trn"].append(loss_trn_val)

            # Test
            self.model.eval()
            with pt.inference_mode():
                # Compute network's output
                tmp_tst_i = self.model(inp_trn_r_t)

                # Compute residual on target
                tmp_tst_b = _gi_fwd(tgt_tst_m_t, tmp_tst_i.mean(dim=(0, 1)))

                loss_tst = loss_data_fn(tmp_tst_b, tgt_tst_b_t) / num_tgt_tst_b
                losses["tst"].append(loss_tst.item())

                tmp_tst_b_sbi = (tmp_tst_b - tmp_tst_b.mean()) / (tmp_tst_b.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(tmp_tst_b_sbi, tgt_tst_b_t_sbi) / num_tgt_tst_b
                losses["tst_sbi"].append(loss_tst_sbi.item())

            # Check improvement
            if losses[loss_track_type][-1] < best_loss:
                best_loss = losses[loss_track_type][-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

            # if epoch in [1, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, epochs - 1]:
            if self.verbose and epoch in [1000, 2000, 5000, 10000, 20000, epochs - 1]:
                tmp_trn_i = self.model(inp_trn_r_t)
                # tmp_trn_i, latent = self.model(inp_trn_r_t, return_latent=True)
                tmp_trn_b = _gi_fwd(tgt_trn_m_t, tmp_trn_i[..., 0, :, :])
                # latent_l1_norm = pt.linalg.vector_norm(latent, ord=1) / latent.numel()
                print(
                    f"It {epoch}: loss_trn = {loss_trn_val:.5}, loss_{loss_track_type} = {losses[loss_track_type][-1]:.5}"
                    f" (best: {best_loss:.5}, ep: {best_epoch})"
                )  # , l1 = {latent_l1_norm:.5}

                fig, axs = plt.subplots(1, 4, figsize=[12, 3.25])
                fig.suptitle(f"Iteration: {epoch}, n.inp: {inp_trn_r_t.shape[0]}")
                axs[0].imshow(inp_trn_r_t[0, 0].detach().cpu().numpy())
                axs[0].set_title("input_0")
                axs[1].imshow(tmp_trn_i[0, 0].detach().cpu().numpy())
                axs[1].set_title("net(input_0)")
                axs[2].imshow(tmp_trn_i.mean(dim=(0, 1)).detach().cpu().numpy())
                axs[2].set_title("net(input).mean()")
                axs[3].plot(tmp_trn_b.mean(dim=0).detach().cpu().numpy(), label="fwd")
                axs[3].plot(tgt_trn_mb[1], label="tgt")
                axs[3].plot(tmp_trn_b.mean(dim=0).detach().cpu().numpy() - tgt_trn_mb[1], label="diff")
                axs[3].grid()
                axs[3].legend()
                axs[3].set_title("buckets")
                fig.tight_layout()

        self.model.load_state_dict(best_state)

        if self.verbose:
            print(f"Best epoch: {best_epoch}, with loss_{loss_track_type}: {best_loss:.5}")
        if self.save_epochs_dir is not None:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        losses = {f"loss_{loss_type}": np.array(loss_vals) for loss_type, loss_vals in losses.items()}

        self._plot_loss_curves(losses, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

        return losses


class INR(Denoiser):
    """Perform INR reconstruction of GI."""

    model: SIREN
    encoder: PositionalEncoder

    def __init__(
        self,
        model: NetworkParamsINR | SIREN,
        data_scaling_bias: DataScaleBias | None = None,
        encoder: PositionalEncoder | None = None,
        reg_val: float | None = 1e-5,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        save_epochs_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the Implicit Neural Representation (INR) based reconstruction algorithm.

        Parameters
        ----------
        model : NetworkParamsINR | SIREN
            The neural network model to be used. It can be an instance of
            `NetworkParamsINR` or `SIREN`.
        data_scaling_bias : DataScaleBias | None, optional
            An optional data scaling and bias object, by default None.
        encoder : PositionalEncoder | None, optional
            An optional positional encoder, by default None.
        reg_val : float | None, optional
            Regularization value, by default 1e-5.
        device : str, optional
            The device to run the model on, either 'cuda' or 'cpu', by default
            "cuda" if CUDA is available, otherwise "cpu".
        save_epochs_dir : str | None, optional
            Directory to save model checkpoints, by default None.
        verbose : bool, optional
            If True, print verbose output, by default True.

        Raises
        ------
        ValueError
            If the provided model is not supported.
        """
        if isinstance(model, NetworkParamsINR):
            self.model = model.get_model(device=device)
        elif isinstance(model, SIREN):
            self.model = model.to(device)
            self.model.device = device
        else:
            raise ValueError(f"Unsupported model: {model}")
        if verbose:
            get_num_parameters(self.model, verbose=True)

        if encoder is None:
            encoder = PositionalEncoder(num_embeddings=self.model.n_embeddings, ndims=self.model.n_channels_in, device=device)
        self.encoder = encoder

        self.data_sb = data_scaling_bias

        self.reg_val = reg_val
        self.device = device
        self.save_epochs_dir = save_epochs_dir
        self.verbose = verbose

    def prepare_data(
        self,
        inp_masks: NDArray,
        inp_buckets: NDArray,
        tst_fraction: float = 0.1,
        cv_fraction: float = 0.1,
        force_scaling: bool = False,
    ) -> tuple[pt.Tensor, tuple[NDArray, NDArray], tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """
        Prepare the data for training, testing, and validation.

        Parameters
        ----------
        inp_masks : NDArray
            Input masks array.
        inp_buckets : NDArray
            Input buckets array.
        tst_fraction : float, optional
            Fraction of the data to be used for testing, by default 0.1.
        cv_fraction : float, optional
            Fraction of the data to be used for cross-validation, by default 0.1.
        force_scaling : bool, optional
            If True, forces the computation of data scaling and bias, by default False
            .

        Returns
        -------
        tuple
            A tuple containing:
            - encode_grid : pt.Tensor
                The encoded grid for the neural network.
            - data_trn_tgt : tuple[NDArray, NDArray]
                Training data (masks and buckets).
            - data_tst_tgt : tuple[NDArray, NDArray]
                Testing data (masks and buckets).
            - data_val_tgt : tuple[NDArray, NDArray]
                Validation data (masks and buckets).

        Notes
        -----
        This method scales the input masks and buckets, splits the data into training,
        testing, and validation sets, and encodes the grid for the neural network.
        """
        adjust_scaling = False

        if self.data_sb is None or force_scaling:
            self.data_sb = compute_scaling_ghost_imaging(inp_masks, inp_buckets, adjust_scaling=adjust_scaling)

        inp_buckets = inp_buckets - self.data_sb.bias_tgt

        inp_buckets = inp_buckets * self.data_sb.scale_tgt
        inp_masks = inp_masks * self.data_sb.scale_tgt / self.data_sb.scale_out

        if self.verbose and not (self.data_sb is None or force_scaling):
            mc = cct.struct_illum.MaskCollection(inp_masks)
            p = cct.struct_illum.ProjectorGhostImaging(mc)

            rec_ls = p.fbp(inp_buckets, adjust_scaling=adjust_scaling)

            stats_det = get_normalization_range(inp_buckets, percentile=0.01)
            stats_rec = get_normalization_range(rec_ls, percentile=0.01)
            print("Input ranges:")
            print(f"- buckets: range [{stats_det[0]:.3}, {stats_det[1]:.3}], mean {stats_det[2]:.3}")
            print(f"- reconstruction: range [{stats_rec[0]:.3}, {stats_rec[1]:.3}], mean {stats_rec[2]:.3}")

        grid = self.encoder.create_grid(inp_masks.shape[-2:])
        print(f'grid shape: {grid.shape}')
        encode_grid = self.encoder.embed(grid)
        print(f'encode_grid shape: {encode_grid.shape}')

        tot_realizations = len(inp_buckets)
        print(f"Total number of realizations: {tot_realizations}, split as:")

        trn_size = int(tot_realizations * (1 - cv_fraction - tst_fraction))
        print(f"- Training set size: {trn_size} ({1 - tst_fraction - cv_fraction:%})")

        tst_size = int(tot_realizations * tst_fraction)
        cv_size = tot_realizations - trn_size - tst_size
        print(f"- Test set size: {tst_size} ({tst_fraction:%})\n" f"- Cross-validation set size: {cv_size} ({cv_fraction:%})")

        masks_flat = inp_masks.reshape([-1, *inp_masks.shape[-2:]])

        masks_trn = masks_flat[:trn_size]
        buckets_trn = inp_buckets[:trn_size]

        masks_tst = masks_flat[trn_size : trn_size + tst_size]
        buckets_tst = inp_buckets[trn_size : trn_size + tst_size]

        masks_val = masks_flat[trn_size + tst_size :]
        buckets_val = inp_buckets[trn_size + tst_size :]

        data_trn_tgt = (masks_trn, buckets_trn)
        data_tst_tgt = (masks_tst, buckets_tst)
        data_val_tgt = (masks_val, buckets_val)

        return encode_grid, data_trn_tgt, data_tst_tgt, data_val_tgt

    def train(
        self,
        encode_grid: pt.Tensor,
        tgt_trn_mb: tuple[NDArray, NDArray],
        tgt_tst_mb: tuple[NDArray, NDArray],
        epochs: int,
        algo: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
        """
        Train the Implicit Neural Representation (INR) model.

        Parameters
        ----------
        encode_grid : pt.Tensor
            The encoded grid for the neural network.
        tgt_trn_mb : tuple[NDArray, NDArray]
            Training data (masks and buckets).
        tgt_tst_mb : tuple[NDArray, NDArray]
            Testing data (masks and buckets).
        epochs : int
            Number of training epochs.
        algo : str, optional
            Optimization algorithm to use, by default "adam".
        learning_rate : float, optional
            Learning rate for the optimizer, by default 1e-4.
        weight_decay : float, optional
            Weight decay for the optimizer, by default 1e-2.
        lower_limit : float | NDArray | None, optional
            Lower limit constraint for the model output, by default None.

        Returns
        -------
        dict[str, NDArray]
            A dictionary containing the training and testing losses:
            - loss_trn : NDArray
                Training losses for each epoch.
            - loss_tst : NDArray
                Testing losses for each epoch.
            - loss_tst_sbi : NDArray
                Standardized testing losses for each epoch.

        Raises
        ------
        ValueError
            If the number of epochs is less than 1.

        Notes
        -----
        This method trains the INR model using the provided training and testing data.
        It supports different optimization algorithms and includes regularization and
        lower limit constraints if specified.
        """
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []

        loss_data_fn = nn.MSELoss(reduction="mean")
        loss_reg_fn = self._get_regularization()
        optim = create_optimizer(self.model, algo=algo, learning_rate=learning_rate, weight_decay=weight_decay)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_out - self.data_sb.bias_out

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        tgt_trn_m_t = pt.tensor(tgt_trn_mb[0].astype(np.float32), device=self.device, requires_grad=True)
        tgt_trn_b_t = pt.tensor(tgt_trn_mb[1].astype(np.float32), device=self.device)

        tgt_tst_m_t = pt.tensor(tgt_tst_mb[0].astype(np.float32), device=self.device)
        tgt_tst_b_t = pt.tensor(tgt_tst_mb[1].astype(np.float32), device=self.device)
        tgt_tst_b_t_sbi = (tgt_tst_b_t - tgt_tst_b_t.mean()) / (tgt_tst_b_t.std() + 1e-5)

        for epoch in trange(epochs, desc=f"Training {algo.upper()}", disable=(not self.verbose)):
            # Train
            self.model.train()

            optim.zero_grad()

            # Compute network's output
            tmp_trn_i = self.model(encode_grid)
            tmp_trn_i = tmp_trn_i.reshape([1, 1, *tgt_trn_m_t.shape[-2:]])

            # Compute residual on target
            tmp_trn_b = _gi_fwd(tgt_trn_m_t, tmp_trn_i)

            loss_trn = loss_data_fn(tmp_trn_b, tgt_trn_b_t)
            if loss_reg_fn is not None:
                loss_trn += loss_reg_fn(tmp_trn_i)
            if lower_limit is not None:
                loss_trn += nn.ReLU(inplace=False)(-tmp_trn_i.flatten() + lower_limit).mean()

            loss_trn.backward()
            loss_trn_val = loss_trn.item()

            fix_invalid_gradient_values(self.model)

            optim.step()

            losses_trn.append(loss_trn_val)

            # Test
            self.model.eval()
            with pt.inference_mode():
                # Compute network's output
                tmp_tst_i = self.model(encode_grid)
                tmp_tst_i = tmp_trn_i.reshape([1, 1, *tgt_trn_m_t.shape[-2:]])

                # Compute residual on target
                tmp_tst_b = _gi_fwd(tgt_tst_m_t, tmp_tst_i.mean(dim=(0, 1)))
                loss_tst = loss_data_fn(tmp_tst_b, tgt_tst_b_t)

                loss_tst_val = loss_tst.item()
                losses_tst.append(loss_tst_val)

                tmp_tst_b_sbi = (tmp_tst_b - tmp_tst_b.mean()) / (tmp_tst_b.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(tmp_tst_b_sbi, tgt_tst_b_t_sbi)
                losses_tst_sbi.append(loss_tst_sbi.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch, optim.state_dict())

            # if epoch in [1, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, epochs - 1]:
            if self.verbose and epoch in [1000, 2000, 5000, 10000, 20000, epochs - 1]:
                tmp_trn_i = self.model(encode_grid)
                tmp_trn_i = tmp_trn_i.reshape([1, 1, *tgt_trn_m_t.shape[-2:]])
                # tmp_trn_i, latent = self.model(inp_trn_r_t, return_latent=True)
                tmp_trn_b = _gi_fwd(tgt_trn_m_t, tmp_trn_i.mean(dim=(0, 1)))
                # latent_l1_norm = pt.linalg.vector_norm(latent, ord=1) / latent.numel()
                print(
                    f"It {epoch}: loss_trn = {loss_trn_val:.5}, loss_tst = {loss_tst_val:.5}"
                    f" (best: {best_loss_tst:.5}, ep: {best_epoch})"
                )  # , l1 = {latent_l1_norm:.5}

                fig, axs = plt.subplots(1, 3, figsize=[9, 3.25])
                fig.suptitle(f"Iteration: {epoch}, n.inp: {encode_grid.shape[0]}")
                axs[0].imshow(tmp_trn_i[0, 0].detach().cpu().numpy())
                axs[0].set_title("net(input_0)")
                axs[1].imshow(tmp_trn_i.mean(dim=(0, 1)).detach().cpu().numpy())
                axs[1].set_title("net(input).mean()")
                axs[2].plot(tmp_trn_b.mean(dim=0).detach().cpu().numpy(), label="fwd")
                axs[2].plot(tgt_trn_mb[1], label="tgt")
                axs[2].plot(tmp_trn_b.mean(dim=0).detach().cpu().numpy() - tgt_trn_mb[1], label="diff")
                axs[2].grid()
                axs[2].legend()
                axs[2].set_title("buckets")
                fig.tight_layout()

        self.model.load_state_dict(best_state)

        if self.verbose:
            print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir is not None:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        losses = dict(loss_trn=np.array(losses_trn), loss_tst=np.array(losses_tst), loss_tst_sbi=np.array(losses_tst_sbi))

        self._plot_loss_curves(losses, f"Unsupervised (INR) {self.__class__.__name__} {algo.upper()}")

        return losses

    def infer(self, inp: pt.Tensor) -> NDArray:
        """
        Inference, given an encoded coordinate grid.

        Parameters
        ----------
        inp : pt.Tensor
            The encoded coordinate grid.

        Returns
        -------
        NDArray
            The reconstructed image.
        """
        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp)
            output = out_t.to("cpu").numpy()

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scale_out

        return output


def post_process_scale_bias(
    out_rec: NDArray,
    inp_masks: NDArray | cct.struct_illum.MaskCollection | cct.struct_illum.ProjectorGhostImaging,
    inp_buckets: NDArray,
    verbose: bool = False,
) -> NDArray:
    """
    Post-process the reconstructed image by fitting and applying the correct scale and bias (according to the data).

    Parameters
    ----------
    out_rec : NDArray
        The reconstructed image to be post-processed.
    inp_masks : NDArray | cct.struct_illum.MaskCollection | cct.struct_illum.ProjectorGhostImaging,
        The input masks or projector used for the reconstruction.
    inp_buckets : NDArray
        The input buckets used for the reconstruction.
    verbose : bool, optional
        If True, print the fitted scale and bias values. Default is False.

    Returns
    -------
    NDArray
        The post-processed image with the scale and bias applied.

    Notes
    -----
    This function takes a reconstructed image and applies a scale and bias to it based on the input masks and buckets.
    The scale and bias are fitted using the `fit_scale_bias` function from the `cct.processing.post` module.
    """
    if isinstance(inp_masks, cct.struct_illum.ProjectorGhostImaging):
        prj = inp_masks
    else:
        prj = cct.struct_illum.ProjectorGhostImaging(inp_masks)

    scale, bias = cct.processing.post.fit_scale_bias(out_rec, inp_buckets, prj)
    if verbose:
        print(f"Fitted: {scale = }, {bias = }")
    return out_rec * scale + bias
