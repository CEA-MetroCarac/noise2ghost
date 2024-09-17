"""Algorithms."""

import copy as cp
from collections.abc import Mapping, Sequence
from typing import Union
from dataclasses import dataclass

import corrct as cct
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn as nn
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm, trange

from .models import NetworkParams, create_network, create_optimizer
from .io import load_model_state, save_model_state
from .losses import LossRegularizer, LossTV


def _get_normalization(vol: NDArray, percentile: Union[float, None] = None) -> tuple[float, float, float]:
    if percentile is not None:
        vol_sort = np.sort(vol.flatten())
        ind_min = int(np.fmax(vol_sort.size * percentile, 0))
        ind_max = int(np.fmin(vol_sort.size * (1 - percentile), vol_sort.size - 1))
        return vol_sort[ind_min], vol_sort[ind_max], vol_sort[ind_min : ind_max + 1].mean()
    else:
        return vol.min(), vol.max(), vol.mean()


def _single_channel_imgs_to_tensor(imgs: NDArray, device: str, dtype: DTypeLike = np.float32) -> pt.Tensor:
    imgs = np.array(imgs, ndmin=3).astype(dtype)[..., None, :, :]
    return pt.tensor(imgs, device=device)


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
    num_splits: Union[int, None] = None,
    num_perms: int = 1,
    tst_fraction: float = 0.1,
    cv_fraction: float = 0.1,
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
    print(f"Total number of realizations: {len(buckets)}, split as:")

    if num_splits is not None:
        split_size = int((len(buckets) * (1 - cv_fraction - tst_fraction)) // num_splits)
        trn_size = split_size * num_splits
        print(
            f"- A set of {num_splits} splits of {split_size} buckets/masks each,"
            f" resulting in training set size: {trn_size} ({1 - tst_fraction - cv_fraction:%})"
        )
        # Slices are within the set
        slices_trn = [slice(ii * split_size, (ii + 1) * split_size) for ii in range(num_splits)]
    else:
        trn_size = int(len(buckets) * (1 - cv_fraction - tst_fraction))
        print(f"- Training set size: {trn_size} ({1 - tst_fraction - cv_fraction:%})")
        slices_trn = []

    tst_size = int(len(buckets) * tst_fraction)
    cv_size = len(buckets) - trn_size - tst_size
    print(f"- Test set size: {tst_size} ({tst_fraction:%})\n" f"- Cross-validation set size: {cv_size} ({cv_fraction:%})")

    masks_flat = masks.reshape([-1, *masks.shape[-2:]])
    # Think of randomizing masks and buckets, in order to avoid bias towards last realizations

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


@dataclass
class DataScalingBias:
    """Data scaling and bias."""

    scaling_inp: Union[float, NDArray] = 1.0
    scaling_out: Union[float, NDArray] = 1.0
    scaling_tgt: Union[float, NDArray] = 1.0

    bias_inp: Union[float, NDArray] = 0.0
    bias_out: Union[float, NDArray] = 0.0
    bias_tgt: Union[float, NDArray] = 0.0


def compute_scaling_supervised(inp: NDArray, tgt: NDArray) -> DataScalingBias:
    range_vals_inp = _get_normalization(inp, percentile=0.001)
    range_vals_tgt = _get_normalization(tgt, percentile=0.001)

    sb = DataScalingBias()
    sb.scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])
    sb.scaling_out = sb.scaling_tgt

    sb.bias_inp = range_vals_inp[2] * sb.scaling_inp
    sb.bias_tgt = range_vals_tgt[2] * sb.scaling_tgt
    sb.bias_out = sb.bias_tgt

    return sb


def compute_scaling_selfsupervised(inp: NDArray) -> DataScalingBias:
    range_vals_inp = _get_normalization(inp, percentile=0.001)

    sb = DataScalingBias()
    sb.scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scaling_out = sb.scaling_tgt = sb.scaling_inp

    sb.bias_inp = range_vals_inp[2] * sb.scaling_inp
    sb.bias_out = sb.bias_tgt = sb.bias_inp

    return sb


class Denoiser:
    """Denoising images."""

    data_sb: DataScalingBias | None

    model: pt.nn.Module
    device: str

    save_epochs_dir: str | None
    verbose: bool

    def __init__(
        self,
        model: str | NetworkParams | pt.nn.Module | Mapping | None,
        data_scaling_bias: DataScalingBias | None = None,
        reg_tv_val: float | None = 1e-5,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        save_epochs_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the noise2noise method.

        Parameters
        ----------
        model : str | NetworkParams | pt.nn.Module | Mapping | None
            Type of neural network to use or a specific network (or state) to use
        data_scaling_inp : Union[float, None], optional
            Scaling of the input data, by default None
        data_scaling_tgt : Union[float, None], optional
            Scaling of the output, by default None
        reg_tv_val : Union[float, None], optional
            Deep-image prior regularization value, by default 1e-5
        device : str, optional
            Device to use, by default "cuda" if cuda is available, otherwise "cpu"
        save_epochs_dir : str | None, optional
            Directory where to save network states at each epoch.
            If None disabled, by default None
        verbose : bool, optional
            Whether to produce verbose output, by default True
        """
        # if model is None or isinstance(model, (int, Mapping)):
        #     if isinstance(model, int):
        #         if self.save_epochs_dir is None:
        #             raise ValueError("Directory for saving epochs not specified")

        #         state_dict = load_model_state(self.save_epochs_dir, epoch_num=model)
        #         model = state_dict["state_dict"]

        #     if isinstance(model, Mapping):
        #         self.model.load_state_dict(model)
        #         self.model.to(self.device)
        #     else:
        #         raise ValueError(f"Invalid model state: {model}")
        # el
        if isinstance(model, (str, NetworkParams)):
            self.model = create_network(model, device=device)
        elif isinstance(model, pt.nn.Module):
            self.model = model.to(device)
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.data_sb = data_scaling_bias

        self.reg_val = reg_tv_val
        self.device = device
        self.save_epochs_dir = save_epochs_dir
        self.verbose = verbose

    def train_supervised(
        self,
        inp: NDArray,
        tgt: NDArray,
        epochs: int,
        tst_inds: Sequence[int] | NDArray,
        algo: str = "adam",
    ):
        """Supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images
        tgt : NDArray
            The target images
        epochs : int
            Number of training epochs
        tst_inds : Sequence[int] | NDArray
            The validation set indices
        algo : str, optional
            Learning algorithm to use, by default "adam"
        """
        num_imgs = inp.shape[0]
        tst_inds = np.array(tst_inds, dtype=int)
        if np.any(tst_inds < 0) or np.any(tst_inds >= num_imgs):
            raise ValueError(
                f"Each cross-validation index should be greater or equal than 0, and less than the number of images {num_imgs}"
            )
        trn_inds = np.delete(np.arange(num_imgs), obj=tst_inds)

        if tgt.ndim == (inp.ndim - 1):
            tgt = np.tile(tgt[None, ...], [num_imgs, *np.ones_like(tgt.shape)])

        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp
        tgt = tgt * self.data_sb.scaling_tgt - self.data_sb.bias_tgt

        # Create datasets
        dset_trn = (inp[trn_inds], tgt[trn_inds])
        dset_tst = (inp[tst_inds], tgt[tst_inds])

        reg = LossTV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        loss_trn, loss_tst = self._train_selfsimilar(dset_trn, dset_tst, epochs=epochs, algo=algo, regularizer=reg)

        if self.verbose:
            self._plot_loss_curves(loss_trn, loss_tst, f"Supervised {algo.upper()}")

    def _train_selfsimilar(
        self,
        dset_trn: tuple[NDArray, NDArray],
        dset_tst: tuple[NDArray, NDArray],
        epochs: int,
        algo: str = "adam",
        regularizer: Union[LossRegularizer, None] = None,
        lower_limit: Union[float, NDArray, None] = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scaling_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = _single_channel_imgs_to_tensor(dset_trn[0], device=self.device)
        tgt_trn_t = _single_channel_imgs_to_tensor(dset_trn[1], device=self.device)

        inp_tst_t = _single_channel_imgs_to_tensor(dset_tst[0], device=self.device)
        tgt_tst_t = _single_channel_imgs_to_tensor(dset_tst[1], device=self.device)

        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            self.model.train()

            optim.zero_grad()
            out_trn: pt.Tensor = self.model(inp_trn_t)
            loss_trn = loss_data_fn(out_trn, tgt_trn_t)
            if regularizer is not None:
                loss_trn += regularizer(out_trn)
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_trn.flatten() + lower_limit).mean()
            loss_trn.backward()

            loss_trn_val = loss_trn.item()
            losses_trn.append(loss_trn_val)

            optim.step()

            # Test
            self.model.eval()
            loss_tst_val = 0
            with pt.inference_mode():
                out_tst = self.model(inp_tst_t)
                loss_tst = loss_data_fn(out_tst, tgt_tst_t)

                loss_tst_val = loss_tst.item()
                losses_tst.append(loss_tst_val)

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir:
                self._save_state(epoch, self.model.state_dict(), optim.state_dict())

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir:
            self._save_state(best_epoch, best_state, best_optim, is_best=True)

        self.model.load_state_dict(best_state)

        return np.array(losses_trn), np.array(losses_tst)

    def _train_pixelmask_small(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        algo: str = "adam",
        regularizer: Union[LossRegularizer, None] = None,
        lower_limit: Union[float, NDArray, None] = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scaling_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        n_dims = inp.ndim

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)
        tgt_trn = pt.tensor(tgt[mask_trn].astype(np.float32), device=self.device)
        tgt_tst = pt.tensor(tgt[np.logical_not(mask_trn)].astype(np.float32), device=self.device)

        mask_trn_t = pt.tensor(mask_trn, device=self.device)
        mask_tst_t = pt.tensor(np.logical_not(mask_trn), device=self.device)

        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            optim.zero_grad()
            out_t: pt.Tensor = self.model(inp_t)
            if n_dims == 2:
                out_t_mask = out_t[0, 0]
            else:
                out_t_mask = out_t[:, 0]
            if tgt.ndim == 3 and out_t_mask.ndim == 2:
                out_t_mask = pt.tile(out_t_mask[None, :, :], [tgt.shape[-3], 1, 1])

            out_trn = out_t_mask[mask_trn_t].flatten()

            loss_trn = loss_data_fn(out_trn, tgt_trn)
            if regularizer is not None:
                loss_trn += regularizer(out_t)
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_t.flatten() + lower_limit).mean()
            loss_trn.backward()

            losses_trn.append(loss_trn.item())
            optim.step()

            # Test
            out_tst = out_t_mask[mask_tst_t]
            loss_tst = loss_data_fn(out_tst, tgt_tst)
            losses_tst.append(loss_tst.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                save_model_state(
                    self.save_epochs_dir, epoch_num=epoch, model_state=self.model.state_dict(), optim_state=optim.state_dict()
                )

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir is not None:
            save_model_state(
                self.save_epochs_dir, epoch_num=best_epoch, model_state=best_state, optim_state=best_optim, is_best=True
            )

        self.model.load_state_dict(best_state)

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)

        return losses_trn, losses_tst

    def _save_state(self, epoch_num: int, model_state: Mapping, optim_state: Mapping, is_best: bool = False) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        save_model_state(
            self.save_epochs_dir, epoch_num=epoch_num, model_state=model_state, optim_state=optim_state, is_best=is_best
        )

    def _load_state(self, epoch_num: int | None = None) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        state_dict = load_model_state(self.save_epochs_dir, epoch_num=epoch_num)
        self.model.load_state_dict(state_dict["state_dict"])

    def _plot_loss_curves(self, train_loss: NDArray, test_loss: NDArray, title: Union[str, None] = None) -> None:
        test_argmin = int(np.argmin(test_loss))
        fig, axs = plt.subplots(1, 1, figsize=[7, 2.6])
        if title is not None:
            axs.set_title(title)
        axs.semilogy(np.arange(train_loss.size), train_loss, label="training loss")
        axs.semilogy(np.arange(test_loss.size) + 1, test_loss, label="test loss")
        axs.stem(test_argmin + 1, test_loss[test_argmin], linefmt="C1--", markerfmt="C1o", label=f"Best epoch: {test_argmin}")
        axs.legend()
        axs.grid()
        fig.tight_layout()
        plt.show(block=False)

    def infer(self, inp: NDArray) -> NDArray:
        """Inference, given an initial stack of images.

        Parameters
        ----------
        inp : NDArray
            The input stack of images

        Returns
        -------
        NDArray
            The denoised stack of images
        """
        # Rescale input
        if self.data_sb is not None:
            inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)

        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp_t)
            output = out_t.to("cpu").numpy().reshape(inp.shape)

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scaling_out

        return output


class N2N(Denoiser):
    """Self-supervised denoising from pairs of images."""

    def train_selfsupervised(
        self, inp: NDArray, epochs: int, num_tst_ratio: float = 0.2, strategy: str = "1:X", algo: str = "adam"
    ) -> None:
        if self.data_sb is None:
            self.data_sb = compute_scaling_selfsupervised(inp)

        # Rescale the datasets
        inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp

        mask_trn = np.ones_like(inp, dtype=bool)
        rnd_inds = np.random.random_integers(low=0, high=mask_trn.size - 1, size=int(mask_trn.size * num_tst_ratio))
        mask_trn[np.unravel_index(rnd_inds, shape=mask_trn.shape)] = False

        inp_x = np.stack([np.delete(inp, obj=ii, axis=0).mean(axis=0) for ii in range(len(inp))], axis=0)
        if strategy.upper() == "1:X":
            tmp_inp = inp
            tmp_tgt = inp_x
        elif strategy.upper() == "X:1":
            tmp_inp = inp_x
            tmp_tgt = inp
        else:
            raise ValueError(f"Strategy {strategy} not implemented. Please choose one of: ['1:X', 'X:1']")

        tmp_inp = tmp_inp.astype(np.float32)
        tmp_tgt = tmp_tgt.astype(np.float32)

        reg = LossTV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, mask_trn, epochs=epochs, algo=algo, regularizer=reg
        )

        if self.verbose:
            self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")


class N2G(N2N):
    """Perform self-supervised reconstruction of GI."""

    def _fwd(self, masks: pt.Tensor, image: pt.Tensor) -> pt.Tensor:
        new_masks_shape = [*masks.shape[:-2], int(np.prod(np.array(masks.shape[-2:])))]
        new_image_shape = [*image.shape[:-2], int(np.prod(np.array(image.shape[-2:]))), 1]
        masks = masks.reshape(new_masks_shape)
        image = image.reshape(new_image_shape)

        out = masks.matmul(image)
        return out.squeeze(dim=-1)

    def prepare_data(
        self,
        inp_masks: NDArray,
        inp_buckets: NDArray,
        num_splits: Union[int, None] = 4,
        num_perms: int = 1,
        tst_fraction: float = 0.1,
        cv_fraction: float = 0.1,
        force_scaling: bool = True,
    ) -> Sequence:
        adjust_scaling = False

        if self.data_sb is None or force_scaling:
            self.data_sb = DataScalingBias()

            print("Computing least-squares reconstruction for normalization:")
            mc = cct.struct_illum.MaskCollection(inp_masks)
            p = cct.struct_illum.ProjectorGhostImaging(mc)
            rec_ls = p.fbp(inp_buckets, adjust_scaling=adjust_scaling)

            range_vals_det = _get_normalization(inp_buckets, percentile=0.01)
            range_vals_rec = _get_normalization(rec_ls, percentile=0.01)
            if self.verbose:
                print("Input ranges:")
                print(f"- buckets: min {range_vals_det[0]:.3}, max {range_vals_det[1]:.3}, mean {range_vals_det[2]:.3}")
                print(f"- reconstruction: min {range_vals_rec[0]:.3}, max {range_vals_rec[1]:.3}, mean {range_vals_rec[2]:.3}")

            data_scaling_recs = 1 / (range_vals_rec[1] - range_vals_rec[0])
            data_bias_recs = range_vals_rec[2] * data_scaling_recs

            self.data_sb.scaling_out = data_scaling_recs
            self.data_sb.bias_out = data_bias_recs

            inp_masks = inp_masks / data_scaling_recs

            mc = cct.struct_illum.MaskCollection(inp_masks)
            p = cct.struct_illum.ProjectorGhostImaging(mc)

            data_bias_det = p(np.ones_like(rec_ls) * data_bias_recs)
            inp_buckets = inp_buckets - data_bias_det
            rec_ls = p.fbp(inp_buckets, adjust_scaling=adjust_scaling)

            range_vals_det = _get_normalization(inp_buckets, percentile=0.01)
            range_vals_rec = _get_normalization(rec_ls, percentile=0.01)
            if self.verbose:
                print("Input ranges:")
                print(f"- buckets: min {range_vals_det[0]:.3}, max {range_vals_det[1]:.3}, mean {range_vals_det[2]:.3}")
                print(f"- reconstruction: min {range_vals_rec[0]:.3}, max {range_vals_rec[1]:.3}, mean {range_vals_rec[2]:.3}")

            self.data_sb.scaling_tgt = 1 / (range_vals_det[1] - range_vals_det[0])
            inp_buckets = inp_buckets * self.data_sb.scaling_tgt
            inp_masks = inp_masks * self.data_sb.scaling_tgt

            mc = cct.struct_illum.MaskCollection(inp_masks)
            p = cct.struct_illum.ProjectorGhostImaging(mc)

            rec_ls = p.fbp(inp_buckets, adjust_scaling=adjust_scaling)

            range_vals_det = _get_normalization(inp_buckets, percentile=0.01)
            range_vals_rec = _get_normalization(rec_ls, percentile=0.01)
            if self.verbose:
                print("Input ranges:")
                print(f"- buckets: min {range_vals_det[0]:.3}, max {range_vals_det[1]:.3}, mean {range_vals_det[2]:.3}")
                print(f"- reconstruction: min {range_vals_rec[0]:.3}, max {range_vals_rec[1]:.3}, mean {range_vals_rec[2]:.3}")

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
        tgt_trn_inds: Union[Sequence[NDArray], None],
        tgt_tst_mb: tuple[NDArray, NDArray],
        epochs: int,
        algo: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        lower_limit: Union[float, NDArray, None] = None,
    ) -> tuple[NDArray, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scaling_out - self.data_sb.bias_out

        loss_data_fn = nn.MSELoss(reduction="mean")
        loss_tv_fn = LossTV(lambda_val=self.reg_val) if self.reg_val is not None else None

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        optim = create_optimizer(self.model, algo=algo, learning_rate=learning_rate, weight_decay=weight_decay)

        if tgt_trn_inds is not None:
            tgt_trn_b = np.stack([tgt_trn_mb[1][inds] for inds in tgt_trn_inds], axis=0)
        else:
            tgt_trn_b = tgt_trn_mb[1][None, ...]

        tgt_trn_m_t = pt.tensor(tgt_trn_mb[0].astype(np.float32), device=self.device, requires_grad=True)
        tgt_trn_b_t = pt.tensor(tgt_trn_b.astype(np.float32), device=self.device)
        inp_trn_r_t = pt.tensor(inp_trn_r[:, None, ...].astype(np.float32), device=self.device)

        tgt_tst_m_t = pt.tensor(tgt_tst_mb[0].astype(np.float32), device=self.device)
        tgt_tst_b_t = pt.tensor(tgt_tst_mb[1].astype(np.float32), device=self.device)

        all_num_chunks = _compute_num_chunks(epochs, num_inps=inp_trn_r_t.shape[0])
        previous_chunks = all_num_chunks[0] * 2

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

                # Compute network's output
                tmp_trn_i = self.model(inp_trn_r_ep_ch)

                # Compute residual on target
                tmp_trn_b = self._fwd(tgt_trn_m_t, tmp_trn_i[..., 0, :, :])

                if tgt_trn_inds is not None:
                    tmp_trn_b = pt.stack(
                        [tmp_trn_b[ii][inds] for ii, inds in enumerate(tgt_trn_inds[chunk::num_chunks])], dim=0
                    )
                    tgt_tmp_b = tgt_trn_b_t[chunk::num_chunks]
                else:
                    tgt_tmp_b = tgt_trn_b_t

                if inp_trn_r_t.shape[0] > 1:
                    tmp_trn_i = pt.concatenate((tmp_trn_i, tmp_trn_i.mean(dim=0, keepdim=True)), dim=0)

                loss_trn = loss_data_fn(tmp_trn_b, tgt_tmp_b)
                if loss_tv_fn is not None:
                    loss_trn += loss_tv_fn(tmp_trn_i)
                if lower_limit is not None:
                    loss_trn += nn.ReLU(inplace=False)(-tmp_trn_i.flatten() + lower_limit).mean()
                if num_chunks > 1:
                    s = pt.linalg.svdvals(tmp_trn_i.reshape([tmp_trn_i.shape[0], -1]))
                    loss_trn += 1e-7 * pt.linalg.norm(s[1:], ord=1)

                loss_trn.backward()
                loss_trn_val += loss_trn.item()

                optim.step()

            loss_trn_val /= num_chunks
            losses_trn.append(loss_trn_val)

            # Test
            self.model.eval()
            with pt.inference_mode():
                # Compute network's output
                tmp_tst_i = self.model(inp_trn_r_t)

                # Compute residual on target
                tmp_tst_b = self._fwd(tgt_tst_m_t, tmp_tst_i.mean(dim=(0, 1)))
                loss_tst = loss_data_fn(tmp_tst_b, tgt_tst_b_t)

                loss_tst_val = loss_tst.item()
                losses_tst.append(loss_tst_val)

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch, self.model.state_dict(), optim.state_dict())

            # if epoch in [1, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, epochs - 1]:
            if self.verbose and epoch in [1000, 2000, 5000, 10000, 20000, epochs - 1]:
                tmp_trn_i = self.model(inp_trn_r_t)
                # tmp_trn_i, latent = self.model(inp_trn_r_t, return_latent=True)
                tmp_trn_b = self._fwd(tgt_trn_m_t, tmp_trn_i[..., 0, :, :])
                # latent_l1_norm = pt.linalg.vector_norm(latent, ord=1) / latent.numel()
                print(
                    f"It {epoch}: loss_trn = {loss_trn_val:.5}, loss_tst = {loss_tst_val:.5}"
                    f" (best: {best_loss_tst:.5}, ep: {best_epoch})"
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

        if self.verbose:
            print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir is not None:
            save_model_state(
                self.save_epochs_dir, epoch_num=best_epoch, model_state=best_state, optim_state=best_optim, is_best=True
            )

        self.model.load_state_dict(best_state)

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)
        self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

        return losses_trn, losses_tst
