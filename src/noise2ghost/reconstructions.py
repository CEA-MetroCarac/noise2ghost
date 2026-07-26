"""
This module provides functions for reconstructing images from noisy data using various algorithms.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal

import corrct as cct
from corrct.solvers import SolutionInfo
import matplotlib.pyplot as plt
import numpy as np
from autoden.losses import LossRegularizer
from autoden.algorithms.noise2noise import N2N
from autoden.algorithms.noise2void import N2V
from autoden.models.io import load_model
from autoden.models.config import NetworkParams, create_network
from corrct.param_tuning import PerfMeterTask, PerfMeterBatch
from corrct.struct_illum import MaskCollection, ProjectorGhostImaging
from numpy.typing import NDArray
from torch.nn import Module

from noise2ghost.algos import (
    INR,
    N2G,
    post_process_scale_bias,
    DataScaleBias,
    compute_scaling_ghost_imaging,
    rescale_bias_inputs,
)
from noise2ghost.models.config import NetworkParamsINR
from noise2ghost.models.inr import SIREN

SAVE_MODEL_CNN_PATH = Path("./model.pt").expanduser()


@dataclass
class RecParsCNN:
    """Dataclass for storing reconstruction parameters for a CNN model."""

    model: str | Path | Module | NetworkParams = SAVE_MODEL_CNN_PATH
    data_sb: DataScaleBias | None = None
    num_splits: int | None = 4
    num_perms: int = 6
    lower_limit: float | None = None
    epochs: int = 1024 * 3
    lr: float = 3e-4  # https://x.com/karpathy/status/801621764144971776
    optim_algo: str = "adam"
    cv_fraction: float = 0.1
    num_averages: int = 1
    rng_seed: int | None = None


def _get_model(model: str | Path | Module | NetworkParams) -> Module:
    """
    Load or create a model based on the input.

    Parameters
    ----------
    model : str | Path | Module | NetworkParams
        The model to load or create. Can be a file path, a module, or network parameters.

    Returns
    -------
    Module
        The loaded or created model.
    """
    if isinstance(model, (str, Path)):
        model_state = load_model(model)
        return create_network(model_state)
    elif isinstance(model, NetworkParams):
        return create_network(model)
    else:
        return deepcopy(model)


def _get_projector(
    masks_or_proj: NDArray | MaskCollection | ProjectorGhostImaging,
    projector_cls: Callable[[NDArray | MaskCollection], ProjectorGhostImaging] = ProjectorGhostImaging,
) -> ProjectorGhostImaging:
    if isinstance(masks_or_proj, ProjectorGhostImaging):
        prj = masks_or_proj
    elif isinstance(masks_or_proj, (MaskCollection, np.ndarray)):
        prj = projector_cls(masks_or_proj)
    else:
        raise ValueError("You should provide one of [NDArray | MaskCollection | ProjectorGhostImaging]")
    return prj


def reconstruct_variational(
    masks: NDArray | MaskCollection | ProjectorGhostImaging,
    buckets: NDArray,
    bucket_weights: NDArray | None = None,
    iterations: int = 2000,
    reg: cct.regularizers.BaseRegularizer | None = None,
    projector_cls: Callable[[NDArray | MaskCollection], ProjectorGhostImaging] = ProjectorGhostImaging,
    normalize: bool = False,
    verbose: bool = False,
) -> tuple[NDArray, cct.solvers.SolutionInfo, PerfMeterTask]:
    """
    Perform variational reconstruction.

    Parameters
    ----------
    masks : NDArray | MaskCollection | ProjectorGhostImaging
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    iterations : int, optional
        Number of iterations for the solver, by default 2000.
    reg : cct.regularizers.BaseRegularizer | None, optional
        Regularizer to use, by default None.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    tuple[NDArray, cct.solvers.SolutionInfo | None, PerfMeterTask]
        The reconstructed image and solver information.
    """
    c0 = perf_counter()

    prj = _get_projector(masks, projector_cls=projector_cls)

    data_sb = None
    if normalize:
        data_sb = compute_scaling_ghost_imaging(masks=prj.mc.masks_enc, buckets=buckets, projector_cls=projector_cls)
        masks, buckets = rescale_bias_inputs(data_sb, prj.mc.masks_enc, buckets)
        prj = _get_projector(masks, projector_cls=projector_cls)

    c1 = perf_counter()

    if reg is None and bucket_weights is None:
        rec = prj.fbp(buckets, adjust_scaling=False)
        info = SolutionInfo("FBP", 0, None)
    else:
        if bucket_weights is None:
            dt = cct.data_terms.DataFidelity_l2()
        else:
            dt = cct.data_terms.DataFidelity_l2w(bucket_weights)
        solver = cct.solvers.PDHG(verbose=verbose, regularizer=reg, data_term=dt, leave_progress=False, criterion="loss_rec")
        rec, info = solver(prj, buckets, iterations=iterations)

    rec = post_process_scale_bias(rec, prj, buckets)
    if data_sb is not None:
        rec = (rec + data_sb.bias_out) / data_sb.scale_out

    c2 = perf_counter()

    return rec, info, PerfMeterTask(init_time_s=c1 - c0, exec_time_s=c2 - c1, total_time_s=c2 - c0)


def denoise_neural_cnn(
    masks: NDArray,
    buckets: NDArray,
    rec_pars: RecParsCNN = RecParsCNN(),
    reg_val: float | LossRegularizer | None = None,
) -> tuple[NDArray, dict[str, NDArray], PerfMeterTask]:
    """
    Perform neural network-based denoising of least-squares reconstructions using CNN.

    Parameters
    ----------
    masks : NDArray
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    rec_pars : RecParsCNN, optional
        Reconstruction parameters, by default RecParsCNN().
    reg_val : float | LossRegularizer | None, optional
        Regularization value, by default None.

    Returns
    -------
    tuple
        The reconstructed image, training losses, and performance metrics.
    """
    c0 = perf_counter()

    model = deepcopy(_get_model(rec_pars.model))
    solver_n2g = N2G(model=model, data_scale_bias=rec_pars.data_sb)

    inp_recs_trn, _, _, _, _ = solver_n2g.prepare_data(
        masks, buckets, num_splits=rec_pars.num_splits, num_perms=1, tst_fraction=0.0, cv_fraction=0.0
    )

    if rec_pars.num_splits is not None:
        denoiser = N2N(model, reg_val=reg_val)
    else:
        denoiser = N2V(model, reg_val=reg_val)

    den_data = denoiser.prepare_data(np.squeeze(inp_recs_trn))

    c1 = perf_counter()

    losses = denoiser.train(
        *den_data,
        epochs=rec_pars.epochs,
        learning_rate=rec_pars.lr,
        lower_limit=rec_pars.lower_limit,
        optimizer=rec_pars.optim_algo,
    )

    gi_rec = denoiser.infer(den_data[0])

    gi_rec = post_process_scale_bias(gi_rec, masks, buckets)

    c2 = perf_counter()

    return gi_rec, losses, PerfMeterTask(init_time_s=c1 - c0, exec_time_s=c2 - c1, total_time_s=c2 - c0)


def reconstruct_neural_cnn(
    masks: NDArray,
    buckets: NDArray,
    rec_pars: RecParsCNN = RecParsCNN(),
    reg_val: float | LossRegularizer | None = None,
    device: str = "cuda",
) -> tuple[NDArray, dict[str, NDArray], PerfMeterTask]:
    """
    Perform neural network-based reconstruction using CNN.

    Parameters
    ----------
    masks : NDArray
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    rec_pars : RecParsCNN, optional
        Reconstruction parameters, by default RecParsCNN().
    reg_val : float | LossRegularizer | None, optional
        Regularization value, by default None.
    device : str, optional
        Device to use for computation, by default "cuda".

    Returns
    -------
    tuple
        The reconstructed image, training losses, and performance metrics.
    """
    c0 = perf_counter()

    is_n2g = rec_pars.num_splits is not None

    model = _get_model(rec_pars.model)
    solver_n2g = N2G(model=model, data_scale_bias=rec_pars.data_sb, reg_val=reg_val, device=device)

    inp_recs_trn, tgt_trn_data, _, tgt_cv_data, tgt_trn_inds = solver_n2g.prepare_data(
        masks,
        buckets,
        num_splits=rec_pars.num_splits,
        num_perms=rec_pars.num_perms,
        tst_fraction=0.0,
        cv_fraction=rec_pars.cv_fraction,
        rng_seed=rec_pars.rng_seed,
    )

    c1 = perf_counter()

    losses = solver_n2g.train(
        inp_recs_trn,
        tgt_trn_data,
        tgt_trn_inds if is_n2g else None,
        tgt_cv_data,
        epochs=rec_pars.epochs,
        learning_rate=rec_pars.lr,
        lower_limit=rec_pars.lower_limit,
        algo=rec_pars.optim_algo,
    )
    gi_rec = solver_n2g.infer(inp_recs_trn)
    if is_n2g:
        gi_rec = gi_rec.mean(axis=0)

    gi_rec = post_process_scale_bias(gi_rec, masks, buckets)

    c2 = perf_counter()

    return gi_rec, losses, PerfMeterTask(init_time_s=c1 - c0, exec_time_s=c2 - c1, total_time_s=c2 - c0)


def fit_variational_reg_weight(
    masks: NDArray | MaskCollection | ProjectorGhostImaging,
    buckets: NDArray,
    bucket_weights: NDArray | None = None,
    reg: Callable[[float], cct.regularizers.BaseRegularizer] = cct.regularizers.Regularizer_TV2D,
    lambda_range: tuple[float, float, int] | NDArray = (1e-3, 1e2, 2),
    iterations: int = 2000,
    lower_limit: float | None = None,
    num_averages: int = 3,
    parallel_eval: bool | int | Executor = False,
    projector_cls: Callable[[NDArray | MaskCollection], ProjectorGhostImaging] = ProjectorGhostImaging,
    normalize: bool = False,
    criterion: Literal["max_iter", "loss_rec", "loss_val"] = "loss_val",
    plot_final_loss: bool = True,
) -> tuple[float, NDArray, PerfMeterBatch]:
    """
    Fit the regularization weight for variational reconstruction.

    Parameters
    ----------
    masks : NDArray | MaskCollection | ProjectorGhostImaging
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    reg : Callable[[float], cct.regularizers.BaseRegularizer], optional
        Function to create a regularizer given a lambda value, by default cct.regularizers.Regularizer_TV2D.
    lambda_range : tuple[float, float, int] | NDArray, optional
        Range of lambda values to test, by default (1e-3, 1e2, 2).
    iterations : int, optional
        Number of iterations for the solver, by default 2000.
    lower_limit : float | None, optional
        Lower limit for the reconstruction, by default None.
    num_averages : int, optional
        Number of averages for cross-validation, by default 3.
    parallel_eval : bool | int | Executor, optional
        Whether to use parallel evaluation, by default False.

    Returns
    -------
    tuple[float, NDArray]
        The best lambda value and the reconstructed image.
    """
    solver_verbose = not isinstance(parallel_eval, Executor)

    prj = _get_projector(masks, projector_cls=projector_cls)

    data_sb = None
    if normalize:
        data_sb = compute_scaling_ghost_imaging(masks=prj.mc.masks_enc, buckets=buckets, projector_cls=projector_cls)
        masks, buckets = rescale_bias_inputs(data_sb, prj.mc.masks_enc, buckets)
        prj = _get_projector(masks, projector_cls=projector_cls)

    def solve_reg(lam: float, b_val_mask: NDArray | None = None) -> tuple[NDArray, SolutionInfo]:
        if bucket_weights is None:
            data_term = cct.data_terms.DataFidelity_l2()
        else:
            data_term = cct.data_terms.DataFidelity_l2w(bucket_weights)
        solver = cct.solvers.PDHG(
            regularizer=reg(lam), verbose=solver_verbose, leave_progress=False, criterion=criterion, data_term=data_term
        )
        return solver(prj, buckets, iterations=iterations, lower_limit=lower_limit, b_val_mask=b_val_mask)

    cv = cct.param_tuning.CrossValidation(
        buckets.shape, num_averages=num_averages, verbose=True, plot_result=True, parallel_eval=parallel_eval
    )
    cv.task_exec_function = solve_reg

    if isinstance(lambda_range, np.ndarray):
        lams = np.array(lambda_range, ndmin=1)
    else:
        lams = cct.param_tuning.get_lambda_range(*lambda_range)

    f_avgs, _, all_info = cv.compute_loss_values(lams, return_all=True)
    lam_min, _ = cv.fit_loss_min(lams, f_avgs)
    print(f"Minimum lambda found at: {lam_min:.3e}")

    c0 = perf_counter()

    b_cv_mask = None
    if criterion.lower() == "loss_val":
        b_cv_mask = cct.param_tuning.create_random_test_mask(buckets.shape)
    rec_reg, info = solve_reg(lam_min, b_cv_mask)
    rec_reg = post_process_scale_bias(rec_reg, prj, buckets)
    if data_sb is not None:
        rec_reg = (rec_reg + data_sb.bias_out) / data_sb.scale_out

    c1 = perf_counter()
    final_rec_perfs = PerfMeterTask(init_time_s=0.0, exec_time_s=c1 - c0, total_time_s=c1 - c0)

    stats = sum([info[2] for info in all_info], PerfMeterBatch())
    stats.append(final_rec_perfs)

    if plot_final_loss:
        fig, axs = plt.subplots(1, 1)
        axs.plot(info.residuals_rec_rel, label="Residual - REC")
        axs.stem(info.best_residual_ind_rec - 1, info.get_best_residual_rec(), label="Best - REC")
        if criterion.lower() == "loss_val":
            axs.plot(info.residuals_val_rel, label="Residual - VAL")
            axs.stem(info.best_residual_ind_val - 1, info.get_best_residual_val(), label="Best - VAL")
        axs.grid()
        axs.legend()
        axs.set_ylabel("Loss")
        axs.set_xlabel("Iterations")
        axs.set_yscale("log")
        fig.tight_layout()

    return lam_min, rec_reg, stats


@dataclass
class TempResults:
    """Stores reconstruction results including input and output reconstructions and associated losses."""

    recs_in: NDArray
    recs_out: list[NDArray] = field(default_factory=lambda: [])
    losses: list[dict[str, NDArray]] = field(default_factory=lambda: [])


def fit_neural_cnn_reg_weight(
    masks: NDArray,
    buckets: NDArray,
    rec_pars: RecParsCNN,
    reg_vals: Sequence[float | None] | NDArray | float | None = None,
    device: str = "cuda",
) -> tuple[float, NDArray, dict[str, NDArray], PerfMeterBatch]:
    """
    Fit the regularization weight for neural network-based reconstruction using CNN.

    Parameters
    ----------
    masks : NDArray
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    rec_pars : RecParsCNN
        Reconstruction parameters.
    reg_vals : Sequence[float | None] | NDArray | float | None, optional
        Regularization values to test, by default None.
    device : str, optional
        Device to use for computation, by default "cuda".

    Returns
    -------
    tuple[float, NDArray, dict[str, NDArray], PerfMeterBatch]
        The best regularization weight, the reconstructed image, the training losses, and the performance metrics.
    """
    cb0 = perf_counter()

    is_n2g = rec_pars.num_splits is not None

    model = _get_model(rec_pars.model)

    reg_vals = np.array(reg_vals, ndmin=1)
    stats_tasks: list[PerfMeterTask] = []
    all_results: list[TempResults] = []
    min_losses = np.zeros((rec_pars.num_averages, len(reg_vals)), dtype=np.float32)

    cb_i = perf_counter() - cb0

    for ii_a in range(rec_pars.num_averages):
        cba0 = perf_counter()

        solver_n2g = N2G(model=model, reg_val=None, data_scale_bias=rec_pars.data_sb)
        recs_trn_inp, data_trn_tgt, _, data_val_tgt, inds_trn_tgt = solver_n2g.prepare_data(
            masks,
            buckets,
            num_splits=rec_pars.num_splits,
            num_perms=rec_pars.num_perms,
            tst_fraction=0.0,
            cv_fraction=rec_pars.cv_fraction,
        )
        data_sb = deepcopy(solver_n2g.data_sb)

        results = TempResults(recs_in=recs_trn_inp)

        cb_i += perf_counter() - cba0

        for ii_r, reg_val in enumerate(reg_vals):
            print(f"{ii_r+1}/{len(reg_vals)} Lambda: {reg_val:.3e}")
            ct0 = perf_counter()
            solver_n2g = N2G(model=deepcopy(model), reg_val=reg_val, data_scale_bias=data_sb, device=device)
            ct1 = perf_counter()
            losses = solver_n2g.train(
                recs_trn_inp,
                data_trn_tgt,
                inds_trn_tgt if is_n2g else None,
                data_val_tgt,
                epochs=rec_pars.epochs,
                learning_rate=rec_pars.lr,
                algo=rec_pars.optim_algo,
                lower_limit=rec_pars.lower_limit,
            )
            gi_rec = solver_n2g.infer(recs_trn_inp)
            if is_n2g:
                gi_rec = gi_rec.mean(axis=0)

            gi_rec = post_process_scale_bias(gi_rec, masks, buckets)
            ct2 = perf_counter()

            results.recs_out.append(gi_rec)
            results.losses.append(losses)
            stats_tasks.append(PerfMeterTask(init_time_s=(ct1 - ct0), exec_time_s=(ct2 - ct1), total_time_s=(ct2 - ct0)))

        all_results.append(results)
        min_losses[ii_a, ...] = [np.nanmin(losses["loss_tst"]) for losses in results.losses]

    min_losses_avg = min_losses.mean(axis=0)
    min_losses_std = min_losses.std(axis=0)
    best_rec_ind = np.argmin(min_losses_avg)

    min_reg_weight, _ = cct.param_tuning.fit_func_min(
        reg_vals, f_vals=min_losses_avg, f_stds=min_losses_std, verbose=True, plot_result=True
    )

    gi_rec, losses, rec_perf = reconstruct_neural_cnn(masks, buckets, rec_pars=rec_pars, reg_val=min_reg_weight, device=device)
    stats_tasks.append(rec_perf)

    cb2 = perf_counter()
    cb_i += rec_perf.init_time_s
    cb_p = sum([rp.exec_time_s for rp in stats_tasks])
    stats_batch = PerfMeterBatch(init_time_s=cb_i, proc_time_s=cb_p, total_time_s=cb2 - cb0, tasks_perf=stats_tasks)

    print(f"{'N2G' if is_n2g else 'GIDC'}: Found lowest loss for lambda = {min_reg_weight} (ind: {best_rec_ind})")
    return min_reg_weight, gi_rec, losses, stats_batch


def fit_neural_inr_reg_weight(
    masks: NDArray,
    buckets: NDArray,
    reg_vals: Sequence[float | None] | NDArray | float | None,
    epochs: int = 1024 * 6,
    device: str = "cuda",
    lower_limit: float | None = None,
) -> tuple[float, NDArray, dict[str, NDArray], PerfMeterBatch]:
    """
    Fit the regularization weight for neural network-based reconstruction using INR.

    Parameters
    ----------
    masks : NDArray
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    reg_vals : Sequence[float | None] | NDArray | float | None
        Regularization values to test.
    epochs : int, optional
        Number of training epochs, by default 1024 * 6.
    device : str, optional
        Device to use for computation, by default "cuda".

    Returns
    -------
    tuple[float, NDArray, dict[str, NDArray], PerfMeterBatch]
        The best regularization weight, the reconstructed image, the training losses, and the performance metrics.
    """
    cb0 = perf_counter()

    if reg_vals is None or isinstance(reg_vals, float):
        reg_vals = [reg_vals]
    model_def = NetworkParamsINR(n_features=512, n_layers=2, n_embeddings=256)
    solver_inr_base = INR(model=model_def, reg_val=None, device=device)
    encode_grid, data_trn_tgt, _, data_val_tgt = solver_inr_base.prepare_data(
        masks, buckets, tst_fraction=0.0, cv_fraction=0.1
    )
    data_sb = deepcopy(solver_inr_base.data_sb)
    model: SIREN = solver_inr_base.model

    reg_vals = np.array(reg_vals, ndmin=1)
    results = TempResults(recs_in=encode_grid.detach().to("cpu").numpy().copy())
    stats_tasks = []

    cb1 = perf_counter()

    for ii_r, reg_val in enumerate(reg_vals):
        print(f"{ii_r+1}/{len(reg_vals)} Lambda: {reg_val:.3e}")
        ct0 = perf_counter()
        solver_inr = INR(
            model=deepcopy(model), reg_val=reg_val, data_scaling_bias=data_sb, encoder=solver_inr_base.encoder, device=device
        )
        ct1 = perf_counter()
        losses = solver_inr.train(
            encode_grid,
            data_trn_tgt,
            data_val_tgt,
            epochs=epochs,
            algo="adam",
            learning_rate=1e-4,
            weight_decay=0.0,
            lower_limit=lower_limit,
        )
        gi_rec = solver_inr.infer(encode_grid).mean(axis=0)

        gi_rec = post_process_scale_bias(gi_rec, masks, buckets)
        ct2 = perf_counter()

        results.recs_out.append(gi_rec)
        results.losses.append(losses)
        stats_tasks.append(PerfMeterTask(init_time_s=(ct1 - ct0), exec_time_s=(ct2 - ct1), total_time_s=(ct2 - ct0)))

    min_losses = np.array([np.nanmin(losses["loss_tst"]) for losses in results.losses])
    best_rec_ind = np.argmin(min_losses)

    min_reg_weight, _ = cct.param_tuning.fit_func_min(
        reg_vals, f_vals=min_losses, f_stds=min_losses, verbose=True, plot_result=True
    )

    ct0 = perf_counter()
    solver_inr = INR(
        model=deepcopy(model),
        reg_val=min_reg_weight,
        data_scaling_bias=data_sb,
        encoder=solver_inr_base.encoder,
        device=device,
    )
    ct1 = perf_counter()
    losses = solver_inr.train(
        encode_grid, data_trn_tgt, data_val_tgt, epochs=epochs, algo="adam", learning_rate=1e-4, weight_decay=0.0
    )
    gi_rec = solver_inr.infer(encode_grid).mean(axis=0)

    gi_rec = post_process_scale_bias(gi_rec, masks, buckets)
    ct2 = perf_counter()

    stats_tasks.append(PerfMeterTask(init_time_s=(ct1 - ct0), exec_time_s=(ct2 - ct1), total_time_s=(ct2 - ct0)))
    cb2 = perf_counter()
    stats_batch = PerfMeterBatch(init_time_s=cb1 - cb0, proc_time_s=cb2 - cb1, total_time_s=cb2 - cb0, tasks_perf=stats_tasks)

    print(f"INR: Found lowest loss for lambda = {min_reg_weight} (ind: {best_rec_ind})")
    return min_reg_weight, gi_rec, losses, stats_batch
