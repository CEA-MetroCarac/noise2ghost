"""
This module provides functions for reconstructing images from noisy data using various algorithms.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import corrct as cct
from corrct.solvers import SolutionInfo
import numpy as np
from autoden.losses import LossRegularizer
from autoden.models.io import load_model
from autoden.models.config import NetworkParams, create_network
from corrct.param_tuning import PerfMeterTask, PerfMeterBatch
from corrct.struct_illum import MaskCollection
from numpy.typing import NDArray
from torch.nn import Module

from noise2ghost.algos import INR, N2G, post_process_scale_bias
from noise2ghost.models.config import NetworkParamsINR
from noise2ghost.models.inr import SIREN

SAVE_MODEL_CNN_PATH = Path("./model.pt").expanduser()


@dataclass
class RecParsCNN:
    """Dataclass for storing reconstruction parameters for a CNN model."""

    model: str | Path | Module | NetworkParams = SAVE_MODEL_CNN_PATH
    num_splits: int | None = 4
    num_perms: int = 6
    lower_limit: float | None = None
    epochs: int = 1024 * 3
    lr: float = 3e-4  # https://x.com/karpathy/status/801621764144971776
    optim_algo: str = "adam"
    cv_fraction: float = 0.1
    accum_grads: bool = False


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


def reconstruct_variational(
    masks: NDArray | MaskCollection,
    buckets: NDArray,
    iterations: int = 2000,
    reg: cct.regularizers.BaseRegularizer | None = None,
    verbose: bool = False,
) -> tuple[NDArray, cct.solvers.SolutionInfo | None, PerfMeterTask]:
    """
    Perform variational reconstruction.

    Parameters
    ----------
    masks : NDArray | MaskCollection
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

    if not isinstance(masks, MaskCollection):
        masks = MaskCollection(masks)

    p = cct.struct_illum.ProjectorGhostImaging(masks)

    c1 = perf_counter()

    if reg is None:
        rec = p.fbp(buckets, adjust_scaling=False)
        info = None
    else:
        solver = cct.solvers.PDHG(verbose=verbose, regularizer=reg, leave_progress=False)
        rec, info = solver(p, buckets, iterations=iterations)

    rec = post_process_scale_bias(rec, masks, buckets)

    c2 = perf_counter()

    return rec, info, PerfMeterTask(init_time_s=c1 - c0, exec_time_s=c2 - c1, total_time_s=c2 - c0)


def reconstruct_neural_cnn(
    masks: NDArray,
    buckets: NDArray,
    rec_pars: RecParsCNN = RecParsCNN(),
    reg_val: float | LossRegularizer | None = None,
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

    Returns
    -------
    tuple
        The reconstructed image, training losses, and performance metrics.
    """
    c0 = perf_counter()

    is_n2g = rec_pars.num_splits is not None

    model = _get_model(rec_pars.model)
    solver_n2g = N2G(model=model, reg_val=reg_val)

    inp_recs_trn, tgt_trn_data, _, tgt_cv_data, tgt_trn_inds = solver_n2g.prepare_data(
        masks,
        buckets,
        num_splits=rec_pars.num_splits,
        num_perms=rec_pars.num_perms,
        tst_fraction=0.0,
        cv_fraction=rec_pars.cv_fraction,
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
        accum_grads=rec_pars.accum_grads,
    )
    gi_rec = solver_n2g.infer(inp_recs_trn)
    if is_n2g:
        gi_rec = gi_rec.mean(axis=0)

    gi_rec = post_process_scale_bias(gi_rec, masks, buckets)

    c2 = perf_counter()

    return gi_rec, losses, PerfMeterTask(init_time_s=c1 - c0, exec_time_s=c2 - c1, total_time_s=c2 - c0)


def fit_variational_reg_weight(
    masks: NDArray | MaskCollection,
    buckets: NDArray,
    reg: Callable[[float], cct.regularizers.BaseRegularizer] = cct.regularizers.Regularizer_TV2D,
    lambda_range: tuple[float, float] = (1e-3, 1e2),
    iterations: int = 2000,
    lower_limit: float | None = None,
    num_averages: int = 3,
    parallel_eval: bool | int | Executor = False,
) -> tuple[float, NDArray, PerfMeterBatch]:
    """
    Fit the regularization weight for variational reconstruction.

    Parameters
    ----------
    masks : NDArray | MaskCollection
        The masks used for reconstruction.
    buckets : NDArray
        The bucket data.
    reg : Callable[[float], cct.regularizers.BaseRegularizer], optional
        Function to create a regularizer given a lambda value, by default cct.regularizers.Regularizer_TV2D.
    lambda_range : tuple[float, float], optional
        Range of lambda values to test, by default (1e-3, 1e2).
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

    def solve_reg(lam: float, mask: NDArray | None = None) -> tuple[NDArray, SolutionInfo]:
        solver = cct.solvers.PDHG(regularizer=reg(lam), verbose=solver_verbose, leave_progress=False)
        prj = cct.struct_illum.ProjectorGhostImaging(masks)
        return solver(prj, buckets, iterations=iterations, lower_limit=lower_limit, b_test_mask=mask)

    cv = cct.param_tuning.CrossValidation(
        buckets.shape, num_averages=num_averages, verbose=True, plot_result=True, parallel_eval=parallel_eval
    )
    cv.task_exec_function = solve_reg

    lams = cct.param_tuning.get_lambda_range(*lambda_range, num_per_order=2)
    f_avgs, _, all_info = cv.compute_loss_values(lams, return_all=True)
    lam_min, _ = cv.fit_loss_min(lams, f_avgs)
    print(lam_min)

    c0 = perf_counter()

    rec_reg, _ = solve_reg(lam_min)
    rec_reg = post_process_scale_bias(rec_reg, masks, buckets)

    c1 = perf_counter()
    final_rec_perfs = PerfMeterTask(init_time_s=0.0, exec_time_s=c1 - c0, total_time_s=c1 - c0)

    stats = sum([info[2] for info in all_info], PerfMeterBatch())
    stats.append(final_rec_perfs)

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
    solver_n2g = N2G(model=model, reg_val=None)
    recs_trn_inp, data_trn_tgt, _, data_val_tgt, inds_trn_tgt = solver_n2g.prepare_data(
        masks,
        buckets,
        num_splits=rec_pars.num_splits,
        num_perms=rec_pars.num_perms,
        tst_fraction=0.0,
        cv_fraction=rec_pars.cv_fraction,
    )
    data_sb = deepcopy(solver_n2g.data_sb)

    reg_vals = np.array(reg_vals, ndmin=1)
    results = TempResults(recs_in=recs_trn_inp)
    stats_tasks = []

    cb1 = perf_counter()

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
            accum_grads=rec_pars.accum_grads,
        )
        gi_rec = solver_n2g.infer(recs_trn_inp)
        if is_n2g:
            gi_rec = gi_rec.mean(axis=0)

        gi_rec = post_process_scale_bias(gi_rec, masks, buckets)
        ct2 = perf_counter()

        results.recs_out.append(gi_rec)
        results.losses.append(losses)
        stats_tasks.append(PerfMeterTask(init_time_s=(ct1 - ct0), exec_time_s=(ct2 - ct1), total_time_s=(ct2 - ct0)))

    min_losses = [np.nanmin(losses["loss_tst"]) for losses in results.losses]
    best_rec_ind = np.argmin(min_losses)

    cv = cct.param_tuning.CrossValidation(buckets.shape, verbose=True, plot_result=True)
    min_reg_weight, _ = cv.fit_loss_min(reg_vals, np.array(min_losses))

    ct0 = perf_counter()
    solver_n2g = N2G(model=deepcopy(model), reg_val=min_reg_weight, data_scale_bias=data_sb, device=device)
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

    stats_tasks.append(PerfMeterTask(init_time_s=(ct1 - ct0), exec_time_s=(ct2 - ct1), total_time_s=(ct2 - ct0)))
    cb2 = perf_counter()
    stats_batch = PerfMeterBatch(init_time_s=cb1 - cb0, proc_time_s=cb2 - cb1, total_time_s=cb2 - cb0)

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
    results = TempResults(recs_in=encode_grid.detach().cpu().numpy().copy())
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

    min_losses = [np.nanmin(losses["loss_tst"]) for losses in results.losses]
    best_rec_ind = np.argmin(min_losses)

    cv = cct.param_tuning.CrossValidation(buckets.shape, verbose=True, plot_result=True)
    min_reg_weight, _ = cv.fit_loss_min(reg_vals, np.array(min_losses))

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
    stats_batch = PerfMeterBatch(init_time_s=cb1 - cb0, proc_time_s=cb2 - cb1, total_time_s=cb2 - cb0)

    print(f"INR: Found lowest loss for lambda = {min_reg_weight} (ind: {best_rec_ind})")
    return min_reg_weight, gi_rec, losses, stats_batch
