"""
Created on Tue Jun 20 15:46:33 2023

@author: manni
"""

from typing import Callable, Optional, Union

import corrct as cct
from corrct.struct_illum import MaskCollection
from numpy.typing import NDArray


def get_reconstruction(
    masks: Union[NDArray, MaskCollection],
    buckets: NDArray,
    iterations: int = 2000,
    reg: Union[cct.regularizers.BaseRegularizer, None] = None,
    verbose: bool = False,
) -> NDArray:
    if not isinstance(masks, MaskCollection):
        masks = MaskCollection(masks)

    p = cct.struct_illum.ProjectorGhostImaging(masks)

    if reg is None:
        return p.fbp(buckets, adjust_scaling=False)
    else:
        solver = cct.solvers.PDHG(verbose=verbose, regularizer=reg, leave_progress=False)
        rec_reg, _ = solver(p, buckets, iterations=iterations)

        return rec_reg


def get_reg_value(
    masks: Union[NDArray, MaskCollection],
    buckets: NDArray,
    reg: Callable[[float], cct.regularizers.BaseRegularizer],
    lambda_range: tuple[float, float] = (1e-3, 1e2),
    iterations: int = 2000,
    lower_limit: Optional[float] = None,
    num_averages: int = 3,
) -> tuple[float, NDArray]:
    if not isinstance(masks, MaskCollection):
        masks = MaskCollection(masks)

    p = cct.struct_illum.ProjectorGhostImaging(masks)

    def solver_spawn(lam: float):
        return cct.solvers.PDHG(regularizer=reg(lam), verbose=True, leave_progress=False)

    def solver_call(solver: cct.solvers.Solver, mask: Optional[NDArray] = None):
        return solver(p, buckets, iterations=iterations, lower_limit=lower_limit, b_test_mask=mask)

    cv = cct.param_tuning.CrossValidation(
        buckets.shape, num_averages=num_averages, verbose=True, plot_result=True, parallel_eval=False
    )
    cv.solver_spawning_function = solver_spawn
    cv.solver_calling_function = solver_call

    lams = cv.get_lambda_range(*lambda_range, num_per_order=2)
    f_avgs, _, _ = cv.compute_loss_values(lams)
    lam_min, _ = cv.fit_loss_min(lams, f_avgs)

    solver = solver_spawn(lam_min)
    rec_reg, _ = solver_call(solver)

    print(lam_min)
    return lam_min, rec_reg
