"""
Example 01.
"""

import numpy as np
import matplotlib.pyplot as plt

from noise2ghost.testing import create_datasets
from noise2ghost.reconstructions import fit_variational_reg_weight
from noise2ghost.config import NetworkParamsUNet
from noise2ghost.algos import N2G

from corrct.regularizers import Regularizer_TV2D


BASE_SETTINGS = dict(phantom_type="chromosomes", num_splits=4, num_perms=8, n_features=24, epochs=1024 * 8, sampling_ratio=10)
PHYSICS_SETTINGS = dict(photon_density=1e8, readout_noise_std=5)

REG_VAL_DIP = 5e-6
NET_PARS = NetworkParamsUNet(n_features=24, n_levels=3)


if __name__ == "__main__":
    info, volumes, data = create_datasets(
        phantom_type=BASE_SETTINGS["phantom_type"],
        sampling_ratio=BASE_SETTINGS["sampling_ratio"],
        photon_density=PHYSICS_SETTINGS["photon_density"],
        reg_val_tv=None,
        readout_noise_std=PHYSICS_SETTINGS["readout_noise_std"],
    )

    reg_val_tv, volumes["reconstruction_tv"] = fit_variational_reg_weight(
        data["masks"], data["buckets"], reg=Regularizer_TV2D, lambda_range=(0.1, 10)
    )

    solver_n2g = N2G(model=NET_PARS, reg_val=REG_VAL_DIP)
    inp_recs_trn, tgt_trn_data, _, tgt_cv_data, tgt_trn_inds = solver_n2g.prepare_data(
        data["masks"],
        data["buckets"],
        num_splits=BASE_SETTINGS["num_splits"],
        num_perms=BASE_SETTINGS["num_perms"],
        tst_fraction=0.0,
        cv_fraction=0.1,
    )
    _, losses_val = solver_n2g.train(
        inp_recs_trn,
        tgt_trn_data,
        tgt_trn_inds,
        tgt_cv_data,
        epochs=BASE_SETTINGS["epochs"],
        learning_rate=2e-4,
    )
    gi_n2g = solver_n2g.infer(inp_recs_trn).mean(axis=0)

    phantom = volumes["phantom"]
    gi_ls = np.squeeze(volumes["reconstruction_ls"])
    gi_tv = np.squeeze(volumes["reconstruction_tv"])

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[6, 6.5])
    fig.suptitle(f"N. feat: {NET_PARS.n_features}, Reg. lambda: {REG_VAL_DIP}, N. splits: {BASE_SETTINGS['num_splits']}")
    ax[0, 0].imshow(phantom)
    ax[0, 0].set_title("Phantom")
    ax[0, 1].imshow(gi_ls)
    ax[0, 1].set_title("Least-squares")
    ax[1, 0].imshow(np.squeeze(gi_n2g))
    ax[1, 0].set_title(f"N2G best (e: {np.argmin(losses_val)})")
    ax[1, 1].imshow(np.squeeze(gi_tv))
    ax[1, 1].set_title(f"TV(w:{reg_val_tv:.5})")
    fig.tight_layout()
