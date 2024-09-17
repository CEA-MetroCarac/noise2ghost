#!/usr/bin/env python3
"""
Created on Fri Dec  2 13:30:03 2022

@author: manni
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import corrct as cct
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.legend import Legend
from numpy.typing import NDArray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from noise2ghost.algos import N2G
from noise2ghost.models import NetworkParams, NetworkParamsUNet, Module
from noise2ghost.reconstructions import get_reg_value
from noise2ghost.testing import create_datasets, load_results, save_results


def make_histograms(
    recs: Sequence[NDArray],
    labels: Sequence[str],
    foreground: Optional[NDArray] = None,
    background: Optional[NDArray] = None,
    num_bins: int = 125,
    hist_range: tuple[float, float] = (-0.75, 1.75),
):
    ref = recs[0]

    mean_ref = ref.mean()
    if foreground is None:
        foreground = recs[0] > mean_ref
    if background is None:
        background = recs[0] <= mean_ref

    _, bin_edges = np.histogram(ref, num_bins, range=hist_range)
    bin_cents = (bin_edges[:-1] + bin_edges[1:]) / 2

    def make_single_hist(rec: NDArray) -> Sequence[NDArray]:
        hist_rec, _ = np.histogram(rec, num_bins, range=hist_range)
        hist_rec_fg, _ = np.histogram(rec[foreground], num_bins, range=hist_range)
        hist_rec_bg, _ = np.histogram(rec[background], num_bins, range=hist_range)
        return hist_rec, hist_rec_fg, hist_rec_bg

    fig, axs = plt.subplots(1, len(recs), sharex=True, sharey=False, figsize=[len(recs) * 3, 3])

    for ii, rec in enumerate(recs):
        hist_rec, hist_rec_fg, hist_rec_bg = make_single_hist(rec)

        axs[ii].plot(bin_cents, hist_rec)
        axs[ii].plot(bin_cents, hist_rec_fg)
        axs[ii].plot(bin_cents, hist_rec_bg)
        axs[ii].set_title(labels[ii])
        axs[ii].grid()

    fig.tight_layout()
    plt.show(block=False)


def reconstruct_n2g(
    masks: NDArray,
    buckets: NDArray,
    epochs: int,
    num_splits: Optional[int],
    num_perms: int,
    model: NetworkParams | Module | str,
    reg_val: float,
    lr: float = 2.5e-4,
    lower_limit: Optional[float] = None,
    algo: str = "adam",
) -> tuple:
    solver_n2g = N2G(model=model, reg_tv_val=reg_val)
    inp_recs_trn, tgt_trn_data, _, tgt_cv_data, tgt_trn_inds = solver_n2g.prepare_data(
        masks, buckets, num_splits=num_splits, num_perms=num_perms, tst_fraction=0.0, cv_fraction=0.1
    )
    losses = solver_n2g.train(
        inp_recs_trn,
        tgt_trn_data,
        tgt_trn_inds,
        tgt_cv_data,
        epochs=epochs,
        learning_rate=lr,
        lower_limit=lower_limit,
        algo=algo,
    )
    gi_n2g = solver_n2g.infer(inp_recs_trn).mean(axis=0)

    return gi_n2g, solver_n2g, losses


def main_n2g_gi(
    phantom_type: str,
    num_splits: int,
    num_perms: int,
    n_features: int,
    epochs: int,
    sampling_ratio: float,
    photon_density: Optional[float],
    readout_noise_std: Optional[float],
    lower_limit: Optional[float] = None,
    shape_fov: Optional[Sequence[int]] = None,
    save: bool = True,
) -> Sequence:
    info, volumes, data = create_datasets(
        phantom_type=phantom_type,
        shape_fov=shape_fov,
        sampling_ratio=sampling_ratio,
        photon_density=photon_density,
        reg_val_tv=None,
        readout_noise_std=readout_noise_std,
        save=save,
        overwrite=False,  # Let's make sure we always use the same!
    )

    reg_val_tv, volumes["reconstruction_tv"] = get_reg_value(
        data["masks"], data["buckets"], reg=cct.regularizers.Regularizer_TV2D, lambda_range=[0.1, 10], lower_limit=lower_limit
    )

    if phantom_type.lower() == "chromosomes":
        if readout_noise_std is None:
            # reg_val = 2e-6  # Poisson noise 1e2
            reg_val = 3e-5  # Poisson noise 2e0
        else:
            reg_val = 5e-6  # Gaussian noise
    elif phantom_type.lower() == "shepp-logan":
        # reg_val = 0.5e-5  # Poisson noise 1e2
        reg_val = 2.5e-5  # Poisson noise 2e0
    elif phantom_type.lower() == "ghost":
        reg_val = 3e-6
    else:
        reg_val = 5e-6

    net_pars = NetworkParamsUNet(n_features=n_features, bilinear=True, n_levels=3)

    gi_gidc, solver_gidc, losses_gidc = reconstruct_n2g(
        data["masks"],
        data["buckets"],
        epochs=epochs,
        num_splits=None,
        num_perms=1,
        model=net_pars,
        reg_val=reg_val,
        lower_limit=lower_limit,
    )

    gi_n2g, solver_n2g, losses_n2g = reconstruct_n2g(
        data["masks"],
        data["buckets"],
        epochs=epochs,
        num_splits=num_splits,
        num_perms=num_perms,
        model=net_pars,
        reg_val=reg_val,
        lower_limit=lower_limit,
    )

    phantom = volumes["phantom"]
    gi_ls = np.squeeze(volumes["reconstruction_ls"])
    gi_tv = np.squeeze(volumes["reconstruction_tv"])

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[6, 6.5])
    fig.suptitle(f"N. feat: {net_pars.n_features}, Reg. lambda: {solver_n2g.reg_val}, N. perms: {num_perms}")
    ax[0, 0].imshow(phantom)
    ax[0, 0].set_title("Phantom")
    ax[0, 1].imshow(gi_ls)
    ax[0, 1].set_title("Least-squares")
    ax[1, 0].imshow(np.squeeze(gi_n2g))
    ax[1, 0].set_title(f"N2G best (e: {np.argmin(losses_n2g[1])})")
    ax[1, 1].imshow(np.squeeze(gi_tv))
    ax[1, 1].set_title(f"TV(w:{reg_val_tv:.5})")
    fig.tight_layout()

    # p2 = (gi_ls, phantom)
    # p3 = (gi_tv, phantom)
    # p5 = (gi_n2g, phantom)
    # labels = ["Least-squares", "TV", "N2G"]
    # cct.processing.post.plot_frcs(volume_pairs=[p2, p3, p5], labels=labels)

    # background = volumes["background"]
    # foreground = volumes["foreground"]
    # make_histograms([phantom, gi_ls, gi_tv, gi_n2g], labels=["Phantom", *labels], foreground=foreground, background=background)

    plt.show(block=False)

    # Saving results
    recs = dict(phantom=phantom, gi_ls=gi_ls, gi_tv=gi_tv, gi_gidc=gi_gidc, gi_n2g=gi_n2g)
    if save:
        save_results(info, recs)

    vols = [gi_ls, gi_tv, gi_gidc, gi_n2g]
    labs = ["Least-squares", "TV", "GIDC", "N2G"]
    compute_metrics(phantom, volumes=vols, labels=labs)

    return solver_n2g, data, recs


def fit_scales_and_biases(volumes: dict, data: dict) -> dict:
    mc = cct.struct_illum.MaskCollection(data["masks"])
    prj = cct.struct_illum.ProjectorGhostImaging(mc)

    data_sb = dict()
    for key, vol in volumes.items():
        if key.lower() == "phantom":
            print(f"Leaving {key} alone")
            data_sb[key] = vol
            continue

        scale, bias = cct.processing.post.fit_scale_bias(vol, data["buckets"], prj)
        print(f"{key}: {scale = }, {bias = }")
        data_sb[key] = vol * scale + bias

    return data_sb


def make_figure(
    img: NDArray,
    scale_pix2inch: float = 0.05,
    cmap: Optional[str] = None,
    border: float = 0.0,
    clim: Optional[tuple[float, float]] = None,
    cbar: bool = False,
):
    fig_size = np.flip(img.shape) * scale_pix2inch
    print(f"Figure size: {fig_size}")
    axs_size = np.array(fig_size, copy=True)
    axs_size = np.array(fig_size, copy=True)
    if cbar:
        fraction: float = 0.17
        pad: float = 0.02
        cbar_fig_fraction = fraction + pad
        cbar_axs_fraction = fraction / 5 + pad
        cbar_ext_axis = 0
        fig_size[cbar_ext_axis] /= 1 - cbar_fig_fraction
        axs_size[cbar_ext_axis] /= 1 - cbar_axs_fraction

        cbar_offset = 0
        print(f"Figure size (after color map): {fig_size}")
    else:
        cbar_offset = 0

    fig = plt.figure(None, figsize=tuple(fig_size))
    img = np.squeeze(img)

    rect = [*((border + cbar_offset) / fig_size), *(axs_size / fig_size - 2 * border / fig_size)]

    print(f"Axes size: {rect}")
    axs = fig.add_axes(rect, label="image")
    axs.set_facecolor((0, 0, 0))
    imo = axs.imshow(img, cmap=cmap)
    if clim is not None:
        imo.set_clim(*clim)
    imo.set_clim()
    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    axs.spines["left"].set_visible(False)

    if cbar:
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size=f"{fraction/5:%}", pad=f"{pad:%}")
        cb = plt.colorbar(imo, cax=cax, orientation="vertical", extend="both")
        cb.ax.tick_params(labelsize=36)

    fig.tight_layout()
    plt.show(block=False)

    return fig, axs


def compute_metrics(
    phantom: NDArray, volumes: Sequence[NDArray], labels: Sequence[str], verbose: bool = True
) -> dict[str, NDArray]:
    result = dict(
        ssims=np.array([ssim(phantom, vol, data_range=(phantom.max() - phantom.min())) for vol in volumes]),
        psnrs=np.array([psnr(phantom, vol, data_range=(phantom.max() - phantom.min())) for vol in volumes]),
        mses=np.array([mse(phantom, vol) for vol in volumes]),
    )
    if verbose:
        print("PSNRs:")
        for lab, ssim_vol in zip(labels, result["psnrs"]):
            print(f"- {lab:<16}: {ssim_vol:.4}")
        print("SSIMs:")
        for lab, ssim_vol in zip(labels, result["ssims"]):
            print(f"- {lab:<16}: {ssim_vol:.4}")
        print("MSEs:")
        for lab, mse_vol in zip(labels, result["mses"]):
            print(f"- {lab:<16}: {mse_vol:.4}")

    # for lab in labels:
    #     print(f" & {lab}", end="", flush=True)
    # print(r" \\")
    # print(r"\hline")
    # for key, values in result.items():
    #     print(key.upper(), end="", flush=True)
    #     for val in values:
    #         print(f" & {val:.4f}", end="", flush=True)
    #     print(r" \\")
    #     print(r"\hline")

    return result


def do_all_figures(
    prefix: Union[str, Path],
    phantom: NDArray,
    gi_ls: NDArray,
    gi_tv: NDArray,
    gi_n2g: NDArray,
    gi_gidc: Optional[NDArray] = None,
    gi_sup: Optional[NDArray] = None,
    di_pb: Optional[NDArray] = None,
    save: bool = True,
):
    save_exts = ("png", "eps")
    prefix = Path(prefix)

    figures_path = Path(".") / "results" / "n2g" / prefix.parent
    figures_path = figures_path.expanduser()
    print(f"{figures_path = }")
    figures_path.mkdir(parents=True, exist_ok=True)

    prefix = prefix.name

    volumes = [gi_ls, gi_tv]
    labels = ["LS", "TV"]
    if gi_sup is not None:
        volumes.append(gi_sup)
        labels.append("Supervised")
    if gi_gidc is not None:
        volumes.append(gi_gidc)
        labels.append("GIDC")
    if di_pb is not None:
        volumes.insert(0, di_pb)
        labels.insert(0, "Pencil-beam")
    volumes.append(gi_n2g)
    labels.append("N2G")
    volume_pairs = [(vol, phantom) for vol in volumes]

    ph_plots = make_figure(phantom, cbar=True)
    fig_ph: Figure = ph_plots[0]
    if save:
        for ext in save_exts:
            fig_file = figures_path / (prefix + "_phantom." + ext)
            fig_ph.savefig(str(fig_file))
            print(f"Saved to: {fig_file}")

    for lab, vol in zip(labels, volumes):
        vol_plot = make_figure(vol, clim=(phantom.min(), phantom.max()))
        if save:
            fig_vol: Figure = vol_plot[0]
            for ext in save_exts:
                fig_file = figures_path / (prefix + f"_{lab.lower()}.{ext}")
                fig_vol.savefig(str(fig_file))
                print(f"Saved to: {fig_file}")

    # We repeat the for loop, because there is too much output from the previous one
    compute_metrics(phantom, volumes=volumes, labels=labels)

    frc_plots = cct.processing.post.plot_frcs(volume_pairs=volume_pairs, labels=labels)  # , snrt=0.4142
    axs_frcs: Axes = frc_plots[1]
    legend_frcs: Legend = axs_frcs.get_legend()
    if legend_frcs is not None:
        for txt in legend_frcs.get_texts():
            txt.set_fontsize(18)
        legend_frcs.get_title().set_fontsize(14)
    axs_frcs.xaxis.get_label().set_fontsize(24)
    for txt in axs_frcs.xaxis.get_ticklabels():
        txt.set_fontsize(20)
    axs_frcs.yaxis.get_label().set_fontsize(24)
    for txt in axs_frcs.yaxis.get_ticklabels():
        txt.set_fontsize(20)
    for line in axs_frcs.get_lines():
        line.set_linewidth(2)
    fig_frcs: Figure = frc_plots[0]
    fig_ph_size = fig_ph.get_size_inches()
    # fig_frcs.set_size_inches(fig_ph_size[0] * 1.76, fig_ph_size[1])
    fig_frcs.set_size_inches(fig_ph_size[0] * 2, fig_ph_size[1] / 1.2)
    fig_frcs.tight_layout()
    fig_frcs.canvas.draw()
    if save:
        for ext in save_exts:
            fig_file = figures_path / (prefix + "_frc." + ext)
            fig_frcs.savefig(str(fig_file))
            print(f"Saved to: {fig_file}")


if __name__ == "__main__":
    BASE_SETTINGS = dict(
        phantom_type="chromosomes", num_splits=4, num_perms=8, n_features=24, epochs=1024 * 8, sampling_ratio=10
    )
    # PHYSICS_SETTINGS = dict(photon_density=1e2, readout_noise_std=None)
    # PHYSICS_SETTINGS = dict(photon_density=2e0, readout_noise_std=None)
    PHYSICS_SETTINGS = dict(photon_density=1e8, readout_noise_std=5)

    solver_n2g, gi_data, gi_recs = main_n2g_gi(**BASE_SETTINGS, **PHYSICS_SETTINGS, save=False)
    save_prefix = (
        f"abstract/{BASE_SETTINGS['phantom_type']}"
        f"_samp-ratio-{BASE_SETTINGS['sampling_ratio']}"
        f"_photons-{PHYSICS_SETTINGS['photon_density']:.0g}"
        f"_readout-{PHYSICS_SETTINGS['readout_noise_std']}"
    )
    # do_all_figures(save_prefix, **gi_recs)
