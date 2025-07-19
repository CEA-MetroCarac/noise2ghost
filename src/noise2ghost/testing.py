"""
Testing utilities.
"""

import shutil
from collections.abc import Sequence
from datetime import datetime as dt
from pathlib import Path

import corrct as cct
import numpy as np
import skimage.color as skc
import skimage.data as skd
import skimage.io as skio
import skimage.transform as skt
from numpy.typing import DTypeLike, NDArray
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm.auto import tqdm

from noise2ghost.io import DataGI
from noise2ghost.reconstructions import reconstruct_variational

DATASETS_DIR = Path("data/datasets/").expanduser()


def _create_phantom(
    shape_fov: Sequence[int] | None = None, phantom_type: str = "chromosomes", dtype: DTypeLike = np.float32
) -> tuple[NDArray, NDArray, NDArray]:
    print(f"Creating a new dataset for {phantom_type = }")
    if phantom_type.lower() == "dots":
        shape_fov_tmp = [101, 101]
        phantom = cct.processing.circular_mask(shape_fov_tmp, radius_offset=-45, vol_origin_zxy=[+30, -30])
        phantom += cct.processing.circular_mask(shape_fov_tmp, radius_offset=-40, vol_origin_zxy=[+10, 20])
        phantom += cct.processing.circular_mask(shape_fov_tmp, radius_offset=-30, vol_origin_zxy=[-20, -10])
    elif phantom_type.lower() == "chromosomes":
        phantom = skio.imread("data/chromosomes.png")[:, :, :3]
        phantom = skc.rgb2gray(phantom)
        one = np.ones(phantom.shape)
        phantom = one - phantom
    elif phantom_type.lower() == "ghost":
        phantom = skio.imread("data/ghost.png")[:, :, :3]
        phantom = skc.rgb2gray(phantom)
        one = np.ones(phantom.shape)
        phantom = one - phantom
    elif phantom_type.lower() == "shepp-logan":
        phantom = skd.shepp_logan_phantom()
    else:
        raise ValueError(f"No phantom type called: {phantom_type}")

    if shape_fov is not None:
        phantom = skt.resize(phantom, output_shape=shape_fov, order=1)

    threshold = (phantom.max() + phantom.min()) / 2
    foreground = np.where(phantom > threshold)[0]
    background = np.where(phantom <= threshold)[0]

    return phantom.astype(dtype), foreground, background


def _generate_data(phantom: NDArray, sampling_ratio: float) -> tuple[cct.struct_illum.MaskCollection, NDArray]:
    mc_gen = cct.struct_illum.MaskGeneratorHalfGaussian(phantom.shape)
    mc = mc_gen.generate_collection(buckets_fraction=1 / sampling_ratio)

    projector = cct.struct_illum.ProjectorGhostImaging(mc)
    buckets = projector(phantom)
    print(f"Created new masks and realizations: {buckets.shape = }, {mc.masks_enc.shape = }")

    return mc, buckets


def _get_dataset_filename(info: dict, extension: str = "h5") -> str:
    ph_dens_str = f"{info['photon_density']:.0g}" if info['photon_density'] is not None else "None"
    multip = f"_multi-{info['multiplicity']}" if 'multiplicity' in info and info['multiplicity'] > 1 else ""
    return (
        f"{info['phantom_type']}_FoV-{'-'.join(str(x) for x in info['shape_fov'])}_sampling-ratio-{info['sampling_ratio']}"
        f"_photon-density-{ph_dens_str}_readout-noise-std-{info['readout_noise_std']}{multip}.{extension}"
    )


def compute_noise_level(buckets_clean: NDArray, buckets: NDArray, compute_variances: bool = False) -> tuple[NDArray, NDArray]:
    buckets = np.array(buckets, ndmin=2)
    bkt_cln_std = buckets_clean.std(axis=-1)
    bkt_err_std = np.std(buckets - buckets_clean, axis=-1)
    bkt_err_std_perc = bkt_err_std / bkt_cln_std
    mean_bucket_value = buckets_clean.mean(axis=-1)
    multiplicity = bkt_err_std.size
    b_psnr = np.zeros(multiplicity)
    b_mse = np.zeros(multiplicity)
    for ii in range(multiplicity):
        if buckets_clean.ndim > 1:
            b_psnr[ii] = psnr(buckets_clean[ii], buckets[ii], data_range=(buckets_clean[ii].max() - buckets_clean[ii].min()))
            b_mse[ii] = mse(buckets_clean[ii] - mean_bucket_value[ii], buckets[ii] - mean_bucket_value[ii])
            print(f"Bucket std: {bkt_cln_std[ii]:.3}, error std: {bkt_err_std[ii]:.3} ({bkt_err_std_perc[ii]:.3%})")
            print(f"PSNR: {b_psnr[ii]}, MSE: {b_mse[ii]}")
        else:
            b_psnr[ii] = psnr(buckets_clean, buckets[ii], data_range=(buckets_clean.max() - buckets_clean.min()))
            b_mse[ii] = mse(buckets_clean - mean_bucket_value, buckets[ii] - mean_bucket_value)
            print(f"Bucket std: {bkt_cln_std:.3}, error std: {bkt_err_std[ii]:.3} ({bkt_err_std_perc[ii]:.3%})")
            print(f"PSNR: {b_psnr[ii]}, MSE: {b_mse[ii]}")
    if compute_variances:
        variances = cct.processing.noise.compute_variance_poisson(buckets)
        variances /= variances.mean()
        print(f"{variances.min() = }, {variances.max() = }")
    return b_psnr, b_mse


def create_datasets(
    sampling_ratio: float,
    shape_fov: Sequence[int] | None = None,
    phantom_type: str = "chromosomes",
    photon_density: float | None = None,
    readout_noise_std: float | None = None,
    multiplicity: int = 1,
    reg_val_tv: float | None = None,
    compute_ls: bool = True,
    save: bool = False,
    overwrite: bool = False,
) -> tuple[dict, dict, dict]:
    print("Creating phantom")
    phantom, foreground, background = _create_phantom(shape_fov, phantom_type=phantom_type)
    if shape_fov is None:
        shape_fov = phantom.shape
    info: dict = dict(
        shape_fov=shape_fov,
        phantom_type=phantom_type,
        sampling_ratio=sampling_ratio,
        photon_density=photon_density,
        readout_noise_std=readout_noise_std,
        multiplicity=multiplicity,
    )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    dset_fname = DATASETS_DIR / _get_dataset_filename(info)
    dset_fname = dset_fname.expanduser()

    if overwrite or not dset_fname.exists():
        print("Creating NEW data")
        mc, buckets_clean = _generate_data(phantom, sampling_ratio)
        print(f"Created {mc.num_buckets} masks (shape: {mc.masks_enc.shape}) and {len(buckets_clean)} corresponding buckets.")

        all_buckets = []
        for _ in range(multiplicity):
            buckets = buckets_clean.copy()

            if photon_density is not None:
                buckets, _, _ = cct.testing.add_noise(buckets, num_photons=photon_density, add_poisson=True)
                buckets /= photon_density

            if readout_noise_std is not None:
                buckets += readout_noise_std * np.random.randn(buckets.shape[0])

            all_buckets.append(buckets)

        buckets = np.stack(all_buckets, axis=0)

        masks = mc.masks_enc
    else:
        print(f"Loading existing data from file: {dset_fname}")
        fid = DataGI(dset_fname)
        masks, buckets = fid.load_data()
        buckets = np.array(buckets, ndmin=2)
        if multiplicity != buckets.shape[0]:
            raise ValueError(f"Inconsistent multiplicity in dataset! Expected {multiplicity}, and found {buckets.shape[0]}")

        if masks.ndim > 3:
            all_buckets_clean = []
            for m in masks:
                mc = cct.struct_illum.MaskCollection(m)
                prj = cct.struct_illum.ProjectorGhostImaging(mc)
                all_buckets_clean.append(prj(phantom))
            buckets_clean = np.stack(all_buckets_clean, axis=0)
        else:
            mc = cct.struct_illum.MaskCollection(masks)
            prj = cct.struct_illum.ProjectorGhostImaging(mc)
            buckets_clean = prj(phantom)

    info["psnr"], info["mse"] = compute_noise_level(buckets_clean, buckets)

    rec_ls = None
    if compute_ls:
        recs_ls = [reconstruct_variational(masks=mc, buckets=bs) for bs in tqdm(buckets, desc="LS reconstructions")]
        rec_ls = np.stack([rec for rec, _ in recs_ls], axis=0)

    rec_tv = None
    if reg_val_tv is not None:
        recs_tv = [
            reconstruct_variational(masks=mc, buckets=bs, reg=cct.regularizers.Regularizer_TV2D(reg_val_tv), verbose=True)
            for bs in tqdm(buckets, desc="LS-TV reconstructions")
        ]
        rec_tv = np.stack([rec for rec, _ in recs_tv], axis=0)

    volumes = dict(
        phantom=phantom, foreground=foreground, background=background, reconstruction_ls=rec_ls, reconstruction_tv=rec_tv
    )

    if save and (overwrite or not dset_fname.exists()):
        dset_fname.parent.mkdir(parents=True, exist_ok=True)
        fid = DataGI(dset_fname)
        fid.save_data(masks=masks, buckets=buckets)

    return info, volumes, dict(masks=masks, buckets=buckets)


def save_results(info: dict, recs: dict, reg_vals: dict | None = None, save_old: bool = True) -> None:
    result_dir = DATASETS_DIR / "results_N2G"
    result_dir.mkdir(parents=True, exist_ok=True)

    results_fname = result_dir / _get_dataset_filename(info, extension="npz")
    results_fname.expanduser()

    if results_fname.exists() and save_old:
        dst = results_fname.with_stem(results_fname.stem + f"_{dt.now().isoformat()}")
        shutil.move(results_fname, dst)

    results = recs.copy()
    if reg_vals is not None:
        for key, val in reg_vals.items():
            results[f"reg_val_{key}"] = val
    np.savez_compressed(results_fname, **results)


def load_results(info: dict, use_external_gidc: bool = False, use_external_sup: bool = False) -> tuple[dict, dict]:
    results_fname = _get_dataset_filename(info, extension="npz")
    results_fpath = DATASETS_DIR / "results_N2G" / results_fname
    results_fpath.expanduser()

    results = dict(**np.load(results_fpath))
    res_recs = {key: val for key, val in results.items() if "reg_val_" not in key}

    if use_external_gidc:
        results_fpath = DATASETS_DIR / "results_GIDC" / results_fname
        results_fpath.expanduser()
        res_recs["gi_gidc"] = np.load(results_fpath)["rec"]

    if use_external_sup:
        results_fpath = DATASETS_DIR / "results_sup" / results_fname
        results_fpath.expanduser()
        res_recs["gi_sup"] = np.load(results_fpath)["rec"]

    return res_recs, {key: results[f"reg_val_{key}"] for key in res_recs.keys() if f"reg_val_{key}" in results}


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
