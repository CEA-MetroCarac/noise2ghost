"""
Data IO module.
"""

from pathlib import Path
from collections.abc import Mapping
from typing import Union

import h5py
import numpy as np
from numpy.typing import NDArray

import torch as pt


def save_model_state(
    save_epochs_dir: Union[str, Path],
    epoch_num: int,
    model_state: Mapping,
    optim_state: Union[Mapping, None] = None,
    is_best: bool = False,
) -> None:
    """Save a model to disk.

    Parameters
    ----------
    save_epochs_dir : str | Path
        The directory where to save the model state
    epoch_num : int
        The epoch number
    model_state : Mapping
        The model state to save
    optim_state : Mapping, optional
        The optimizer state to save, by default None
    is_best : bool, optional
        Whether it is the best fitted model, by default False
    """
    epochs_base_path = Path(save_epochs_dir) / "weights"
    epochs_base_path.mkdir(parents=True, exist_ok=True)

    if is_best:
        pt.save({"epoch": epoch_num, "state_dict": model_state, "optimizer": optim_state}, epochs_base_path / "weights.pt")
    else:
        pt.save(
            {"epoch": epoch_num, "state_dict": model_state, "optimizer": optim_state},
            epochs_base_path / f"weights_epoch_{epoch_num}.pt",
        )


def load_model_state(save_epochs_dir: Union[str, Path], epoch_num: Union[int, None] = None) -> Mapping:
    """Load a model from disk.

    Parameters
    ----------
    save_epochs_dir : str | Path
        The director where the models are saved
    epoch_num : int | None, optional
        The epoch number or if None/-1 the best state will be loaded, by default None

    Returns
    -------
    Mapping
        The loaded model state and possibly an optimizer state.

    Raises
    ------
    ValueError
        When the directory does not exist or the requested model is not available.
    """
    epochs_base_path = Path(save_epochs_dir) / "weights"
    if not epochs_base_path.exists():
        raise ValueError(f"Directory of the model state {epochs_base_path} does not exist!")

    if epoch_num is None or epoch_num == -1:
        state_path = epochs_base_path / "weights.pt"
    else:
        state_path = epochs_base_path / f"weights_epoch_{epoch_num}.pt"
    if not state_path.exists():
        raise ValueError(f"Model state {state_path} does not exist!")

    print(f"Loading state path: {state_path}")
    return pt.load(state_path)


def get_h5_dataset(h5f: h5py.File, path: str) -> h5py.Dataset:
    """Extract a HDF5 dataset from a given file.

    Parameters
    ----------
    h5f : h5py.File
        The file handle
    path : str
        The dataset path

    Returns
    -------
    h5py.Dataset
        The dataset

    Raises
    ------
    ValueError
        In case of wrong path.
    """
    dset = h5f[path]
    if isinstance(dset, h5py.Dataset):
        return dset
    else:
        raise ValueError(f"Path {path} does not point to a dataset")


def get_h5_group(h5f: h5py.File, path: str) -> h5py.Group:
    """Extract a HDF5 group from a given file.

    Parameters
    ----------
    h5f : h5py.File
        The file handle
    path : str
        The group path

    Returns
    -------
    h5py.Group
        The group

    Raises
    ------
    ValueError
        In case of wrong path.
    """
    dset = h5f[path]
    if isinstance(dset, h5py.Group):
        return dset
    else:
        raise ValueError(f"Path {path} does not point to a group")


class DataGI:
    """Handle Ghost Imaging data IO."""

    data_path: Path

    def __init__(self, data_path: Union[Path, str]) -> None:
        self.data_path = Path(data_path).expanduser()

    def save_data(self, masks: NDArray, buckets: NDArray) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_path, mode="w") as hf:
            hf.create_dataset("/patterns", data=masks, compression=9)
            hf.create_dataset("/measurements", data=buckets, compression=9)

    def load_data(self) -> tuple[NDArray, NDArray]:
        with h5py.File(self.data_path, mode="r") as hf:
            masks = np.array(get_h5_dataset(hf, "/patterns")[()])
            buckets = np.array(get_h5_dataset(hf, "/measurements")[()])
        return masks, buckets

    def load_xrf_raw_data(self) -> Mapping:
        with h5py.File(self.data_path, "r") as hf:
            dset_xrf = get_h5_dataset(hf, "/data/xrf")
            dset_dark = get_h5_dataset(hf, "/data/flat_panel/dark")
            dset_beam = get_h5_dataset(hf, "/data/flat_panel/empty_beam")
            dset_flat = get_h5_dataset(hf, "/data/flat_panel/sample")
            dset_mask = get_h5_dataset(hf, "/data/flat_panel/sample_and_masks")

            data_xrf: NDArray = dset_xrf[()]
            data_dark: NDArray = dset_dark[()]
            data_beam: NDArray = dset_beam[()]
            data_flat: NDArray = dset_flat[()]
            data_mask: NDArray = dset_mask[()]

            metadata_bias: NDArray = get_h5_dataset(hf, "/metadata/xrf/bias_keV")[()]
            metadata_gain: NDArray = get_h5_dataset(hf, "/metadata/xrf/gain_keV")[()]
            metadata_xrf_ranges = get_h5_group(hf, "/metadata/xrf/ranges")

            bins_ranges = {el: dset[()] for el, dset in metadata_xrf_ranges.items()}

        return dict(
            xrf=data_xrf,
            darks=data_dark,
            flats=data_flat,
            beams=data_beam,
            masks=data_mask,
            bins_ranges=bins_ranges,
            spectrum_gain=metadata_gain,
            spectrum_bias=metadata_bias,
        )

    def load_preprocessed_xrf_data(self) -> Mapping:
        with h5py.File(self.data_path, "r") as hf:
            dset_xrf = get_h5_dataset(hf, "/data/xrf")
            dset_beam = get_h5_dataset(hf, "/data/flat_panel/empty_beam")
            dset_smpl = get_h5_dataset(hf, "/data/flat_panel/sample")
            dset_mask = get_h5_dataset(hf, "/data/flat_panel/masks")

            data_xrf: NDArray = dset_xrf[()]
            data_beam: NDArray = dset_beam[()]
            data_smpl: NDArray = dset_smpl[()]
            data_mask: NDArray = dset_mask[()]

            metadata_bias: NDArray = get_h5_dataset(hf, "/metadata/xrf/bias_keV")[()]
            metadata_gain: NDArray = get_h5_dataset(hf, "/metadata/xrf/gain_keV")[()]
            metadata_xrf_ranges = get_h5_group(hf, "/metadata/xrf/ranges")

            bins_ranges = {el: dset[()] for el, dset in metadata_xrf_ranges.items()}

            metadata_xrf = get_h5_group(hf, "/metadata")
            metadata = {key: dset[()] for key, dset in metadata_xrf.items() if key != "xrf"}

        return dict(
            xrf=data_xrf,
            flats=data_smpl,
            beams=data_beam,
            masks=data_mask,
            bins_ranges=bins_ranges,
            spectrum_gain=metadata_gain,
            spectrum_bias=metadata_bias,
            **metadata,
        )


class VolumeIO:
    """Handle Ghost Imaging reconstructions IO."""

    data_path: Path

    def __init__(self, data_path: Union[Path, str]) -> None:
        self.data_path = Path(data_path).expanduser()

    def save_training_images(self, images: NDArray, file_format: str = "h5") -> None:
        if file_format.lower() in ("h5", "hdf5"):
            with h5py.File(self.data_path / "training_images.h5", mode="w") as hf:
                hf.create_dataset("/images", data=images, compression=9)
        else:
            raise ValueError(f"Format: '{file_format}' not supported, yet.")

    def load_training_images(self, file_format: str = "h5") -> NDArray:
        if file_format.lower() in ("h5", "hdf5"):
            with h5py.File(self.data_path / "training_images.h5", mode="r") as hf:
                return np.array(get_h5_dataset(hf, "/images")[()])
        else:
            raise ValueError(f"Format: '{file_format}' not supported, yet.")
