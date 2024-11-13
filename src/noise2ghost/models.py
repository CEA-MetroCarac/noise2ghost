"""Models and losses handling tools."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch as pt
import torch.nn as nn
from torch.cuda import is_available as is_cuda_available
from torch.nn import Module
from numpy.typing import NDArray

from .dncnn import DnCNN
from .msd import MSDnet
from .unet import UNet
from .inr import SIREN, PositionalEncoder


class NetworkParams(ABC):
    """Store network parameters."""

    n_channels_in: int
    n_channels_out: int
    n_features: int

    def __init__(self, n_features: int, n_channels_in: int = 1, n_channels_out: int = 1) -> None:
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_features = n_features

    def __repr__(self) -> str:
        """Produce the string representation of the object.

        Returns
        -------
        str
            The string representation.
        """
        return self.__class__.__name__ + " {\n" + ",\n".join([f"  {k} = {v}" for k, v in self.__dict__.items()]) + "\n}"

    @abstractmethod
    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get the associated model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The model.
        """


class NetworkParamsMSD(NetworkParams):
    """Store MS-D net parameters."""

    dilations: Sequence[int] | NDArray[np.integer]
    n_layers: int

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 80,
        n_features: int = 1,
        dilations: Sequence[int] | NDArray[np.integer] = np.arange(1, 10),
    ) -> None:
        """Initialize the MS-D network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels, by default 1.
        n_channels_out : int, optional
            Number of output channels, by default 1.
        n_layers : int, optional
            Number of layers in the network, by default 80.
        n_features : int, optional
            Number of features, by default 1.
        dilations : Sequence[int] | NDArray[np.integer], optional
            Dilation values for the network, by default np.arange(1, 10).
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.dilations = dilations

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a MS-D net model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The model.
        """
        return MSDnet(
            self.n_channels_in,
            self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            dilations=list(self.dilations),
            device=device,
        )


class NetworkParamsUNet(NetworkParams):
    """Store UNet parameters."""

    n_levels: int

    DEFAULT_LEVELS: int = 3
    DEFAULT_FEATURES: int = 32

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_levels: int = DEFAULT_LEVELS,
        n_features: int = DEFAULT_FEATURES,
        n_channels_skip: int | None = None,
        bilinear: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        """Initialize the UNet network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_levels : int, optional
            Number of levels in the UNet. Default is 3.
        n_features : int, optional
            Number of features in the UNet. Default is 32.
        n_channels_skip : int, optional
            Number of skip connections channels. Default is None.
        bilinear : bool, optional
            Whether to use bilinear interpolation. Default is True.
        pad_mode : str, optional
            Padding mode for convolutional layers. Default is "replicate".
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_levels = n_levels
        self.n_channels_skip = n_channels_skip
        self.bilinear = bilinear
        self.pad_mode = pad_mode

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a U-net model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The U-net model.
        """
        return UNet(
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_features=self.n_features,
            n_levels=self.n_levels,
            n_channels_skip=self.n_channels_skip,
            bilinear=self.bilinear,
            pad_mode=self.pad_mode,
            device=device,
        )


class NetworkParamsDnCNN(NetworkParams):
    """Store DnCNN parameters."""

    n_layers: int

    def __init__(self, n_channels_in: int = 1, n_channels_out: int = 1, n_layers: int = 20, n_features: int = 64) -> None:
        """Initialize the DnCNN network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_layers : int, optional
            Number of layers. Default is 20.
        n_features : int, optional
            Number of features. Default is 64.
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a DnCNN model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The DnCNN model.
        """
        return DnCNN(
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            device=device,
        )


class NetworkParamsINR(NetworkParams):
    """Store INR parameters."""

    n_embeddings: int
    n_layers: int

    def __init__(
        self,
        n_channels_in: int = 2,
        n_channels_out: int = 1,
        n_layers: int = 3,
        n_features: int = 256,
        n_embeddings: int = 256,
    ) -> None:
        """Initialize the DnCNN network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_layers : int, optional
            Number of layers. Default is 8.
        n_features : int, optional
            Number of features. Default is 128.
        n_embeddings : int, optional
            Number of embeddings. Default is 128.
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.n_embeddings = n_embeddings

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a INR model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The INR model.
        """
        return SIREN(
            n_channels_in=self.n_channels_in,
            n_embeddings=self.n_embeddings,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            device=device,
        )


def create_optimizer(
    network: nn.Module, algo: str = "adam", learning_rate: float = 1e-3, weight_decay: float = 1e-2
) -> pt.optim.Optimizer:
    """Instantiates the desire optimizer for the given model.

    Parameters
    ----------
    network : torch.nn.Module
        The network to train
    algo : str, optional
        The requested optimizer, by default "adam"
    learning_rate : float, optional
        The desired learning rate, by default 1e-3
    weight_decay : float, optional
        The desired weight decay, by default 1e-2

    Returns
    -------
    torch.optim.Optimizer
        The chosen optimizer

    Raises
    ------
    ValueError
        In case an unsupported algorithm is requested
    """
    if algo.lower() == "adam":
        return pt.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "sgd":
        return pt.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "rmsprop":
        return pt.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "lbfgs":
        return pt.optim.LBFGS(network.parameters(), lr=learning_rate, max_iter=10000, history_size=50)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def get_num_parameters(model: nn.Module, verbose: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model {model.__class__.__name__} - num. parameters: {num_params}")
    return num_params


def set_parameters(model: nn.Module, values: NDArray, info: Sequence[tuple[str, Sequence[int]]]) -> None:
    # Add error checking
    if len(values) != sum([np.prod(v) for _, v in info]):
        raise ValueError("Inconsistent length of values array and parameters shapes")
    state_dict = model.state_dict()
    # if any([np.array(state_dict[k].shape) != ])
    params_start = 0
    for name, p_shape in info:
        params_end = params_start + np.prod(p_shape)
        state_dict[name][:] = pt.tensor(values[params_start:params_end].reshape(p_shape))
        params_start = params_end


def get_parameters(
    model: nn.Module, parameter_type: Union[str, None] = None, filter_params: bool = True
) -> tuple[NDArray, Sequence[tuple[str, Sequence[int]]]]:
    vals = []
    info = []
    for name, params in model.named_parameters():
        p1 = params.view(-1)
        if parameter_type is None or name.split(".")[-1] == parameter_type.lower():
            vals.append(p1.detach().cpu().numpy().copy().flatten())
            info.append((name, [*params.shape]))
        elif not filter_params:
            vals.append(np.zeros_like(p1.detach().cpu().numpy()).flatten())
            info.append((name, [*params.shape]))
    return np.concatenate(vals), info


def get_gradients(model: nn.Module, flatten: bool = True) -> tuple[NDArray, Sequence[tuple[str, Sequence[int]]]]:
    grads = []
    info = []
    for name, params in model.named_parameters():
        g1 = params.grad.view(-1)
        grad = g1.detach().cpu().numpy().copy()
        if flatten:
            grad = grad.flatten()
        grads.append(grad)
        info.append((name, [*params.shape]))
    return np.concatenate(grads), info
