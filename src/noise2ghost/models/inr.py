from collections.abc import Sequence
from math import sqrt

import torch as pt
import torch.nn as nn


class SirenLayer(nn.Module):
    """A single layer of the SIREN (Sinusoidal Representation Networks) model."""

    def __init__(self, in_f: int, out_f: int, w0: float = 30, is_first: bool = False, is_last: bool = False):
        """
        Initialize a SirenLayer.

        Parameters
        ----------
        in_f : int
            Number of input features.
        out_f : int
            Number of output features.
        w0 : float, optional
            Omega_0 parameter for the SIREN activation function, by default 30.
        is_first : bool, optional
            Whether this is the first layer in the network, by default False.
        is_last : bool, optional
            Whether this is the last layer in the network, by default False.
        """
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the linear layer.
        """
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with pt.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and activation function.
        """
        x = self.linear(x)
        return pt.moveaxis(x, -1, 0) if self.is_last else pt.sin(self.w0 * x)


class SIREN(nn.Sequential):
    """SIREN (Sinusoidal Representation Networks) model."""

    def __init__(
        self,
        n_channels_in: int = 2,
        n_channels_out: int = 1,
        n_embeddings: int = 128,
        n_layers: int = 3,
        n_features: int = 256,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
        """
        Initialize a SIREN model.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels, by default 2.
        n_channels_out : int, optional
            Number of output channels, by default 1.
        n_embeddings : int, optional
            Number of embeddings, by default 128.
        n_layers : int, optional
            Number of layers in the network, by default 3.
        n_features : int, optional
            Number of features in each hidden layer, by default 256.
        device : str, optional
            Device to run the model on, by default "cuda" if CUDA is available, otherwise "cpu".
        """
        layers = [SirenLayer(n_embeddings * n_channels_in, n_features, is_first=True)]
        for _ in range(1, n_layers - 1):
            layers.append(SirenLayer(n_features, n_features))
        layers.append(SirenLayer(n_features, n_channels_out, is_last=True))
        # layers.append(nn.Sigmoid())
        super().__init__(*layers)
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_embeddings = n_embeddings
        self.device = device
        self.to(device)


class PositionalEncoder:
    """Positional Encoder for encoding input grid coordinates."""

    def __init__(
        self,
        num_embeddings: int = 128,
        ndims: int = 2,
        scale: float = 4.0,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
        """
        Initialize a PositionalEncoder.

        Parameters
        ----------
        num_embeddings : int, optional
            Number of embeddings, by default 128.
        ndims : int, optional
            Number of dimensions, by default 2.
        scale : float, optional
            Scale factor for the embeddings, by default 4.0.
        device : str, optional
            Device to run the encoder on, by default "cuda" if CUDA is available, otherwise "cpu".
        """
        self.ndims = ndims
        self.base = pt.randn((num_embeddings, ndims)) * scale
        self.base = self.base.to(device)
        self.device = device

    def embed(self, inp_grid: pt.Tensor):
        """
        Embed the input grid coordinates.

        Parameters
        ----------
        inp_grid : torch.Tensor
            Input grid tensor.

        Returns
        -------
        torch.Tensor
            Embedded grid tensor.
        """
        x_embedding = (2.0 * pt.pi * inp_grid.to(self.device)) @ self.base.t()
        return pt.cat([pt.sin(x_embedding), pt.cos(x_embedding)], dim=-1)

    def create_grid(self, dims: int | Sequence[int]):
        """
        Create a grid of coordinates.

        Parameters
        ----------
        dims : int or Sequence[int]
            Dimensions of the grid. If an integer is provided, it is used for all dimensions.

        Returns
        -------
        torch.Tensor
            Grid tensor.

        Raises
        ------
        ValueError
            If the length of the dimensions list does not match the number of dimensions.
        """
        if isinstance(dims, int):
            dims = [dims] * self.ndims
        elif len(dims) != self.ndims:
            raise ValueError(
                f"List of dimension sizes (#{len(dims)}) should either be scalar or vector of {self.ndims} values."
            )

        return pt.stack(pt.meshgrid([pt.linspace(-0.5, 0.5, steps=d) for d in dims]), dim=-1)
