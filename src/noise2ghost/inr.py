from collections.abc import Sequence

import torch as pt
import torch.nn as nn

from numpy import sqrt


class SirenLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, w0: float = 30, is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self._init_weights()

    def _init_weights(self):
        b = 1 / self.in_f if self.is_first else sqrt(6 / self.in_f) / self.w0
        with pt.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return pt.moveaxis(x, -1, 0) if self.is_last else pt.sin(self.w0 * x)


class SIREN(nn.Sequential):
    def __init__(
        self,
        n_channels_in: int = 2,
        n_channels_out: int = 1,
        n_embeddings: int = 128,
        n_layers: int = 3,
        n_features: int = 256,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
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
    def __init__(
        self,
        num_embeddings: int = 128,
        ndims: int = 2,
        scale: float = 4.0,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
        self.ndims = ndims
        self.base = pt.randn((num_embeddings, ndims)) * scale
        self.base = self.base.to(device)
        self.device = device

    def embed(self, inp_grid: pt.Tensor):
        x_embedding = (2.0 * pt.pi * inp_grid.to(self.device)) @ self.base.t()
        return pt.cat([pt.sin(x_embedding), pt.cos(x_embedding)], dim=-1)

    def create_grid(self, dims: int | Sequence[int]):
        if isinstance(dims, int):
            dims = [dims] * self.ndims
        elif len(dims) != self.ndims:
            raise ValueError(
                f"List of dimension sizes (#{len(dims)}) should either be scalar or vector of {self.ndims} values."
            )

        return pt.stack(pt.meshgrid([pt.linspace(-0.5, 0.5, steps=d) for d in dims]), dim=-1)
