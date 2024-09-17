"""
Data losses definitions.
"""

import torch as pt
import torch.nn as nn


def _differentiate(inp: pt.Tensor, dim: int) -> pt.Tensor:
    diff = pt.diff(inp, 1, dim=dim)
    return pt.concatenate((diff, inp.index_select(index=pt.tensor(inp.shape[dim] - 1, device=inp.device), dim=dim)), dim=dim)


class LossRegularizer(nn.MSELoss):
    """Base class for the regularizer losses."""


class LossTV(LossRegularizer):
    """Total Variation loss function."""

    def __init__(
        self,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        ndims: int = 2,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.ndims = ndims

    def _check_input_tensor(self, img: pt.Tensor) -> None:
        if img.ndim != (2 + self.ndims):
            raise RuntimeError(
                f"Expected input `img` to be a {self.ndims + 2}D tensor (for a {self.ndims} image)"
                f", but got {img.ndim}D (shape: {img.shape})"
            )

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        self._check_input_tensor(img)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim) for dim in range(-self.ndims, 0)]
        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)

        return self.lambda_val * tv_val.sum(axes).mean()


class LossTGV(LossTV):
    """Total Generalized Variation loss function."""

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        self._check_input_tensor(img)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim) for dim in range(-self.ndims, 0)]
        diffdiffs = [_differentiate(d, dim=dim) for dim in range(-self.ndims, 0) for d in diffs]

        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            jac_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffdiffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            jac_val = pt.stack([d.abs() for d in diffdiffs], dim=0).sum(dim=0)

        return self.lambda_val * (tv_val.sum(axes).mean() + jac_val.sum(axes).mean() / 4)


class LossHaarL1(LossRegularizer):
    """Single-level Haar wavelet loss function."""

    def __init__(
        self, lambda_val: float, size_average=None, reduce=None, reduction: str = "mean", isotropic: bool = True
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        if img.ndim != 4:
            raise RuntimeError(f"Expected input `img` to be a 4D tensor, but got {img.shape}")
        axes = [-3, -2, -1]

        kern_hf = pt.tensor([[[[1.0, -1.0]]]])

        kern_u_hf = pt.tensor(kern_hf, dtype=img.dtype, device=img.device) / pt.sqrt(pt.tensor(2.0))
        kern_v_hf = kern_u_hf.swapaxes(-2, -1)
        kern_u_lf = kern_u_hf.abs()
        kern_v_lf = kern_v_hf.abs()

        wl_ll = nn.functional.conv2d(img, kern_u_lf * kern_v_lf, padding=1)
        wl_lh = nn.functional.conv2d(img, kern_u_lf * kern_v_hf, padding=1)
        wl_hl = nn.functional.conv2d(img, kern_u_hf * kern_v_lf, padding=1)
        wl_hh = nn.functional.conv2d(img, kern_u_hf * kern_v_hf, padding=1)
        if self.isotropic:
            wl_val = pt.sqrt(pt.pow(wl_hl, 2) + pt.pow(wl_lh, 2)).sum(axes) + wl_ll.abs().sum(axes) + wl_hh.abs().sum(axes)
        else:
            wl_val = wl_hl.abs().sum(axes) + wl_lh.abs().sum(axes) + wl_ll.abs().sum(axes) + wl_hh.abs().sum(axes)

        return self.lambda_val * wl_val.mean()


class LossHaarNL1(LossRegularizer):
    def __init__(
        self,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        levels: int = 2,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.levels = levels

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute wavelet decomposition on current batch."""
        if img.ndim != 4:
            raise RuntimeError(f"Expected input `img` to be a 4D tensor, but got {img.shape}")
        axes = [-3, -2, -1]

        kern_u_hf = pt.tensor([[[[-1.0, 1.0]]]], dtype=img.dtype, device=img.device) / pt.sqrt(pt.tensor(2.0))
        kern_v_hf = pt.tensor([[[[-1.0], [1.0]]]], dtype=img.dtype, device=img.device) / pt.sqrt(pt.tensor(2.0))
        kern_u_lf = kern_u_hf.abs()
        kern_v_lf = kern_v_hf.abs()

        tmp_img = img

        wl_val = []
        for lvl in range(1, self.levels + 1):
            wl_lh = nn.functional.conv2d(tmp_img, kern_u_lf * kern_v_hf, padding=lvl, dilation=lvl)
            wl_hl = nn.functional.conv2d(tmp_img, kern_u_hf * kern_v_lf, padding=lvl, dilation=lvl)
            wl_hh = nn.functional.conv2d(tmp_img, kern_u_hf * kern_v_hf, padding=lvl, dilation=lvl)
            wl_ll = nn.functional.conv2d(tmp_img, kern_u_lf * kern_v_lf, padding=lvl, dilation=lvl)

            wl_val.append(wl_hh.abs().sum(axes))
            if self.isotropic:
                wl_val.append(pt.sqrt(pt.pow(wl_hl, 2) + pt.pow(wl_lh, 2)).sum(axes))
            else:
                wl_val.append(wl_hl.abs().sum(axes) + wl_lh.abs().sum(axes))

            if lvl == self.levels:
                wl_val.append(wl_ll.abs().sum(axes))
            else:
                tmp_img = wl_ll

        return self.lambda_val * pt.tensor(wl_val).sum(dim=0).mean()
