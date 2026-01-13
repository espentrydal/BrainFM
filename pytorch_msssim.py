"""
Lightweight drop-in replacement for the ``pytorch_msssim`` package.

Based on https://github.com/VainF/pytorch-msssim (MIT License).
Only the ``ssim`` and ``ms_ssim`` entry points we use in BrainFM are
implemented here to avoid pip installs inside the pixi-managed env.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["ssim", "ms_ssim"]

_DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _check_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape != y.shape:
        raise ValueError(f"Expected matching shapes, got {x.shape} and {y.shape}")
    if x.dim() not in (4, 5):
        raise ValueError("SSIM expects 4D (N,C,H,W) or 5D (N,C,D,H,W) tensors.")
    if x.size(1) != y.size(1):
        raise ValueError("Channel count must match for SSIM computation.")


def _gaussian_window(
    window_size: int,
    sigma: float,
    channel: int,
    dims: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    if dims == 2:
        kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
        shape = (channel, 1, window_size, window_size)
    else:
        kernel = (
            gauss[:, None, None]
            * gauss[None, :, None]
            * gauss[None, None, :]
        ).unsqueeze(0).unsqueeze(0)
        shape = (channel, 1, window_size, window_size, window_size)
    window = kernel.expand(shape).contiguous()
    return window


def _reduce_map(ssim_map: torch.Tensor, size_average: bool) -> torch.Tensor:
    if size_average:
        return ssim_map.mean()
    reduce_dims = tuple(range(1, ssim_map.dim()))
    return ssim_map.mean(dim=reduce_dims)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    full: bool = False,
    weights: Optional[Sequence[float]] = None,
    K: Tuple[float, float] = (0.01, 0.03),
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Structural Similarity Index (SSIM) for 2D or 3D images.

    Parameters match ``pytorch_msssim.ssim`` for drop-in compatibility.
    """
    _check_inputs(x, y)
    dims = x.dim() - 2
    conv = F.conv2d if dims == 2 else F.conv3d
    channel = x.size(1)
    window = _gaussian_window(
        win_size, win_sigma, channel, dims, x.device, x.dtype
    )
    padding = win_size // 2

    mu1 = conv(x, window, padding=padding, groups=channel)
    mu2 = conv(y, window, padding=padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(x * x, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = conv(y * y, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = conv(x * y, window, padding=padding, groups=channel) - mu1_mu2

    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) * cs_map) / (mu1_sq + mu2_sq + C1)

    if full:
        return _reduce_map(ssim_map, size_average), _reduce_map(cs_map, size_average)
    return _reduce_map(ssim_map, size_average)


def ms_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Multi-scale SSIM following https://doi.org/10.1109/TIP.2003.819861.
    """
    _check_inputs(x, y)
    dims = x.dim() - 2
    pool = F.avg_pool2d if dims == 2 else F.avg_pool3d
    weights_tensor = torch.tensor(
        weights if weights is not None else _DEFAULT_WEIGHTS,
        dtype=x.dtype,
        device=x.device,
    )

    mssim_vals = []
    mcs_vals = []
    for idx in range(weights_tensor.numel()):
        ssim_val, cs_val = ssim(
            x,
            y,
            data_range=data_range,
            size_average=False,
            win_size=win_size,
            win_sigma=win_sigma,
            full=True,
        )
        mssim_vals.append(ssim_val)
        mcs_vals.append(torch.clamp(cs_val, min=1e-6))
        if idx < weights_tensor.numel() - 1:
            x = pool(x, kernel_size=2, stride=2, padding=0)
            y = pool(y, kernel_size=2, stride=2, padding=0)

    mssim_stack = torch.stack(mssim_vals, dim=0)
    mcs_stack = torch.stack(mcs_vals[:-1], dim=0)

    pow_mcs = mcs_stack ** weights_tensor[:-1].view(-1, 1)
    mssim_term = mssim_stack[-1] ** weights_tensor[-1]
    ms_ssim_map = torch.prod(pow_mcs, dim=0) * mssim_term
    if size_average:
        return ms_ssim_map.mean()
    return ms_ssim_map
