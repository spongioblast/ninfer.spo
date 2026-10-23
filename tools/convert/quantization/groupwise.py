"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats; this
module only performs the registered numeric transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.artifact.layouts import (
    row_split_geometry,
)
from tools.artifact.formats import QuantFormat, get_format

_FP16_MIN_SUBNORMAL = 2.0**-24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def _canonical_scale_words(
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical binary16 scales and binary32 reciprocals on the host.

    CUDA division is not correctly rounded at every binary16 scale boundary.
    The host oracle performs the specified division in binary64, explicitly
    rounds through binary32 and binary16, then computes the reciprocal in the
    same way.  A binary32 input divided by these small integer denominators has
    enough binary64 precision for the final binary32 rounding to be exact.
    """

    host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def quantize_q4_g32_qat(
    weight: torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Recover the 4-bit group-32 grid already present in a QAT bf16 matrix.

    Each group of 32 values is an integer code in [-8, 7] times one binary16
    scale. The scale is the group's absolute maximum divided by the largest
    code it uses (8 when code -8 is present, otherwise 7 or less). Picking
    that divisor keeps the stored values; dividing by 7 for every group
    quantizes them a second time.
    """

    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise ValueError("q4_g32_qat requires a floating-point rank-2 matrix")
    geometry = row_split_geometry(Q4_G32_NAME, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical
    grouped = logical.reshape(geometry.n, geometry.groups_per_row, 32)
    amax = grouped.abs().amax(dim=2)
    best_codes = best_scales = best_error = None
    for divisor in range(1, 9):
        host_scales, host_reciprocal = _canonical_scale_words(amax / divisor, 1)
        scales = host_scales.to(target)
        reciprocal = host_reciprocal.to(target)
        codes = torch.clamp(
            torch.round(grouped * reciprocal.unsqueeze(-1)), -8, 7
        ).to(torch.int8)
        error = (codes.float() * scales.unsqueeze(-1) - grouped).square().sum(dim=2)
        if best_error is None:
            best_codes, best_scales, best_error = codes, scales, error
        else:
            take = error < best_error
            best_codes = torch.where(take.unsqueeze(-1), codes, best_codes)
            best_scales = torch.where(take, scales, best_scales)
            best_error = torch.where(take, error, best_error)
    return QuantizedMatrix(codes=best_codes, scales=best_scales)


Q4_G32_NAME = "q4_g32_fp16"


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    Scales are rounded to binary16 before codes are selected because those are
    the exact scales consumed after loading.  Padding values are zero and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(
            f"grouped quantization requires rank 2, got {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    logical = weight.detach().to(device=target, dtype=torch.float32)
    if geometry.k_pad != geometry.k:
        physical = torch.zeros(
            (geometry.n, geometry.k_pad), dtype=torch.float32, device=target
        )
        physical[:, : geometry.k].copy_(logical)
        logical = physical

    grouped = logical.reshape(geometry.n, geometry.groups_per_row, spec.group_size)
    max_abs = grouped.abs().amax(dim=2)
    host_scales, host_reciprocal = _canonical_scale_words(max_abs, spec.qmax)
    scales = host_scales.to(target)
    reciprocal = host_reciprocal.to(target)
    codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    return QuantizedMatrix(codes=codes, scales=scales)
