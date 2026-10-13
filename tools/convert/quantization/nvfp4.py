"""NVFP4 weight quantization: E2M1 codes, E4M3 block scales over 16 elements, one FP32 divisor.

Decoded value = E2M1(code) * E4M3(scale) / divisor (the compressed-tensors semantics). The
divisor maps the tensor's absolute maximum to block scale 256 at block maximum 6, so the
alternative block maximum 4 still fits E4M3 (384 < 448).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BLOCK = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass(frozen=True, slots=True)
class Nvfp4Rows:
    codes: torch.Tensor  # uint8 [N, K/2], low nibble = even element
    scales: torch.Tensor  # uint8 E4M3 words [N, K/16]


def nvfp4_divisor(amax: float) -> float:
    """FP32 divisor for a tensor whose largest magnitude is ``amax``."""
    if not amax > 0.0:
        return 1.0
    return float(torch.tensor(256.0 * 6.0 / amax, dtype=torch.float32))


def _codes(scaled: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude (ties toward the smaller code) with the sign in bit 3."""
    midpoints = torch.tensor(_MIDPOINTS, device=scaled.device, dtype=scaled.dtype)
    index = torch.bucketize(scaled.abs().clamp(max=6.0), midpoints)
    return index.to(torch.uint8) | ((scaled < 0).to(torch.uint8) << 3)


def _decode(codes: torch.Tensor, scales: torch.Tensor, divisor: float) -> torch.Tensor:
    table = torch.tensor(E2M1, device=codes.device, dtype=torch.float32)
    magnitude = table[(codes & 7).long()]
    signed = torch.where(codes >= 8, -magnitude, magnitude)
    return signed * scales / divisor


def quantize_nvfp4_rows(
    values: torch.Tensor, divisor: float, *, mse_46: bool = True, device: str = "cuda"
) -> Nvfp4Rows:
    """Quantize ``[N, K]`` values with a shared ``divisor``.

    Each 16-element block takes scale ``block_max * divisor / M`` rounded to E4M3. With
    ``mse_46`` both M = 6 and M = 4 are tried and the block keeps the one with the lower
    squared reconstruction error; otherwise M = 6 (absmax).
    """
    if values.dim() != 2 or values.shape[1] % BLOCK:
        raise ValueError("NVFP4 rows must be [N, K] with K divisible by 16")
    n, k = values.shape
    blocks = values.to(device=device, dtype=torch.float32).reshape(n, k // BLOCK, BLOCK)
    if not bool(torch.isfinite(blocks).all()):
        raise ValueError("NVFP4 source contains NaN or infinity")
    block_max = blocks.abs().amax(-1, keepdim=True)
    best_codes = best_words = best_error = None
    for limit in (6.0, 4.0) if mse_46 else (6.0,):
        raw = block_max * divisor / limit
        if bool((raw > 448.0).any()):
            raise ValueError("NVFP4 block scale exceeds E4M3; the divisor is too large")
        words = raw.to(torch.float8_e4m3fn)
        scale = words.float()
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.where(
            scale > 0,
            _codes(blocks * divisor / safe),
            torch.zeros_like(blocks, dtype=torch.uint8),
        )
        error = (_decode(codes, scale, divisor) - blocks).pow(2).sum(-1, keepdim=True)
        if best_error is None:
            best_codes, best_words, best_error = codes, words.view(torch.uint8), error
        else:
            pick = error < best_error
            best_codes = torch.where(pick, codes, best_codes)
            best_words = torch.where(pick, words.view(torch.uint8), best_words)
            best_error = torch.where(pick, error, best_error)
    flat = best_codes.reshape(n, k)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    scales = best_words.reshape(n, k // BLOCK)
    return Nvfp4Rows(packed.cpu().contiguous(), scales.cpu().contiguous())
