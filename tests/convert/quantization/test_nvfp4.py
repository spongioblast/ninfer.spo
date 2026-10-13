from __future__ import annotations

import json

import numpy as np
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import nvfp4_divisor, quantize_nvfp4_rows
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource, tensor_source

TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def decode(codes: torch.Tensor, scales: torch.Tensor, divisor: float, k: int) -> np.ndarray:
    """FP64 decode with the compressed-tensors rule value = code * scale / divisor."""
    packed = codes.numpy().astype(np.int64)
    nibbles = np.stack((packed & 15, packed >> 4), axis=-1).reshape(packed.shape[0], k)
    magnitude = TABLE[nibbles & 7]
    signed = np.where(nibbles >= 8, -magnitude, magnitude)
    scale = scales.view(torch.float8_e4m3fn).double().numpy().repeat(16, axis=1)
    return signed * scale / divisor


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def matrices() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    gaussian = torch.randn(256, 512, generator=generator)
    heavy = torch.distributions.StudentT(2.5).sample((256, 512)) * 0.02
    outliers = gaussian.clone()
    outliers[::17, ::31] *= 40.0
    return {"gaussian": gaussian, "student_t": heavy, "outliers": outliers}


def test_low_nibble_holds_the_even_element() -> None:
    row = torch.zeros(1, 16)
    row[0, 0] = 6.0
    row[0, 1] = -0.5
    divisor = nvfp4_divisor(6.0)
    encoded = quantize_nvfp4_rows(row, divisor, device="cpu")
    assert encoded.codes[0, 0].item() == 0x7 | (0x9 << 4)
    values = decode(encoded.codes, encoded.scales, divisor, 16)
    assert values[0, 0] == 6.0 and values[0, 1] == -0.5 and not values[0, 2:].any()


def test_mse46_never_loses_to_absmax_and_codes_are_nearest() -> None:
    for name, values in matrices().items():
        host = values.double().numpy()
        divisor = nvfp4_divisor(float(values.abs().max()))
        absmax = quantize_nvfp4_rows(values, divisor, mse_46=False, device="cpu")
        best = quantize_nvfp4_rows(values, divisor, mse_46=True, device="cpu")
        plain = decode(absmax.codes, absmax.scales, divisor, values.shape[1])
        chosen = decode(best.codes, best.scales, divisor, values.shape[1])

        blocks = host.reshape(host.shape[0], -1, 16)
        plain_err = ((plain.reshape(blocks.shape) - blocks) ** 2).sum(-1)
        chosen_err = ((chosen.reshape(blocks.shape) - blocks) ** 2).sum(-1)
        assert (chosen_err <= plain_err + 1e-18).all(), name
        assert mse(chosen, host) < mse(plain, host), name
        relative = mse(chosen, host) / float(np.mean(host**2))
        assert relative < 0.02, (name, relative)

        # Every element decodes to the nearest representable value of its block scale.
        scale = best.scales.view(torch.float8_e4m3fn).double().numpy().repeat(16, axis=1)
        grid = np.concatenate((TABLE, -TABLE[1:]))
        candidates = grid[None, None, :] * scale[..., None] / divisor
        nearest = np.abs(candidates - host[..., None]).min(-1)
        assert (np.abs(chosen - host) <= nearest + 1e-12 * np.abs(host).max()).all(), name


def test_streamed_nvfp4_conversion_rereads_the_same_values(tmp_path) -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    source = matrices()["outliers"].to(torch.bfloat16)
    shard = "model.safetensors"
    save_file({name: source}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text("{}")
    model = Model({"text": {"config": {}}})
    shape = tuple(source.shape)
    with SafetensorsSource(tmp_path) as reader:
        model.add(
            Parameter("down", shape, tensor_source(reader, name, shape), inputs=("input",))
        )
        recipe = Recipe(model)
        recipe.assign("down", format="nvfp4", method="nvfp4_mse46")
        path = tmp_path / "down.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=128)
    with Artifact(path) as artifact:
        object_id = artifact.directory.bindings["down"]["object"]
        codes, scales, stored = decode_nvfp4_words(artifact.read_object(object_id), shape)
    divisor = float(stored)
    assert divisor == nvfp4_divisor(float(source.float().abs().max()))
    expected = quantize_nvfp4_rows(source.float(), divisor, device="cpu")
    assert torch.equal(codes, expected.codes)
    assert torch.equal(scales, expected.scales)
