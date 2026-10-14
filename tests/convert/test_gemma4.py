from __future__ import annotations

import json
import struct

import pytest
from safetensors.torch import load_file, save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.gemma4 import build_model, embed_normalizer, prope_perm
from tools.convert.official_recipes import gemma4_31b_bf16
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import nvfp4_divisor, quantize_nvfp4_rows
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource


# NVFP4's block_scale_k16_m128x4 layout needs every packed parent at N % 128 and K % 64.
NVFP4_DIMS = {"hidden": 128, "inter": 64, "dim": 128}


def _config(hidden=16, inter=32, dim=8):
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": hidden,
            "vocab_size": 8,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
            "intermediate_size": inter,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": dim,
            "global_head_dim": dim,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    }


def _checkpoint(path, hidden=16, inter=32, dim=8):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config(hidden, inter, dim)))
    vocab = 8
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(vocab * hidden)
        .reshape(vocab, hidden)
        .to(torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    for index, global_layer in ((0, False), (1, True)):
        src = f"model.language_model.layers.{index}."
        kv = 1 if global_layer else 2
        q_rows, k_rows = 2 * dim, kv * dim
        tensors[src + "self_attn.q_proj.weight"] = torch.arange(q_rows * hidden).reshape(
            q_rows, hidden
        ).to(torch.bfloat16)
        tensors[src + "self_attn.k_proj.weight"] = (
            torch.arange(k_rows * hidden).reshape(k_rows, hidden) + 3
        ).to(torch.bfloat16)
        if not global_layer:
            tensors[src + "self_attn.v_proj.weight"] = torch.arange(
                k_rows * hidden
            ).reshape(k_rows, hidden).to(torch.bfloat16)
        tensors[src + "self_attn.o_proj.weight"] = torch.arange(hidden * q_rows).reshape(
            hidden, q_rows
        ).to(torch.bfloat16)
        tensors[src + "self_attn.q_norm.weight"] = torch.arange(dim).to(torch.bfloat16)
        key_norm = (
            torch.full((dim,), 0.5, dtype=torch.bfloat16)
            if global_layer
            else torch.arange(dim).to(torch.bfloat16)
        )
        tensors[src + "self_attn.k_norm.weight"] = key_norm
        for role in ("gate", "up"):
            tensors[src + f"mlp.{role}_proj.weight"] = torch.ones(
                inter, hidden, dtype=torch.bfloat16
            )
        tensors[src + "mlp.down_proj.weight"] = torch.ones(
            hidden, inter, dtype=torch.bfloat16
        )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[src + name + ".weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        tensors[src + "layer_scalar"] = torch.tensor([0.25], dtype=torch.float32)
    save_file(tensors, path / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def _values(model, name):
    parameter = model.parameters[name]
    return parameter.source.values().reshape(parameter.shape)


def test_mapping_scales_embedding_and_permutes_global_rope(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        scale = torch.tensor(embed_normalizer(16), dtype=torch.float32).to(
            torch.bfloat16
        )
        raw = source.read_flat(
            "model.language_model.embed_tokens.weight", 0, 8 * 16
        ).reshape(8, 16)
        assert torch.equal(_values(model, "text/token_embedding"), raw.to(torch.bfloat16) * scale)
        assert torch.equal(_values(model, "text/output_head"), raw)
        perm = prope_perm(8, 0.25)
        q = source.read_flat(
            "model.language_model.layers.1.self_attn.q_proj.weight", 0, 16 * 16
        ).reshape(2, 8, 16)
        got = _values(model, "text/layers/1/attention/query").reshape(2, 8, 16)
        assert torch.equal(got, q[:, perm, :])
        restored = got[:, torch.argsort(perm), :].reshape(16, 16)
        assert torch.equal(restored, q.reshape(16, 16))
        o = source.read_flat(
            "model.language_model.layers.1.self_attn.o_proj.weight", 0, 16 * 16
        ).reshape(16, 2, 8)
        got_o = _values(model, "text/layers/1/attention/output").reshape(16, 2, 8)
        assert torch.equal(got_o, o[:, :, perm])
        sliding = source.read_flat(
            "model.language_model.layers.0.self_attn.q_proj.weight", 0, 16 * 16
        ).reshape(16, 16)
        assert torch.equal(_values(model, "text/layers/0/attention/query"), sliding)
        assert "text/layers/1/attention/value" not in model.parameters
        assert model.config["global_k_norm_scalar"] == {"1": 0.5}
        assert model.config["prope_rotary_dim"] == 2
        assert model.parameters["text/layers/0/layer_scalar"].direct_format == "fp32"


def _modelopt_checkpoint(
    path, base_path, layers=(0,), shared_gate_up=False, hidden=16, inter=32, dim=8
):
    """Copy the base with the MLP of `layers` stored as NVIDIA ModelOpt NVFP4.

    `shared_gate_up` gives gate and up one weight divisor, as in NVIDIA's checkpoint.
    """
    path.mkdir()
    config = _config(hidden, inter, dim)
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (path / "config.json").write_text(json.dumps(config))
    tensors = load_file(base_path / "model.safetensors")
    generator = torch.Generator().manual_seed(7)
    expected = {}
    for layer in layers:
        values = {
            role: torch.randn(shape, generator=generator)
            for role, shape in (
                ("gate", (inter, hidden)),
                ("up", (inter, hidden)),
                ("down", (hidden, inter)),
            )
        }
        joint = max(float(values[role].abs().max()) for role in ("gate", "up"))
        for role, role_values in values.items():
            amax = joint if shared_gate_up and role != "down" else float(role_values.abs().max())
            divisor = nvfp4_divisor(amax)
            rows = quantize_nvfp4_rows(role_values, divisor, device="cpu")
            src = f"model.language_model.layers.{layer}.mlp.{role}_proj."
            del tensors[src + "weight"]
            tensors[src + "weight"] = rows.codes
            tensors[src + "weight_scale"] = rows.scales.view(torch.float8_e4m3fn)
            tensors[src + "weight_scale_2"] = torch.tensor([1.0 / divisor], dtype=torch.float32)
            tensors[src + "input_scale"] = torch.tensor([0.5], dtype=torch.float32)
            if layer == layers[0]:
                expected[role] = (role_values, rows)
    save_file(tensors, path / "model.safetensors")
    return SafetensorsSource(path), expected


def test_modelopt_mlp_imports_words_and_reciprocal_divisors(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        nvidia, expected = _modelopt_checkpoint(tmp_path / "nvidia", tmp_path / "source")
        with nvidia:
            for role, (values, rows) in expected.items():
                name = f"text/layers/0/mlp/{role}"
                imported = model.source(name, nvidia, "nvfp4")
                words = imported.read_encoded(0, values.shape[0])
                assert torch.equal(words.codes, rows.codes)
                assert torch.equal(words.scales, rows.scales)
                multiplier = struct.unpack("<f", struct.pack("<f", 1.0 / nvfp4_divisor(
                    float(values.abs().max())
                )))[0]
                assert words.weight_divisor == struct.pack("<f", 1.0 / multiplier)
                assert imported.input_divisor() == struct.pack("<f", 2.0)
                decoded = imported.values().reshape(values.shape)
                error = (decoded - values).pow(2).sum() / values.pow(2).sum()
                assert float(error) < 0.02
                signs = (torch.sign(decoded) * torch.sign(values)).sum(dim=1)
                assert bool((signs > 0).all())
            auto = model.source("text/layers/0/mlp/gate", nvidia)
            assert auto.read_encoded(0, 32).format == "nvfp4"
            plain = model.source("text/layers/0/attention/query", nvidia)
            assert torch.equal(plain.values(), _values(model, "text/layers/0/attention/query").reshape(-1))
            with pytest.raises(ValueError, match="untransformed"):
                model.source("text/layers/1/attention/query", nvidia, "nvfp4")


def test_bf16_recipe_rereads_scaled_embedding(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        recipe = Recipe(model)
        gemma4_31b_bf16(model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=4)
    with Artifact(out) as artifact:
        assert artifact.directory.components["text"]["config"]["embed_normalizer"] == (
            embed_normalizer(16)
        )
        binding = artifact.directory.bindings["text/token_embedding"]
        obj = artifact.by_id[binding["object"]]
        raw = artifact.read_object(obj.id)
        stored = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
            obj.shape
        )
        scale = torch.tensor(embed_normalizer(16), dtype=torch.float32).to(
            torch.bfloat16
        )
        expect = (
            torch.arange(8 * 16).reshape(8, 16).to(torch.bfloat16) * scale
        )
        assert torch.equal(stored, expect)


def _objects(artifact, name):
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        return [(binding["object"], None)]
    return [(part["object"], tuple(part["range"])) for part in binding["parts"]]


def _packed(artifact, names):
    """Assert `names` fill one stored object in order and return that object."""
    parts = [_objects(artifact, name) for name in names]
    assert all(len(part) == 1 for part in parts)
    objects = {part[0][0] for part in parts}
    assert len(objects) == 1
    cursor = 0
    for part in parts:
        begin, end = part[0][1]
        assert begin == cursor
        cursor = end
    obj = artifact.by_id[objects.pop()]
    assert cursor == obj.shape[0] * obj.shape[1]
    return obj


@pytest.mark.parametrize(
    "recipe_name,trunk",
    [
        ("gemma4_31b_fp8", "fp8_e4m3fn_row_bf16"),
        ("gemma4_31b_nvfp4_base", "nvfp4"),
        # Both layers of the two-layer model are edge layers.
        ("gemma4_31b_nvfp4_mixed", "fp8_e4m3fn_row_bf16"),
    ],
)
def test_quantized_recipes_pack_qkv_and_gate_up(tmp_path, recipe_name, trunk):
    from tools.convert.official_recipes import RECIPES

    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        recipe = Recipe(model)
        RECIPES[recipe_name](model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        sliding = "text/layers/0/"
        qkv = _packed(artifact, [sliding + "attention/" + r for r in ("query", "key", "value")])
        assert qkv.format == trunk and tuple(qkv.shape) == (768, 128)
        global_qk = _packed(artifact, ["text/layers/1/attention/query", "text/layers/1/attention/key"])
        assert global_qk.format == trunk and tuple(global_qk.shape) == (384, 128)
        for layer in (0, 1):
            gate_up = _packed(artifact, [f"text/layers/{layer}/mlp/gate", f"text/layers/{layer}/mlp/up"])
            assert gate_up.format == trunk and tuple(gate_up.shape) == (128, 128)
            down = artifact.by_id[_objects(artifact, f"text/layers/{layer}/mlp/down")[0][0]]
            assert down.format == trunk
        for edge in ("text/token_embedding", "text/output_head"):
            assert artifact.by_id[_objects(artifact, edge)[0][0]].format == "fp8_e4m3fn_row_bf16"
        norm = artifact.by_id[_objects(artifact, "text/final_norm")[0][0]]
        assert norm.format == "bf16"


def test_mlp_recipe_keeps_attention_fp8(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_nvfp4_mlp

    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        recipe = Recipe(model)
        gemma4_31b_nvfp4_mlp(model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        for layer in (0, 1):
            prefix = f"text/layers/{layer}/"
            for name in ("attention/query", "attention/output"):
                stored = artifact.by_id[_objects(artifact, prefix + name)[0][0]]
                assert stored.format == "fp8_e4m3fn_row_bf16"
            for name in ("mlp/gate", "mlp/down"):
                assert artifact.by_id[_objects(artifact, prefix + name)[0][0]].format == "nvfp4"


def test_sensitive_projections_are_edges_and_global_attention():
    from tools.convert.official_recipes import gemma4_sensitive

    types = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
    config = {"num_hidden_layers": 60, "layer_types": types}
    picked = {
        (layer, part)
        for layer in range(60)
        for part in ("attention/query", "attention/output", "mlp/gate", "mlp/down")
        if gemma4_sensitive(config, f"text/layers/{layer}/{part}")
    }
    edges = {(layer, part) for layer in (0, 1, 58, 59)
             for part in ("attention/query", "attention/output", "mlp/gate", "mlp/down")}
    global_attention = {(layer, part) for layer in range(5, 60, 6)
                        for part in ("attention/query", "attention/output")}
    assert picked == edges | global_attention


def test_nvidia_recipe_imports_shared_gate_up(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_nvfp4_nvidia

    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        nvidia, expected = _modelopt_checkpoint(
            tmp_path / "nvidia",
            tmp_path / "source",
            layers=(0, 1),
            shared_gate_up=True,
            **NVFP4_DIMS,
        )
        with nvidia:
            recipe = Recipe(model)
            gemma4_31b_nvfp4_nvidia(model, recipe, {"base": source, "quantized": nvidia})
            out = tmp_path / "gemma.ninfer"
            convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        gate_up = _packed(artifact, ["text/layers/0/mlp/gate", "text/layers/0/mlp/up"])
        assert gate_up.format == "nvfp4"
        qkv = _packed(artifact, ["text/layers/0/attention/" + r for r in ("query", "key", "value")])
        assert qkv.format == "nvfp4"
        manifest = json.loads((tmp_path / "gemma.ninfer.conversion.json").read_text())
        methods = {job["object"]: job["method"] for job in manifest["methods"]}
        assert methods[gate_up.id] == "import_encoded"
        assert methods[qkv.id] == "nvfp4_mse46"


def _calibration(path, amax):
    save_file({name: torch.tensor([value], dtype=torch.float32) for name, value in amax.items()}, path)
    return SafetensorsSource(path)


def test_a4_override_stores_calibrated_divisors(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_nvfp4_base, gemma4_nvfp4_a4

    roles = ("mixer_input", "attention_output", "ffn_input", "mlp/product")
    amax = {f"text/layers/{layer}/{role}": 2.0 + layer + i for layer in (0, 1) for i, role in enumerate(roles)}
    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        with _calibration(tmp_path / "amax.safetensors", amax) as calibration:
            recipe = Recipe(model)
            sources = {"base": source, "calibration": calibration}
            gemma4_31b_nvfp4_base(model, recipe, sources)
            gemma4_nvfp4_a4(model, recipe, sources)
            out = tmp_path / "gemma.ninfer"
            convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        uses = {(use["parameter"], use["input"]): use for use in artifact.directory.uses}
        for (name, input_name), use in uses.items():
            if not name.startswith("text/layers/"):
                assert use["activation_policy"] == "A16Only"
                assert not use.get("auxiliaries")
                continue
            assert use["activation_policy"] == "AllowA4"
            [binding] = use["auxiliaries"].values()
            stored = artifact.read_object(binding["object"])
            assert stored == struct.pack("<f", 448.0 * 6.0 / amax[input_name])
        assert ("text/layers/1/attention/key", "text/layers/1/mixer_input") in uses


def test_a4_mlp_override_leaves_attention_a16(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_nvfp4_base, gemma4_nvfp4_a4_mlp

    amax = {f"text/layers/{layer}/{role}": 4.0 for layer in (0, 1) for role in ("ffn_input", "mlp/product")}
    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        with _calibration(tmp_path / "amax.safetensors", amax) as calibration:
            recipe = Recipe(model)
            sources = {"base": source, "calibration": calibration}
            gemma4_31b_nvfp4_base(model, recipe, sources)
            gemma4_nvfp4_a4_mlp(model, recipe, sources)
            out = tmp_path / "gemma.ninfer"
            convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        for use in artifact.directory.uses:
            a4 = "/mlp/" in use["parameter"]
            assert use["activation_policy"] == ("AllowA4" if a4 else "A16Only")
            assert bool(use.get("auxiliaries")) == a4


def test_a8_override_allows_a8_on_fp8_layer_projections(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_fp8, gemma4_fp8_a8

    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        recipe = Recipe(model)
        sources = {"base": source}
        gemma4_31b_fp8(model, recipe, sources)
        gemma4_fp8_a8(model, recipe, sources)
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        uses = artifact.directory.uses
        for use in uses:
            a8 = use["parameter"].startswith("text/layers/")
            assert use["activation_policy"] == ("AllowA8" if a8 else "A16Only")
            assert not use.get("auxiliaries")
        assert any(use["parameter"] == "text/output_head" for use in uses)


def test_a4_override_requires_every_input(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_nvfp4_base, gemma4_nvfp4_a4

    with _checkpoint(tmp_path / "source", **NVFP4_DIMS) as source:
        model = build_model(source)
        partial = {"text/layers/0/mixer_input": 1.0}
        with _calibration(tmp_path / "amax.safetensors", partial) as calibration:
            recipe = Recipe(model)
            sources = {"base": source, "calibration": calibration}
            gemma4_31b_nvfp4_base(model, recipe, sources)
            with pytest.raises(ValueError, match="attention_output"):
                gemma4_nvfp4_a4(model, recipe, sources)
