from __future__ import annotations

import json

from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.gemma4 import build_model, embed_normalizer, prope_perm
from tools.convert.official_recipes import gemma4_31b_bf16
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource


def _config():
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 16,
            "vocab_size": 8,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
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


def _checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config()))
    hidden, vocab, inter = 16, 8, 24
    q, k = 16, 16
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(vocab * hidden)
        .reshape(vocab, hidden)
        .to(torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    for index, global_layer in ((0, False), (1, True)):
        src = f"model.language_model.layers.{index}."
        dim = 8
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
