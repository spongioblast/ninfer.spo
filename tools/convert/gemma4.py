"""Gemma 4 architecture adapter: text config, source axes, and packing groups.

Global layers store proportional RoPE as a standard partial neox rotation. The
converter permutes q, k, q_norm, k_norm, and o_proj so rotated dims are
contiguous. The embedding table is pre-multiplied by bf16(sqrt(hidden)).
"""

from __future__ import annotations

from math import isfinite, prod
from pathlib import Path
import struct
from typing import Mapping

import torch

from .model import Model, Parameter
from .resources import load_resources
from .sources.compressed_tensors import matrix_source
from .sources.logical import LogicalSource
from .sources.safetensors import SafetensorsSource, tensor_source


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def _f32(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name}: expected a real value")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name}: not representable as F32") from error
    if not isfinite(result):
        raise ValueError(f"{name}: expected finite F32")
    return result


def prope_perm(head_dim: int, partial_rotary_factor: float) -> torch.Tensor:
    """Map proportional RoPE pairs (j, j+head_dim/2) onto a contiguous prefix."""
    if head_dim % 2:
        raise ValueError("proportional RoPE requires an even head dim")
    pairs = int(partial_rotary_factor * head_dim // 2)
    if pairs <= 0 or pairs * 2 > head_dim:
        raise ValueError("proportional RoPE pair count is outside the head")
    half = head_dim // 2
    rotated = list(range(pairs)) + list(range(half, half + pairs))
    kept = [i for i in range(head_dim) if i not in set(rotated)]
    return torch.tensor(rotated + kept, dtype=torch.int64)


def theta_eff(rope_theta: float, partial_rotary_factor: float) -> float:
    return rope_theta**partial_rotary_factor


def embed_normalizer(hidden_size: int) -> float:
    """bf16(sqrt(hidden)), the value Hugging Face multiplies in bf16."""
    scale = torch.tensor(hidden_size, dtype=torch.float32).sqrt().to(torch.bfloat16)
    return float(scale.float())


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] != "Gemma4ForConditionalGeneration"
    ):
        raise ValueError(f"unsupported Gemma 4 architecture {architectures!r}")
    raw = source.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("Gemma 4 requires text_config")
    if raw.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 text MLP must be gelu_pytorch_tanh")
    if raw.get("enable_moe_block"):
        raise ValueError("Gemma 4 MoE text is not in this adapter")
    layers = raw.get("layer_types")
    count = _positive(raw.get("num_hidden_layers"), "text.num_hidden_layers")
    if (
        not isinstance(layers, list)
        or len(layers) != count
        or any(kind not in ("sliding_attention", "full_attention") for kind in layers)
    ):
        raise ValueError("text.layer_types must describe every block")
    hidden = _positive(raw.get("hidden_size"), "text.hidden_size")
    heads = _positive(raw.get("num_attention_heads"), "text.num_attention_heads")
    sliding_dim = _positive(raw.get("head_dim"), "text.head_dim")
    global_dim = _positive(
        raw.get("global_head_dim", sliding_dim), "text.global_head_dim"
    )
    sliding_kv = _positive(raw.get("num_key_value_heads"), "text.num_key_value_heads")
    global_kv = _positive(
        raw.get("num_global_key_value_heads", sliding_kv),
        "text.num_global_key_value_heads",
    )
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("text.rope_parameters must be an object")
    full = rope.get("full_attention", {})
    sliding = rope.get("sliding_attention", {})
    if full.get("rope_type") != "proportional":
        raise ValueError("full attention must use proportional RoPE")
    factor = _f32(full.get("partial_rotary_factor", 0.25), "partial_rotary_factor")
    if factor <= 0 or factor > 1:
        raise ValueError("partial_rotary_factor must be in (0, 1]")
    pairs = int(factor * global_dim // 2)
    result = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "vocab_size": _positive(raw.get("vocab_size"), "text.vocab_size"),
        "num_hidden_layers": count,
        "max_position_embeddings": _positive(
            raw.get("max_position_embeddings"), "text.max_position_embeddings"
        ),
        "intermediate_size": _positive(
            raw.get("intermediate_size"), "text.intermediate_size"
        ),
        "num_attention_heads": heads,
        "head_dim": sliding_dim,
        "global_head_dim": global_dim,
        "num_key_value_heads": sliding_kv,
        "num_global_key_value_heads": global_kv,
        "layer_types": list(layers),
        "sliding_window": _positive(raw.get("sliding_window"), "text.sliding_window"),
        "rms_norm_eps": _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps"),
        "final_logit_softcapping": _f32(
            raw.get("final_logit_softcapping", 30.0), "text.final_logit_softcapping"
        ),
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_scale": 1.0,
        "embed_normalizer": embed_normalizer(hidden),
        "rope_sliding_theta": _f32(
            sliding.get("rope_theta", 10000.0), "sliding.rope_theta"
        ),
        "rope_global_theta": _f32(full.get("rope_theta", 1e6), "full.rope_theta"),
        "partial_rotary_factor": factor,
        "prope_pairs": pairs,
        "prope_rotary_dim": pairs * 2,
        "prope_theta_eff": _f32(
            theta_eff(float(full.get("rope_theta", 1e6)), factor), "prope_theta_eff"
        ),
        "global_k_norm_scalar": {},
    }
    if result["prope_theta_eff"] <= 0:
        raise ValueError("proportional RoPE theta must be positive")
    return result


# The assistant reads the target's K/V, so its attention geometry and RoPE must match.
_ASSISTANT_SAME_KEYS = (
    "head_dim",
    "global_head_dim",
    "num_key_value_heads",
    "num_global_key_value_heads",
    "sliding_window",
    "vocab_size",
)


def assistant_config(source: dict, target_source: dict, text: dict) -> dict:
    """Config of the MTP assistant (`Gemma4AssistantForCausalLM`) for a target text config.

    Every assistant layer is query-only and attends to the K/V of the target's last layer of
    the same type. Draft steps reuse the target's RoPE at the last target position.
    """
    if source.get("architectures") != ["Gemma4AssistantForCausalLM"]:
        raise ValueError(
            f"unsupported Gemma 4 assistant architecture {source.get('architectures')!r}"
        )
    if source.get("use_ordered_embeddings"):
        raise ValueError("the assistant's centroid head is not in this adapter")
    if source.get("backbone_hidden_size") != text["hidden_size"]:
        raise ValueError("assistant backbone_hidden_size differs from the target hidden size")
    if source.get("tie_word_embeddings") is False:
        raise ValueError("the assistant head must be tied to its embedding")
    if target_source.get("text_config", {}).get("num_kv_shared_layers", 0):
        raise ValueError("a target with KV-shared layers is not in this adapter")
    raw = source.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("the assistant requires text_config")
    if raw.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("assistant MLP must be gelu_pytorch_tanh")
    if raw.get("final_logit_softcapping") is not None:
        raise ValueError("assistant logit softcap is not in this adapter")
    if raw.get("enable_moe_block") or raw.get("hidden_size_per_layer_input"):
        raise ValueError("assistant MoE or per-layer inputs are not in this adapter")
    count = _positive(raw.get("num_hidden_layers"), "assistant.num_hidden_layers")
    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != count
        or any(kind not in ("sliding_attention", "full_attention") for kind in layers)
    ):
        raise ValueError("assistant.layer_types must describe every block")
    if raw.get("num_kv_shared_layers") != count:
        raise ValueError("every assistant layer must read the target's K/V")
    for key in _ASSISTANT_SAME_KEYS:
        value = raw.get(key)
        if value != text[key]:
            raise ValueError(f"assistant {key} {value!r} differs from the target")
    heads = _positive(raw.get("num_attention_heads"), "assistant.num_attention_heads")
    if heads % text["num_key_value_heads"] or heads % text["num_global_key_value_heads"]:
        raise ValueError("assistant heads must group over the target's KV heads")
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("assistant.rope_parameters must be an object")
    full, sliding = rope.get("full_attention", {}), rope.get("sliding_attention", {})
    if (
        full.get("rope_type") != "proportional"
        or _f32(full.get("partial_rotary_factor", 0.25), "partial_rotary_factor")
        != text["partial_rotary_factor"]
        or _f32(full.get("rope_theta", 1e6), "full.rope_theta") != text["rope_global_theta"]
        or _f32(sliding.get("rope_theta", 10000.0), "sliding.rope_theta")
        != text["rope_sliding_theta"]
    ):
        raise ValueError("assistant RoPE differs from the target")
    eps = _f32(raw.get("rms_norm_eps", 1e-6), "assistant.rms_norm_eps")
    last = {kind: i for i, kind in enumerate(text["layer_types"])}
    if any(kind not in last for kind in layers):
        raise ValueError("the target has no layer of an assistant layer's type")
    return {
        "architectures": ["Gemma4Assistant"],
        "hidden_size": _positive(raw.get("hidden_size"), "assistant.hidden_size"),
        "backbone_hidden_size": text["hidden_size"],
        "intermediate_size": _positive(
            raw.get("intermediate_size"), "assistant.intermediate_size"
        ),
        "num_hidden_layers": count,
        "num_attention_heads": heads,
        "head_dim": text["head_dim"],
        "global_head_dim": text["global_head_dim"],
        "num_key_value_heads": text["num_key_value_heads"],
        "num_global_key_value_heads": text["num_global_key_value_heads"],
        "layer_types": list(layers),
        "kv_source_layers": [last[kind] for kind in layers],
        "sliding_window": text["sliding_window"],
        "rms_norm_eps": eps,
        "vocab_size": text["vocab_size"],
        "rope_sliding_theta": text["rope_sliding_theta"],
        "prope_rotary_dim": text["prope_rotary_dim"],
        "prope_theta_eff": text["prope_theta_eff"],
    }


def _scale_bf16(source: LogicalSource, scale: float) -> LogicalSource:
    factor = torch.tensor(scale, dtype=torch.float32).to(torch.bfloat16)

    def read(begin: int, end: int) -> torch.Tensor:
        values = source.read_values(begin, end).to(torch.bfloat16)
        return values * factor

    return LogicalSource(source.shape, f"bf16_scale({source.label},{scale})", read)


def _permute_rows(source: LogicalSource, heads: int, perm: torch.Tensor) -> LogicalSource:
    dim = int(perm.numel())

    def read(begin: int, end: int) -> torch.Tensor:
        rows, cols = source.shape
        if rows != heads * dim:
            raise ValueError(f"{source.label}: row permutation shape {source.shape}")
        row0, row1 = begin // cols, (end + cols - 1) // cols
        head0, head1 = row0 // dim, (row1 + dim - 1) // dim
        block = source.rows(head0 * dim, head1 * dim)
        block = block.reshape(-1, dim, cols)[:, perm, :].reshape(-1, cols)
        flat = block.reshape(-1)
        start = begin - head0 * dim * cols
        return flat[start : start + (end - begin)]

    return LogicalSource(source.shape, f"prope_rows({source.label})", read)


def _permute_cols(source: LogicalSource, heads: int, perm: torch.Tensor) -> LogicalSource:
    dim = int(perm.numel())

    def read(begin: int, end: int) -> torch.Tensor:
        rows, cols = source.shape
        if cols != heads * dim:
            raise ValueError(f"{source.label}: column permutation shape {source.shape}")
        row0, row1 = begin // cols, (end + cols - 1) // cols
        block = source.rows(row0, row1)
        block = block.reshape(-1, heads, dim)[:, :, perm].reshape(-1, cols)
        flat = block.reshape(-1)
        start = begin - row0 * cols
        return flat[start : start + (end - begin)]

    return LogicalSource(source.shape, f"prope_cols({source.label})", read)


def _permute_vector(source: LogicalSource, perm: torch.Tensor) -> LogicalSource:
    def read(begin: int, end: int) -> torch.Tensor:
        values = source.values().reshape(-1)
        if values.numel() != perm.numel():
            raise ValueError(f"{source.label}: vector permutation length")
        return values[perm][begin:end]

    return LogicalSource(source.shape, f"prope_vec({source.label})", read)


_SAME_SHAPE_KEYS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
)


class _Builder:
    def __init__(self, model: Model, base: SafetensorsSource):
        self.model = model
        self.base = base

    def _validate(self, selected: SafetensorsSource, name: str) -> None:
        if selected is self.base:
            return
        theirs = text_config(selected.config)
        for key in _SAME_SHAPE_KEYS:
            if theirs[key] != self.model.config[key]:
                raise ValueError(f"{name}: source {selected.path} differs in {key}")

    def add(self, name, source_name, shape, *, transform=None, inputs=(), direct="bf16"):
        """Register a parameter whose source can be reopened in another checkpoint.

        A transform needs value access, so transformed parameters import values only.
        """

        def factory(selected, format=None):
            self._validate(selected, name)
            if len(shape) == 2 and transform is None:
                return matrix_source(selected, source_name, shape, format)
            if format is not None:
                raise ValueError(f"{name}: encoded import needs an untransformed matrix")
            source = tensor_source(selected, source_name, shape)
            return source if transform is None else transform(source)

        self.model.add(
            Parameter(
                name,
                tuple(shape),
                factory(self.base),
                factory,
                tuple(inputs),
                direct,
                residency=name.split("/", 1)[0],
            )
        )

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))


def _annotate_k_norm(model: Model) -> None:
    config = model.config
    scalars = {}
    for index, kind in enumerate(config["layer_types"]):
        if kind != "full_attention":
            continue
        weight = model.parameters[
            f"text/layers/{index}/attention/key_norm"
        ].source.values()
        unique = torch.unique(weight.float())
        if unique.numel() == 1:
            scalars[str(index)] = float(unique[0])
    config["global_k_norm_scalar"] = scalars


def _add_assistant(builder: _Builder, config: dict, perm: torch.Tensor) -> None:
    """MTP assistant parameters. The query side of global layers takes the target's permutation."""
    hidden = config["hidden_size"]
    backbone = config["backbone_hidden_size"]
    heads = config["num_attention_heads"]
    intermediate = config["intermediate_size"]
    builder.add(
        "mtp/input_projection",
        "pre_projection.weight",
        (hidden, 2 * backbone),
        inputs=("mtp/stem_input",),
    )
    builder.add(
        "mtp/output_projection",
        "post_projection.weight",
        (backbone, hidden),
        inputs=("mtp/final_hidden",),
    )
    builder.add(
        "mtp/output_head",
        "model.embed_tokens.weight",
        (config["vocab_size"], hidden),
        inputs=("mtp/final_hidden",),
    )
    builder.add("mtp/final_norm", "model.norm.weight", (hidden,))
    for index, kind in enumerate(config["layer_types"]):
        global_layer = kind == "full_attention"
        dim = config["global_head_dim"] if global_layer else config["head_dim"]
        stem = f"mtp/layers/{index}/"
        src = f"model.layers.{index}."
        builder.add(
            stem + "attention/query",
            src + "self_attn.q_proj.weight",
            (heads * dim, hidden),
            transform=(
                (lambda source: _permute_rows(source, heads, perm)) if global_layer else None
            ),
            inputs=(stem + "mixer_input",),
        )
        builder.add(
            stem + "attention/query_norm",
            src + "self_attn.q_norm.weight",
            (dim,),
            transform=(lambda source: _permute_vector(source, perm)) if global_layer else None,
        )
        builder.add(
            stem + "attention/output",
            src + "self_attn.o_proj.weight",
            (hidden, heads * dim),
            transform=(
                (lambda source: _permute_cols(source, heads, perm)) if global_layer else None
            ),
            inputs=(stem + "attention_output",),
        )
        for role in ("gate", "up"):
            builder.add(
                stem + "mlp/" + role,
                src + f"mlp.{role}_proj.weight",
                (intermediate, hidden),
                inputs=(stem + "ffn_input",),
            )
        builder.add(
            stem + "mlp/down",
            src + "mlp.down_proj.weight",
            (hidden, intermediate),
            inputs=(stem + "mlp/product",),
        )
        for role, source_name in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_feedforward_norm", "pre_feedforward_layernorm"),
            ("post_feedforward_norm", "post_feedforward_layernorm"),
        ):
            builder.add(stem + role, src + source_name + ".weight", (hidden,))
        builder.add(stem + "layer_scalar", src + "layer_scalar", (1,), direct="fp32")
        builder.group(stem + "mlp/gate", stem + "mlp/up")


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text", "mtp"}:
        raise ValueError("Gemma 4 supports the text and mtp components")
    companions = {} if companions is None else dict(companions)
    if set(companions) - {"mtp"} or set(companions) != selected - {"text"}:
        raise ValueError("Gemma 4 mtp requires --source mtp=ASSISTANT and nothing else")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    if "mtp" in selected:
        records["mtp"] = {
            "config": assistant_config(companions["mtp"].config, base.config, config),
            "target": "text",
        }
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        overrides=resource_overrides,
    )
    records["text"]["resources"] = refs["text"]
    model = Model(
        records, resources=resources, token_count=count, special_token_ids=special
    )
    builder = _Builder(model, base)
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    intermediate = config["intermediate_size"]
    prefix = "model.language_model."
    normalizer = config["embed_normalizer"]
    builder.add(
        "text/token_embedding",
        prefix + "embed_tokens.weight",
        (vocab, hidden),
        transform=lambda source: _scale_bf16(source, normalizer),
    )
    builder.add(
        "text/output_head",
        prefix + "embed_tokens.weight",
        (vocab, hidden),
        inputs=("text/final_hidden",),
    )
    builder.add("text/final_norm", prefix + "norm.weight", (hidden,))
    perm = prope_perm(config["global_head_dim"], config["partial_rotary_factor"])
    for index, kind in enumerate(config["layer_types"]):
        global_layer = kind == "full_attention"
        dim = config["global_head_dim"] if global_layer else config["head_dim"]
        kv = (
            config["num_global_key_value_heads"]
            if global_layer
            else config["num_key_value_heads"]
        )
        stem = f"text/layers/{index}/"
        src = prefix + f"layers.{index}."
        q_rows, k_rows = heads * dim, kv * dim
        for role, rows, groups, source_name in (
            ("query", q_rows, heads, "q_proj"),
            ("key", k_rows, kv, "k_proj"),
        ):
            builder.add(
                stem + "attention/" + role,
                src + f"self_attn.{source_name}.weight",
                (rows, hidden),
                transform=(
                    (lambda source, groups=groups: _permute_rows(source, groups, perm))
                    if global_layer
                    else None
                ),
                inputs=(stem + "mixer_input",),
            )
        if not global_layer:
            builder.add(
                stem + "attention/value",
                src + "self_attn.v_proj.weight",
                (k_rows, hidden),
                inputs=(stem + "mixer_input",),
            )
        for role, source_name in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            builder.add(
                stem + "attention/" + role,
                src + f"self_attn.{source_name}.weight",
                (dim,),
                transform=(
                    (lambda source: _permute_vector(source, perm))
                    if global_layer
                    else None
                ),
            )
        builder.add(
            stem + "attention/output",
            src + "self_attn.o_proj.weight",
            (hidden, q_rows),
            transform=(
                (lambda source: _permute_cols(source, heads, perm))
                if global_layer
                else None
            ),
            inputs=(stem + "attention_output",),
        )
        for role in ("gate", "up"):
            builder.add(
                stem + "mlp/" + role,
                src + f"mlp.{role}_proj.weight",
                (intermediate, hidden),
                inputs=(stem + "ffn_input",),
            )
        builder.add(
            stem + "mlp/down",
            src + "mlp.down_proj.weight",
            (hidden, intermediate),
            inputs=(stem + "mlp/product",),
        )
        for role, source_name in (
            ("input_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_feedforward_norm", "pre_feedforward_layernorm"),
            ("post_feedforward_norm", "post_feedforward_layernorm"),
        ):
            builder.add(stem + role, src + source_name + ".weight", (hidden,))
        builder.add(
            stem + "layer_scalar", src + "layer_scalar", (1,), direct="fp32"
        )
        qkv = [stem + "attention/query", stem + "attention/key"]
        if not global_layer:
            qkv.append(stem + "attention/value")
        builder.group(*qkv)
        builder.group(stem + "mlp/gate", stem + "mlp/up")
    if "mtp" in selected:
        _add_assistant(_Builder(model, companions["mtp"]), records["mtp"]["config"], perm)
    _annotate_k_norm(model)
    if prod(model.parameters["text/token_embedding"].shape) <= 0:
        raise ValueError("empty Gemma 4 embedding")
    return model
