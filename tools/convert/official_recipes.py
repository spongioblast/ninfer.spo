"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from math import isfinite, prod

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    nvfp4_mse46,
)

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def _require_gemma4(model, recipe_name):
    if model.config.get("model_type") != "gemma4_text":
        raise ValueError(f"{recipe_name} requires Gemma 4 text mathematics")


def gemma4_31b_bf16(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_bf16")
    for name, parameter in model.parameters.items():
        recipe.assign(name, format=parameter.direct_format, method=cast_direct)


def gemma4_sensitive(config, name):
    """True for projections of the first two and last two layers and for global attention."""
    layer = int(name.split("/")[2])
    last = config["num_hidden_layers"] - 1
    if layer in (0, 1, last - 1, last):
        return True
    return config["layer_types"][layer] == "full_attention" and "/attention/" in name


def gemma4_attention(config, name):
    """True for every attention projection."""
    return "/attention/" in name


def _gemma4_quantized(model, recipe, trunk, *, values=None, mlp=None, fp8=None):
    """FP8 embedding and heads; every text-layer projection gets `trunk(name)`.

    `values` replaces the base checkpoint for every text parameter. `mlp` imports the MLP's
    encoded NVFP4 words from that checkpoint instead of quantizing them. `fp8(config, name)`
    selects projections that stay FP8 instead of `trunk`. The MTP output head is FP8 like the
    target head. Other MTP parameters stay direct.
    """
    for name, parameter in model.parameters.items():
        text = name.startswith("text/")
        source = None if values is None or not text else model.source(name, values)
        if name in ("text/token_embedding", "text/output_head", "mtp/output_head"):
            recipe.assign(name, format=FP8, method=fp8_row_maxabs, source=source)
        elif name.startswith("text/layers/") and parameter.projection:
            if fp8 is not None and fp8(model.config, name):
                recipe.assign(name, format=FP8, method=fp8_row_maxabs, source=source)
            elif mlp is not None and "/mlp/" in name:
                recipe.assign(
                    name,
                    format="nvfp4",
                    method=import_encoded,
                    source=model.source(name, mlp, "nvfp4"),
                )
            else:
                format, method = trunk
                recipe.assign(name, format=format, method=method, source=source)
        else:
            recipe.assign(
                name, format=parameter.direct_format, method=cast_direct, source=source
            )


def gemma4_31b_fp8(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_fp8")
    _gemma4_quantized(model, recipe, (FP8, fp8_row_maxabs))


def gemma4_31b_nvfp4_base(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_nvfp4_base")
    _gemma4_quantized(model, recipe, ("nvfp4", nvfp4_mse46))


def gemma4_31b_nvfp4_qat(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_nvfp4_qat")
    _gemma4_quantized(model, recipe, ("nvfp4", nvfp4_mse46), values=sources["qat"])


def gemma4_31b_nvfp4_mixed(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_nvfp4_mixed")
    _gemma4_quantized(model, recipe, ("nvfp4", nvfp4_mse46), fp8=gemma4_sensitive)


def gemma4_31b_nvfp4_mlp(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_nvfp4_mlp")
    _gemma4_quantized(model, recipe, ("nvfp4", nvfp4_mse46), fp8=gemma4_attention)


def gemma4_31b_nvfp4_nvidia(model, recipe, sources):
    _require_gemma4(model, "gemma4_31b_nvfp4_nvidia")
    _gemma4_quantized(
        model, recipe, ("nvfp4", nvfp4_mse46), mlp=sources["quantized"]
    )


def _gemma4_a4(model, recipe, sources, recipe_name, mlp_only):
    _require_gemma4(model, recipe_name)
    calibration = sources["calibration"]
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if mlp_only and "/mlp/" not in name:
            continue
        if not all(s.format == "nvfp4" for s in recipe.selections[name]):
            continue
        for input_name in parameter.inputs:
            info = calibration.describe(input_name)
            if info.dtype != "F32" or prod(info.shape) != 1:
                raise ValueError(f"{input_name}: expected an F32 scalar amax")
            amax = float(calibration.read_flat(input_name)[0])
            if not isfinite(amax) or amax <= 0:
                raise ValueError(f"{input_name}: amax must be finite and positive")
            recipe.use(
                name,
                input_name,
                activation_policy="AllowA4",
                auxiliaries={"activation_input_divisor": 448.0 * 6.0 / amax},
            )


def gemma4_nvfp4_a4(model, recipe, sources):
    """Override: allow A4 on every NVFP4 text projection with a calibrated divisor.

    `--source calibration=FILE` is a safetensors file with one F32 activation amax per
    input name (for example ``text/layers/0/mixer_input``); the divisor is 448*6/amax.
    """
    _gemma4_a4(model, recipe, sources, "gemma4_nvfp4_a4", mlp_only=False)


def gemma4_nvfp4_a4_mlp(model, recipe, sources):
    """Override: like `gemma4_nvfp4_a4`, but only the MLP projections; attention stays A16."""
    _gemma4_a4(model, recipe, sources, "gemma4_nvfp4_a4_mlp", mlp_only=True)


def gemma4_fp8_a8(model, recipe, sources):
    """Override: allow A8 (per-token FP8 activations) on every FP8 text-layer projection.

    The embedding and the output head keep A16Only.
    """
    _require_gemma4(model, "gemma4_fp8_a8")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if not all(s.format == FP8 for s in recipe.selections[name]):
            continue
        for input_name in parameter.inputs:
            recipe.use(name, input_name, activation_policy="AllowA8")


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "gemma4_31b_bf16": gemma4_31b_bf16,
    "gemma4_31b_fp8": gemma4_31b_fp8,
    "gemma4_31b_nvfp4_base": gemma4_31b_nvfp4_base,
    "gemma4_31b_nvfp4_qat": gemma4_31b_nvfp4_qat,
    "gemma4_31b_nvfp4_nvidia": gemma4_31b_nvfp4_nvidia,
    "gemma4_31b_nvfp4_mixed": gemma4_31b_nvfp4_mixed,
    "gemma4_31b_nvfp4_mlp": gemma4_31b_nvfp4_mlp,
    "gemma4_nvfp4_a4": gemma4_nvfp4_a4,
    "gemma4_nvfp4_a4_mlp": gemma4_nvfp4_a4_mlp,
    "gemma4_fp8_a8": gemma4_fp8_a8,
}
