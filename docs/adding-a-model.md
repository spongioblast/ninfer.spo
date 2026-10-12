# Adding a model

Gemma 4 is the worked example. It is not in this tree. The facts below are the model card and the tech report (arXiv 2607.02770): dense E2B, E4B, 12B, and 31B, and a 26B-A4B MoE (128 experts, top-8, one shared expert, 30 layers, hidden 2816). Attention only, no recurrent state. Local sliding-window layers alternate with global ones, 5:1 (4:1 on E2B). The window is 1024 tokens (512 on E2B and E4B), and the last layer is global. Global layers use one K=V tensor, a different head geometry, p-RoPE, and a final logit softcap of 30. Vocabulary 262k. Context 256k, or 128k on the small models.

The shared program layer can already describe an attention-only model: a state spec of zero bytes, more than one KV pool, and a checkpoint made of KV pages. `tests/program/test_attention_only_core.cpp` checks that. `src/program/storage/kv_pools.h` still has only full-page retention, so a sliding window cannot be expressed yet.

1. **Artifact.** Add a binder and a converter. Gemma's converter would live at `tools/convert/gemma4_*` and would write a V3 `.ninfer` whose architecture string is new. `src/models/registry.h` is the list of accepted architecture and model-type pairs.
2. **Frontend.** Template, tokenizer, and output session. Gemma's chat template and 262k vocabulary are its own. They do not go through the Qwen frontends.
3. **Execution.** The layer loop, projections, and norms. The MoE reuses `src/ops/sparse_moe` where the shapes match. p-RoPE and the logit softcap are new ops and need an oracle test each.
4. **Program facts.** KV pool specs, the state spec (`bytes == 0` for Gemma), checkpoint extras, and replay. Sliding-window retention has to land in `kv_pools.h` and the logical KV store before a local layer can drop tokens outside its window. A global layer and a local layer are two pools when their page geometry differs. K=V is one pool, not a pair.
5. **Package and runtime types.** A package per size, and a `RuntimeTypes` struct in the shape of `src/models/qwen3_5/program/runtime_types.h`, so `EngineCore` can drive the family.
6. **Registry.** Add the architecture to `src/models/registry.h` and a variant on `ConstructedModel` in `src/runtime/engine/model_instance.h`. The new family must not include `src/models/qwen3_5` or `src/models/qwen3_8_flash_next`.
7. **Tests.** An oracle for each new op. One real test that dumps tokens, skipped when the artifact is absent, the way `ninfer_qwen3_5_grammar_real_test` skips today.
8. **Gates.** `python tools/gates/run.py quick` for the unit tests. `full` once the family name is added to the `full` test pattern in `tools/gates/run.py`. A token change needs equal perplexity and a log entry.
