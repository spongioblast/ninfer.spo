# Perplexity evaluation

`ninfer-spo-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact. `scripts/build.cmd dev` writes it to `build-spo-dev/apps`.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API;
the optional top-k dump below exists only to compare artifacts.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`. The default KV format is FP8; Gemma 4
artifacts store bf16 KV only and need `--kv-dtype bf16`.

`--prefix-ids 2,105,...` prepends the same token ids to every window. They are context only: the
scored targets and token counts do not change. Use it for models that are far from their training
distribution on bare text. Gemma 4 instruction-tuned models score raw prose near perplexity 1,000
but behave normally after a chat preamble that opens a model turn (the ids of the rendered
template up to the generation prompt). The report records the ids under `execution.prefix_ids`.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.

## Top-k dump for KL comparison

`--dump-topk FILE` also writes, for every scored position, the 64 most likely next-token ids and
their log-softmax log-probabilities. Two dumps from the same corpus, context and stride line up
position by position, so a quantized artifact can be compared with its BF16 reference by
KL divergence and top-1 agreement without storing full logits. Only Gemma 4 artifacts support it
today; other families reject the option.

The file is little-endian: the 8 bytes `NTOPK001`, a `u32` k (64), a `u32` 0, then per scored
position k `i32` ids (most likely first) followed by k `f32` log-probabilities. Positions follow
the stream, window and target order of the scoring itself.
