"""Decode and prefill rates from a serve request log.

  python soak_rates.py reqlog_soak_c1.jsonl
"""
import json
import statistics
import sys

decode = []
accept = []
prefill = []
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    if record.get("event") != "request_done":
        continue
    timings = record.get("timings_seconds") or {}
    result = record.get("result") or {}
    spec = record.get("speculative") or {}
    completion = result.get("completion_tokens") or 0
    decode_s = timings.get("decode") or 0
    if completion >= 16 and decode_s > 0:
        decode.append(completion / decode_s)
    drafted = spec.get("drafted_tokens") or 0
    if drafted:
        accept.append(spec.get("accepted_tokens", 0) / drafted)
    new_tokens = result.get("computed_prefill_tokens") or 0
    prefill_s = timings.get("prefill") or 0
    if new_tokens >= 128 and prefill_s > 0.01:
        prefill.append((new_tokens, new_tokens / prefill_s))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def line(label, values, unit):
    print(f"{label}: n={len(values)}  p10 {percentile(values, 0.1):.0f}  "
          f"median {statistics.median(values):.0f}  p90 {percentile(values, 0.9):.0f} {unit}")


print(f"requests with a real decode {len(decode)}")
line("decode", decode, "tok/s")
if accept:
    print(f"MTP accept: n={len(accept)}  median {statistics.median(accept):.2f}  "
          f"p10 {percentile(accept, 0.1):.2f}  p90 {percentile(accept, 0.9):.2f}")
print(f"prefills of at least 128 tokens {len(prefill)}")
line("prefill", [row[1] for row in prefill], "tok/s")
for lo, hi, label in ((128, 512, "128-512"), (512, 2000, "512-2k"), (2000, 8000, "2k-8k"), (8000, 10**9, "8k+")):
    part = [row[1] for row in prefill if lo <= row[0] < hi]
    if part:
        print(f"  {label:8s} n={len(part):5d}  median {statistics.median(part):7.0f} tok/s")
