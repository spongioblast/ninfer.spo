"""Compare two ninfer-perplexity report.json files window by window.

  python ppl_cmp.py dirA dirB
"""
import json
import sys

a = json.load(open(sys.argv[1] + "/report.json"))
b = json.load(open(sys.argv[2] + "/report.json"))
print(f"total nll A {a['overall']['total_nll']:.6f}  B {b['overall']['total_nll']:.6f}  "
      f"ppl A {a['overall']['perplexity']:.8f}  B {b['overall']['perplexity']:.8f}")
wa, wb = a["streams"][0]["windows"], b["streams"][0]["windows"]
keys = [k for k in wa[0] if isinstance(wa[0][k], float)]
diff = [i for i, (x, y) in enumerate(zip(wa, wb)) if any(x[k] != y[k] for k in keys)]
print(f"windows {len(wa)}, differing {len(diff)}, float keys {keys}")
deltas = [wa[i][keys[0]] - wb[i][keys[0]] for i in diff] if keys else []
if deltas:
    print(f"per-window {keys[0]} delta: min {min(deltas):.6f} max {max(deltas):.6f} "
          f"sum {sum(deltas):.6f}, A better in {sum(d < 0 for d in deltas)} of {len(deltas)}")
