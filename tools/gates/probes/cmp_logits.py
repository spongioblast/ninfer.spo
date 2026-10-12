"""Compare two logits dump directories (NINFER_DEBUG_LOGITS_DIR) and two transcripts.

Prefill dumps are matched by order of the "prefill" route; each pair must have the same input token
and position and is compared bitwise. Decode dumps are matched by (position of last input, route).

  python cmp_logits.py logits1_A logits1_B [merge_A.json merge_B.json]
"""
import json
import pathlib
import sys

import numpy as np


def load(directory):
    out = []
    for meta_path in sorted(pathlib.Path(directory).glob("*.json"), key=lambda p: int(p.stem)):
        meta = json.loads(meta_path.read_text())
        values = np.fromfile(meta_path.with_suffix(".bf16"), dtype=np.uint16)
        out.append((meta, values))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
if len(sys.argv) > 4:
    ta = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    tb = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
    same = [x == y for x, y in zip(ta, tb)]
    print("transcript messages equal:", same)

for route in ("prefill",):
    pa = [x for x in a if x[0]["route"] == route]
    pb = [x for x in b if x[0]["route"] == route]
    print(f"{route}: {len(pa)} vs {len(pb)} dumps")
    for index, ((ma, va), (mb, vb)) in enumerate(zip(pa, pb)):
        key_a = (ma["input_tokens"], ma["positions"])
        key_b = (mb["input_tokens"], mb["positions"])
        if key_a != key_b:
            print(f"  #{index}: different inputs {key_a} vs {key_b}")
            continue
        diff = int(np.count_nonzero(va != vb))
        fa = va.astype(np.uint32) << 16
        fb = vb.astype(np.uint32) << 16
        la, lb = fa.view(np.float32), fb.view(np.float32)
        print(f"  #{index}: pos {ma['positions']} differing logits {diff}/{va.size}"
              f"  max abs {float(np.max(np.abs(la - lb))):.4g}  argmax {int(la.argmax())} vs {int(lb.argmax())}")


def by_position(dumps):
    table = {}
    for meta, values in dumps:
        if meta["route"] == "prefill":
            continue
        width = meta["width"]
        vocab = meta["vocab"]
        rows = values.reshape(-1, vocab)
        positions = meta["positions"]
        tokens = meta["input_tokens"]
        for row in range(min(len(positions), rows.shape[0])):
            table.setdefault((positions[row], tokens[row]), rows[row])
    return table


da, db = by_position(a), by_position(b)
common = sorted(set(da) & set(db))
equal = sum(1 for key in common if np.array_equal(da[key], db[key]))
print(f"decode rows keyed by (position, token): {len(da)} vs {len(db)}, common {len(common)}, bit-equal {equal}")
for key in common:
    if not np.array_equal(da[key], db[key]):
        print("  first differing decode row", key)
        break
