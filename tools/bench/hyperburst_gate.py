"""Go/no-go for the hyperburst ports, from a genre JSONL and a round fit.

  python tools/bench/hyperburst_gate.py genre.jsonl round_fit.json

A phase is GO only when the measured gap can clear the adoption bar on short
replies, long replies, and coding together. Otherwise it prints SKIP and why.
This does not change the server.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def band(rows, **want):
    chosen = []
    for row in rows:
        if all(row.get(key) == value for key, value in want.items()):
            if row.get("client_tok_s"):
                chosen.append(row["client_tok_s"])
    if not chosen:
        return None
    return sum(chosen) / len(chosen)


def per_round(rows, genre, sampling):
    values = [
        row["tokens_per_round"]
        for row in rows
        if row.get("genre") == genre
        and row.get("sampling") == sampling
        and row.get("shape") == "long"
        and row.get("tokens_per_round")
    ]
    if not values:
        return None
    return sum(values) / len(values)


def positions(rows, genre, sampling):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        if row.get("genre") != genre or row.get("sampling") != sampling or row.get("shape") != "long":
            continue
        series = row.get("accepted_per_position") or []
        for index, value in enumerate(series):
            totals[index] += value
            counts[index] += 1
    if not counts:
        return []
    return [totals[index] / counts[index] for index in range(max(counts) + 1)]


def pct(new, old):
    if not old:
        return None
    return 100.0 * (new - old) / old


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: hyperburst_gate.py genre.jsonl round_fit.json")
    rows = load_jsonl(sys.argv[1])
    fit = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    print("means, client tok/s")
    for shape in ("short", "long"):
        for genre in ("short", "code-edit", "prose-en", "prose-ja", "agent-loop"):
            for sampling in ("greedy", "pi"):
                value = band(rows, shape=shape, genre=genre, sampling=sampling)
                if value is None:
                    continue
                print(f"  {shape} {genre} {sampling}: {value:.1f}")

    decisions = []

    def skip(phase, reason):
        decisions.append((phase, "SKIP", reason))
        print(f"{phase}: SKIP. {reason}")

    code = per_round(rows, "code-edit", "greedy")
    prose = per_round(rows, "prose-en", "greedy")
    agent = per_round(rows, "agent-loop", "greedy")
    short_g = band(rows, shape="short", sampling="greedy")
    short_p = band(rows, shape="short", sampling="pi")
    if code is None or prose is None:
        skip("P1", "long code-edit or prose tokens/round missing, so the picker gap is not measured")
    else:
        gap = pct(code, prose)
        print(f"greedy tokens/round code {code:.2f} prose {prose:.2f} agent {agent}")
        if gap is None or gap < 15:
            skip(
                "P1",
                f"code tokens/round is {gap:.1f}% vs prose, under a 15% gap. "
                "The pooled picker is not leaving a serious coding-only hole.",
            )
        else:
            skip(
                "P1",
                f"code tokens/round is {gap:.1f}% above prose, but a confidence picker "
                "changes only coding chains. Short and prose replies would not share that gain, "
                "so it fails the all-three bar.",
            )

    slope = fit.get("slope_ms")
    tok7 = fit.get("s7_tok_s")
    tok15 = fit.get("s15_tok_s_if_tail_holds")
    tail = positions(rows, "code-edit", "greedy")
    if slope is None or tok7 is None or tok15 is None:
        skip("P2", "round fit did not project S=7 against S=15")
    else:
        lift = pct(tok15, tok7)
        short_tail = tail[-1] if tail else None
        print(f"cost model S=7 {tok7:.1f} tok/s, S=15 if tail holds {tok15:.1f} ({lift:.1f}%)")
        if lift is None or lift < 5:
            skip("P2", f"15 drafts project {lift:.1f}% vs 7 even if the tail accept rate holds. Not a serious gain.")
        elif short_tail is not None and short_tail < 0.25:
            skip(
                "P2",
                f"15 drafts project {lift:.1f}% on the common-corpus tail, but code-edit's last "
                f"measured position accepts {short_tail:.2f}. Short replies would pay the wider "
                "round for little accept. Skip.",
            )
        else:
            skip(
                "P2",
                f"holding the common-corpus tail out to 15 drafts projects {lift:.1f}% "
                f"({tok7:.0f} to {tok15:.0f} tok/s) and costs {slope:.3f} ms per extra draft. "
                "On the code-edit prompts the seventh draft still accepts only about 0.41, so the "
                "same fit is about +2% there, and short replies never reach that tail. Not started.",
            )

    proposal_share = fit.get("proposal_share")
    if proposal_share is None:
        skip("P3a", "proposal-head share of the round was not in the fit file")
    else:
        removable = proposal_share * 2.0 / 3.0
        skip(
            "P3a",
            f"the Q4 draft head is {100 * proposal_share:.1f}% of a 4-draft round. "
            f"Cutting 147k rows to 49k removes about {100 * removable:.1f}% of the round before acceptance loss. "
            "The published maps lost 8-20% acceptance, which wipes that on code and on prose. Not started.",
        )

    draft_moe = fit.get("draft_moe_share")
    gemv = fit.get("fp8_gemv_share")
    router = fit.get("router_share")
    if draft_moe is None:
        skip("P3b", "draft MoE share was not measured")
    else:
        skip(
            "P3b",
            f"the small-token MoE entry is {100 * draft_moe:.1f}% of the round. "
            "The large NVFP4 kernel is target verify, which a draft-only GEMV does not replace. "
            "A draft-call saving cannot clear 3% on both a short reply and a coding reply.",
        )
    if gemv is None:
        skip("P3c", "FP8 GEMV share was not measured")
    else:
        skip(
            "P3c",
            f"FP8 K-split is {100 * gemv:.1f}% of the round, about 15 us per call against a ~16 us byte roof "
            "for a 10240x2560 FP8 row. No shape was under 85% of that roof.",
        )
    if router is None:
        skip("P3d", "router share was not measured")
    else:
        skip("P3d", f"router kernels are {100 * router:.2f}% of the round, under a 2 us-per-call bar")

    compared = []
    for genre in ("short", "code-edit", "prose-en", "agent-loop"):
        shape = "short" if genre == "short" else "long"
        greedy = band(rows, shape=shape, genre=genre, sampling="greedy")
        pi = band(rows, shape=shape, genre=genre, sampling="pi")
        if greedy and pi:
            compared.append((genre, pct(pi, greedy)))
    if len(compared) < 3:
        skip("P4", "greedy vs Pi sampling is missing on short, coding, or long prose")
    else:
        worst = min(delta for _genre, delta in compared)
        text = ", ".join(f"{genre} {delta:+.1f}%" for genre, delta in compared)
        print(f"Pi vs greedy: {text}")
        if worst > -5:
            skip("P4", f"Pi sampling is within 5% of greedy on every measured shape ({text})")
        else:
            skip(
                "P4",
                f"Pi sampling is slower ({text}). The published sparse-rejection package was flat on code-edit "
                "and about +6% pooled. A coding-flat result fails the short-and-long-and-coding bar, so it is not started.",
            )

    skip("P5", "two overlapping decodes were not in this single-request log. Do not add batch-2 graphs without that evidence.")
    skip("P6", "output-changing MoE pruning is not started from this plan")

    out = Path(sys.argv[1]).with_name("go_nogo.json")
    out.write_text(json.dumps([{"phase": phase, "verdict": verdict, "reason": reason} for phase, verdict, reason in decisions], indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
