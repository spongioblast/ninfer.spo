"""Repository health checks for ninfer.spo. Python 3.11, standard library only.

sizes exits 1 on a hard-cap breach that is not allowlisted.
lang exits 1 on CJK text outside the allowlist.
layers exits 1 when a violation is not in tools/repo_health/layer_baseline.txt.
docs exits 1 when a repo path mentioned in the docs is missing.
dupes exits 1 on a file pair over 30% shared lines or a shared block over 40 lines, unless the
pair is allowlisted as `a::b reason`.
inventory and ownership exit 1 on an open or empty row.
"""
import argparse
import csv
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOFT = 800
HARD_CPP = 1250
HARD_CUDA = 1500
HARD_PY = 1000
CJK = re.compile(r"[\u4e00-\u9fff]")
INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"')
GETENV = re.compile(r"\bgetenv\s*\(")
PATH_MENTION = re.compile(r"`((?:src|include|docs|tools|apps|tests|scripts)/[^`\s]+)`")
WIN32 = re.compile(r"#\s*(?:if|ifdef|ifndef).*\b_WIN32\b|#\s*ifdef\s+_MSC_VER")


def allowlist():
    path = ROOT / "tools" / "repo_health" / "allowlist.txt"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition(" ")
        out[name] = reason or "allowlisted"
    return out


def git_changed():
    """Tracked files changed against HEAD plus untracked files that are not ignored."""
    names = []
    for command in (["git", "diff", "--name-only", "HEAD"],
                    ["git", "ls-files", "--others", "--exclude-standard"]):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        names += [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    return [ROOT / name for name in dict.fromkeys(names) if (ROOT / name).is_file()]


def source_files(changed_only):
    if changed_only:
        return git_changed()
    files = []
    for folder in ("src", "include", "tools", "tests", "docs", "apps", "scripts"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and "third_party" not in path.parts:
                files.append(path)
    return files


def line_count(path):
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def cap_for(path):
    rel = path.relative_to(ROOT).as_posix()
    suffix = path.suffix
    if suffix in {".cu", ".cuh"}:
        return HARD_CUDA, False
    if suffix == ".py":
        return HARD_PY, False
    if suffix in {".h", ".cpp", ".inl"} and rel.startswith(("src/", "include/")):
        return HARD_CPP, True
    return None, False


def cmd_sizes(files):
    allowed = allowlist()
    failures = 0
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        limit, soft = cap_for(path)
        if limit is None:
            continue
        count = line_count(path)
        if count > limit:
            print(f"HARD {count:5} {rel}")
            failures += 1
        elif soft and count > SOFT:
            print(f"soft {count:5} {rel}")
    return failures


def cmd_lang(files):
    allowed = allowlist()
    failures = 0
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        if path.suffix not in {".h", ".cpp", ".cu", ".cuh", ".py", ".md", ".cmake", ".txt", ".inl"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if CJK.search(line):
                print(f"CJK {rel}:{number}")
                failures += 1
                break
    return failures


def layer_of(rel):
    if rel.startswith("src/core/"):
        return "core"
    if rel.startswith("src/ops/"):
        return "ops"
    if rel.startswith("src/program/"):
        return "program"
    if rel.startswith("src/models/qwen3_5/"):
        return "qwen3_5"
    if rel.startswith("src/models/qwen3_8_flash_next"):
        return "flash_next"
    if rel.startswith("src/models/gemma4/"):
        return "gemma4"
    if rel.startswith("src/runtime/engine/"):
        return "engine"
    return ""


def include_layer(header):
    header = header.replace("\\", "/")
    if header.startswith("models/qwen3_5/"):
        return "qwen3_5"
    if header.startswith("models/qwen3_8_flash_next"):
        return "flash_next"
    if header.startswith("models/gemma4/"):
        return "gemma4"
    if header.startswith("models/"):
        return "models"
    if header.startswith(("runtime/", "serve/", "product/", "text/", "artifact/")):
        return "above-core"
    if header.startswith("ops/"):
        return "ops"
    if header.startswith("program/"):
        return "program"
    return ""


def cmd_layers(files):
    baseline_path = ROOT / "tools" / "repo_health" / "layer_baseline.txt"
    known = set()
    if baseline_path.exists():
        known = {line.strip() for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    failures = 0
    fresh = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in {".h", ".cpp", ".cu", ".cuh", ".inl"} or not rel.startswith("src/"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if GETENV.search(text) and rel != "src/core/runtime_options.cpp":
            finding = f"getenv {rel}"
            if finding not in known:
                fresh.append(finding)
        owner = layer_of(rel)
        if not rel.startswith("src/core/") and "/platform/" not in rel and WIN32.search(text):
            finding = f"platform {rel}"
            if finding not in known:
                fresh.append(finding)
        if not owner:
            continue
        for match in INCLUDE.finditer(text):
            target = include_layer(match.group(1))
            bad = False
            if owner == "core" and target in {"qwen3_5", "flash_next", "gemma4", "models", "above-core", "ops", "program"}:
                bad = True
            elif owner == "ops" and target in {"qwen3_5", "flash_next", "gemma4", "models", "above-core", "program"}:
                bad = True
            elif owner == "program" and target in {"qwen3_5", "flash_next", "gemma4", "models"}:
                bad = True
            elif owner == "qwen3_5" and target in {"flash_next", "gemma4"}:
                bad = True
            elif owner == "flash_next" and target in {"qwen3_5", "gemma4"}:
                bad = True
            elif owner == "gemma4" and target in {"qwen3_5", "flash_next"}:
                bad = True
            elif owner == "engine" and target in {"qwen3_5", "flash_next", "gemma4"} and "model_instance" not in rel and "registry" not in rel:
                bad = True
            if bad:
                finding = f"include {rel} -> {match.group(1)}"
                if finding not in known:
                    fresh.append(finding)
    for finding in fresh:
        print(finding)
        failures += 1
    return failures


def cmd_docs():
    failures = 0
    roots = [ROOT / "AGENTS.md", ROOT / "README.md"]
    docs = ROOT / "docs"
    if docs.exists():
        roots.extend(docs.rglob("*.md"))
    for path in roots:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in PATH_MENTION.finditer(text):
            mentioned = match.group(1).rstrip(".,)")
            if any(token in mentioned for token in "*<{}"):
                continue
            if not (ROOT / mentioned).exists():
                print(f"missing {mentioned} (from {path.relative_to(ROOT).as_posix()})")
                failures += 1
    return failures


def cmd_ledger(path, kind):
    """kind is inventory (fail on todo) or ownership (fail on a blank action)."""
    failures = 0
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        location = row.get("old_location") or row.get("path") or ""
        if not location:
            print(f"empty location: {row.get('item') or row}")
            failures += 1
        status = row.get("status") or row.get("action") or ""
        if kind == "inventory" and status == "todo":
            print(f"todo {row.get('item')}")
            failures += 1
        if kind == "ownership" and status not in {"keep", "rewrite", "delete"}:
            print(f"no action {location}")
            failures += 1
        if kind == "ownership" and status == "keep" and not (row.get("used_by") or "").strip():
            print(f"keep without used_by: {location}")
            failures += 1
        if status == "moved":
            loc = row.get("new_location", "")
            row_kind = row.get("kind", "")
            symbol = row.get("item", "")
            parts = [part.strip() for part in loc.split(";") if part.strip()]
            missing = [part for part in parts if not (ROOT / part).exists()]
            if not parts or missing:
                print(f"moved but missing: {';'.join(missing) or loc}")
                failures += 1
            elif row_kind.startswith("method") and symbol and not any(
                symbol in (ROOT / part).read_text(encoding="utf-8", errors="replace")
                for part in parts
            ):
                print(f"symbol {symbol} not in {loc}")
                failures += 1
        if status == "deleted" and not (row.get("note") or "").strip():
            print(f"deleted without a reason: {row.get('item')}")
            failures += 1
        if status == "delete" and location and (ROOT / location).exists():
            print(f"still present: {location}")
            failures += 1
    return failures


def cmd_dupes(left, right):
    def blocks(directory):
        found = []
        base = ROOT / directory
        for path in base.rglob("*"):
            if path.suffix not in {".h", ".cpp", ".cu", ".cuh"}:
                continue
            lines = []
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if len(line) <= 12 or line.startswith("//"):
                    continue
                line = line.replace("qwen3_5", "MODEL").replace("qwen3_8_flash_next", "MODEL").replace("gemma4", "MODEL")
                lines.append(line)
            found.append((path.relative_to(ROOT).as_posix(), lines))
        return found

    a_files = blocks(left)
    b_files = blocks(right)
    allowed = allowlist()
    failures = 0
    for a_name, a_lines in a_files:
        a_set = set(a_lines)
        if not a_set:
            continue
        for b_name, b_lines in b_files:
            b_set = set(b_lines)
            if not b_set or f"{a_name}::{b_name}" in allowed:
                continue
            overlap = len(a_set & b_set) / min(len(a_set), len(b_set))
            shared = 0
            run = 0
            b_join = "\n".join(b_lines)
            for line in a_lines:
                if line in b_set:
                    run += 1
                    shared = max(shared, run)
                else:
                    run = 0
            if overlap > 0.30 or shared > 40:
                print(f"dupe {overlap:.0%} block={shared} {a_name} :: {b_name}")
                failures += 1
    return failures


def cmd_report():
    target = ROOT / "docs" / "code-map.md"
    target.parent.mkdir(exist_ok=True)
    rows = []
    for path in source_files(False):
        limit, _soft = cap_for(path)
        if limit is None:
            continue
        rows.append((line_count(path), path.relative_to(ROOT).as_posix()))
    rows.sort(reverse=True)
    body = ["<!-- health:sizes:start -->", "", "| lines | file |", "|---:|---|"]
    body += [f"| {count} | `{rel}` |" for count, rel in rows[:40]]
    body += ["", "<!-- health:sizes:end -->", ""]
    text = "\n".join(body)
    if target.exists():
        current = target.read_text(encoding="utf-8")
        if "<!-- health:sizes:start -->" in current:
            current = re.sub(r"<!-- health:sizes:start -->.*<!-- health:sizes:end -->",
                             text.strip(), current, flags=re.S)
            target.write_text(current + ("" if current.endswith("\n") else "\n"), encoding="utf-8")
            return 0
        text = current.rstrip() + "\n\n" + text
    target.write_text(text, encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sizes", "lang", "layers", "docs", "dupes", "report", "all"])
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--inventory")
    parser.add_argument("--ownership")
    parser.add_argument("--changed", action="store_true")
    args = parser.parse_args()
    files = source_files(args.changed)
    code = 0
    if args.command in {"sizes", "all"}:
        code += cmd_sizes(files)
    if args.command in {"lang", "all"}:
        code += cmd_lang(files)
    if args.command in {"layers", "all"}:
        code += cmd_layers(files)
    if args.command in {"docs", "all"} and not args.changed:
        code += cmd_docs()
    if args.command == "dupes":
        if not args.a or not args.b:
            print("dupes needs --a and --b")
            return 2
        code += cmd_dupes(args.a, args.b)
    if args.command == "report":
        code += cmd_report()
    if args.inventory:
        code += cmd_ledger(args.inventory, "inventory")
    if args.ownership:
        code += cmd_ledger(args.ownership, "ownership")
    print(f"{args.command}: {code} problem(s)")
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
