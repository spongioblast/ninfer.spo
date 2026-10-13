"""Gate runner. dev is the inner loop. quick is build plus non-real tests plus health.

GPU tests need an sm_120 GPU. Set CUDA_VISIBLE_DEVICES before a real run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEALTH = ROOT / "tools" / "repo_health" / "health.py"
BUILD = ROOT / "scripts" / "build.cmd"
# Gate outputs stay out of the repo, in the workspace tmp folder.
OUT = Path(os.environ.get("NINFER_GATE_OUT", ROOT.parents[1] / "tmp" / "ninfer"))


def run(command, cwd=ROOT):
    print("+", " ".join(command))
    return subprocess.run(command, cwd=cwd).returncode


def health(args):
    return run([sys.executable, str(HEALTH), *args])


def format_changed():
    clang = shutil.which("clang-format")
    if not clang:
        print("clang-format not on PATH; format check skipped")
        return 0
    names = []
    for command in (["git", "diff", "--name-only", "--diff-filter=d", "HEAD"],
                    ["git", "ls-files", "--others", "--exclude-standard"]):
        names += subprocess.run(command, cwd=ROOT, capture_output=True, text=True).stdout.split()
    sources = [name for name in names if name.endswith((".h", ".cpp", ".cu", ".cuh", ".inl"))]
    if not sources:
        return 0
    return run([clang, "--dry-run", "--Werror", *sources])


def build():
    if os.name == "nt":
        return run(["cmd", "/c", str(BUILD), "dev"])
    return run(["bash", str(ROOT / "scripts" / "build.sh"), "dev"])


def windows_only(level):
    if os.name == "nt":
        return False
    print(f"{level} runs .cmd probes and needs Windows; Linux has build, quick, and dev")
    return True


def wsl_path(path):
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def linux_gate(pattern):
    """CTest the existing Linux build in the CUDA 13.3 image. GPU 1 is the PRO 6000."""
    build_dir = ROOT.parent / "build-spo-linux"
    image = "nvidia/cuda:13.3.0-devel-ubuntu24.04"
    if shutil.which("wsl") is None or not build_dir.exists():
        print("Linux configure needs CMake and CUDA 13 in WSL, or build-spo-linux from the CUDA 13.3 image. See BLOCKERS spo-linux-toolchain.")
        return 1
    inspect = subprocess.run(["wsl", "-e", "docker", "image", "inspect", image], capture_output=True)
    if inspect.returncode != 0:
        print("Linux configure needs the CUDA 13.3 devel image. See BLOCKERS spo-linux-toolchain.")
        return 1
    selection = f"-R {pattern}" if pattern else "-E _real"
    script = (
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq && apt-get install -y -qq cmake python3 python3-jinja2 python3-jsonschema >/tmp/apt.log && "
        "cd /build && ctest --output-on-failure --timeout 900 -j 2 " + selection
    )
    return run([
        "wsl", "-e", "docker", "run", "--rm", "--gpus", "all",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-e", "CUDA_VISIBLE_DEVICES=1",
        "-v", f"{wsl_path(ROOT)}:/src:ro",
        "-v", f"{wsl_path(build_dir)}:/build",
        image, "bash", "-lc", script,
    ])


def ctest(regex_exclude=None, regex=None):
    build_dir = ROOT.parent / "build-spo-dev"
    if not build_dir.exists():
        print(f"missing build dir {build_dir}")
        return 1
    command = ["ctest", "--output-on-failure", "--timeout", "900", "-j", "4"]
    if regex_exclude:
        command += ["-E", regex_exclude]
    if regex:
        command += ["-R", regex]
    return run(command, cwd=build_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level", choices=["dev", "quick", "full", "perf", "soak", "linux"])
    parser.add_argument("pattern", nargs="?", default="")
    args = parser.parse_args()
    code = 0
    if args.level == "dev":
        code |= build()
        if args.pattern:
            code |= ctest(regex=args.pattern)
        code |= health(["--changed", "sizes"])
        code |= health(["--changed", "lang"])
        code |= health(["--changed", "layers"])
        code |= format_changed()
    elif args.level == "quick":
        code |= build()
        code |= ctest(regex_exclude="_real")
        code |= health(["all"])
        code |= format_changed()
    elif args.level == "linux":
        return linux_gate(args.pattern)
    elif args.level in ("full", "perf") and windows_only(args.level):
        return 1
    elif args.level == "full":
        code |= build()
        code |= ctest(regex="flash_next|qsa|rewrite_grouping|grammar|frontend|gemma4")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        code |= run(["cmd", "/c", str(ROOT / "tools" / "gates" / "probes" / "ppl_ab.cmd"), "dev", f"ppl_full_{stamp}"])
    elif args.level == "perf":
        # V2 serves Pi on 8003 until the switch; the spo server under test listens on 8013.
        os.environ.setdefault("NINFER_PORT", "8013")
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        probes = ROOT / "tools" / "gates" / "probes"
        code |= run([sys.executable, str(probes / "live_probe.py"), "--turns", "12", "--wait-ready"])
        code |= run([sys.executable, str(probes / "p8_probe.py"), "ttft", "--turns", "1",
                     "--out", str(OUT / f"p8_ttft_{stamp}.jsonl")])
        code |= run([sys.executable, str(probes / "p8_probe.py"), "switch", "--rounds", "1",
                     "--out", str(OUT / f"p8_switch_{stamp}.jsonl")])
        code |= run(["cmd", "/c", str(probes / "mtp_ab.cmd"), "perf"])
        code |= run(["cmd", "/c", str(probes / "prof_decode.cmd"), "perf"])
        code |= run([sys.executable, str(probes / "cancel_probe.py"),
                     "--request-log", str(OUT / "reqlog.jsonl"),
                     "--out", str(OUT / f"cancel_probe_{stamp}.jsonl")])
        code |= run([sys.executable, str(probes / "cross_path.py"),
                     "--build", str(ROOT.parent / "build-spo-dev"),
                     "--out", str(OUT / f"cross_path_{stamp}.jsonl")])
    elif args.level == "soak":
        os.environ.setdefault("NINFER_PORT", "8013")
        OUT.mkdir(parents=True, exist_ok=True)
        code |= run(
            [sys.executable, str(ROOT / "tools" / "gates" / "probes" / "soak.py"),
             "--minutes", "120", "--workers", "2",
             "--out", str(OUT / f"soak_gate_{time.strftime('%Y%m%d_%H%M%S')}")]
        )
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
