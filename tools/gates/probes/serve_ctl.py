"""Restart ninfer-spo-serve through scripts/serve.cmd and wait until it answers.

  python serve_ctl.py restart --build ..\\build-spo-dev --log reqlog_x.jsonl [-- extra serve flags]
  python serve_ctl.py stop

The port is NINFER_PORT (default 8013). Only the ninfer-spo-serve.exe listening on that port is stopped;
if another program owns the port, stop refuses.
The request log and console go to NINFER_GATE_OUT (default <workspace>/tmp/ninfer).
"""
import argparse
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(os.environ.get("NINFER_GATE_OUT", REPO.parents[1] / "tmp" / "ninfer"))
PORT = int(os.environ.get("NINFER_PORT", "8013"))
IMAGE = "ninfer-spo-serve.exe"


def listener_pid():
    """PID listening on PORT, or None."""
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[3] == "LISTENING" and fields[1].rsplit(":", 1)[-1] == str(PORT):
            return int(fields[4])
    return None


def image_of(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return out.split(",")[0].strip('"') if out.startswith('"') else None


def running(pid):
    return pid is not None and image_of(pid) == IMAGE


def port_free():
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", PORT))
            return True
        except OSError:
            return False


def stop():
    pid = listener_pid()
    if pid is not None:
        image = image_of(pid)
        if image != IMAGE:
            sys.exit(f"port {PORT} belongs to {image} (pid {pid}), not {IMAGE}; not stopping it")
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    deadline = time.time() + 120
    while (running(pid) or not port_free()) and time.time() < deadline:
        time.sleep(1)
    if running(pid) or not port_free():
        sys.exit(f"server did not stop or port {PORT} stayed busy")


def restart(build, log, extra, env_extra=None):
    stop()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / log
    if log_path.exists():
        log_path.unlink()
    env = dict(os.environ, NINFER_BUILD=build, NINFER_PORT=str(PORT), **(env_extra or {}))
    console = open(OUT / f"serve_{pathlib.Path(log).stem}.txt", "w", encoding="utf-8")
    flags = ["--request-log-jsonl", str(log_path), *extra]
    script = subprocess.Popen(["cmd", "/c", str(REPO / "scripts" / "serve.cmd"), *flags], cwd=REPO, env=env,
                              stdout=console, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    started = time.time()
    while time.time() - started < 300:
        time.sleep(3)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            print(f"ready after {time.time() - started:.0f} s ({build}, port {PORT}, {' '.join(extra)})")
            return
        except OSError:
            if script.poll() is not None:
                sys.exit("server exited; see " + console.name)
    sys.exit("server not ready after 300 s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["restart", "stop"])
    parser.add_argument("--build", default="..\\build-spo-release")
    parser.add_argument("--log", default="reqlog.jsonl")
    args, extra = parser.parse_known_args()
    extra = [flag for flag in extra if flag != "--"]
    if args.mode == "stop":
        stop()
    else:
        restart(args.build, args.log, extra)


if __name__ == "__main__":
    main()
