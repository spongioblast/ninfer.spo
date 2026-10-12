"""Pi-like soak against ninfer-spo-serve on NINFER_PORT (default 8013) with a resource monitor.

  python soak.py --selftest
  python soak.py --minutes 60 --workers 2 --out soak_c2
  python soak.py --minutes 120 --workers 2 --out soak_r10 --detach

Each worker runs agent sessions like p8_probe.py: a session grows by tool results (common-sample
texts and NInfer sources) from ~8k to 40k-90k tokens, then a new session starts. Every turn streams
the reply; every 5th turn has thinking on and a longer limit; every 15th turn adds an image.
Writes <out>.requests.jsonl (one line per request, with client inter-chunk gaps) and
<out>.monitor.jsonl (one line per minute: process memory, NVML, WDDM, and the per-minute floor and
high-water mark of device_kv_used_pages and host_context_used_bytes from /metrics).
The end of the run prints an ordinary-least-squares slope and a 95% interval for each floor,
widened for the autocorrelation of the per-minute floors.
A long run should be started with --detach so it survives the console closing. That relaunches
this process with no window and appends its own log to <out>.run.log.
"""
import argparse
import base64
import json
import pathlib
import random
import statistics
import subprocess
import threading
import time
import urllib.request

import psutil
import pynvml

import math
import os
import sys

import p8_probe as probe

# Two-sided 95% Student-t critical values, df -> t. Past the table, 1.96.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000,
    80: 1.990, 120: 1.980, 240: 1.970,
}


def t95(df):
    if df in _T95:
        return _T95[df]
    lower = max((key for key in _T95 if key < df), default=None)
    if lower is None:
        return _T95[1]
    if df > 240:
        return 1.96
    upper = min(key for key in _T95 if key > df)
    span = (df - lower) / (upper - lower)
    return _T95[lower] + span * (_T95[upper] - _T95[lower])


def ols_interval(xs, ys):
    """Return (slope, low, high) for the 95% interval of the least-squares slope.

    Per-minute floors are autocorrelated, so the plain OLS interval is too narrow. The standard
    error is widened by the lag-1 autocorrelation r of the residuals, sqrt((1 + r) / (1 - r)),
    and the degrees of freedom use the effective sample size n (1 - r) / (1 + r)."""
    count = len(xs)
    if count < 3 or count != len(ys):
        raise ValueError("a slope needs at least 3 paired samples")
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    squares = sum(r * r for r in residuals)
    degrees = count - 2
    variance = squares / degrees
    lag = sum(a * b for a, b in zip(residuals, residuals[1:])) / squares if squares > 0 else 0.0
    lag = min(max(lag, 0.0), 0.95)
    inflation = (1 + lag) / (1 - lag)
    effective = max(int(count / inflation) - 2, 1)
    error = math.sqrt(variance * inflation / sxx)
    half = t95(effective) * error
    return slope, slope - half, slope + half


def slope_verdict(low, high, high_water, capacity):
    """leak when the whole interval is above zero away from a full pool, saturation when the
    floor rises on a pool whose high-water mark is at least 90% of capacity, falling when the
    whole interval is below zero (memory being returned), otherwise flat."""
    near_full = capacity > 0 and high_water >= 0.9 * capacity
    if low > 0 and near_full:
        return "saturation"
    if low > 0:
        return "leak"
    if high < 0:
        return "falling"
    return "flat"


def slope_selftest():
    xs = list(range(40))
    flat = [100 + (i % 3) - 1 for i in xs]
    _, low, high = ols_interval(xs, flat)
    if slope_verdict(low, high, 100, 1000) != "flat":
        raise SystemExit(f"flat series looked like a leak ({low}, {high})")
    rising = [100 + 5 * i for i in xs]
    _, low, high = ols_interval(xs, rising)
    if slope_verdict(low, high, 300, 10000) != "leak":
        raise SystemExit(f"rising series was not a leak ({low}, {high})")
    if slope_verdict(low, high, 9500, 10000) != "saturation":
        raise SystemExit(f"rising series near capacity was not saturation ({low}, {high})")
    falling = [1000 - 5 * i for i in xs]
    _, low, high = ols_interval(xs, falling)
    if slope_verdict(low, high, 1000, 10000) != "falling":
        raise SystemExit(f"falling series was not reported as falling ({low}, {high})")
    # Autocorrelated noise with no trend (AR(1), phi 0.9). The plain OLS interval for this
    # seed is [0.198, 0.631] and would call it a leak.
    rng = random.Random(1)
    noise, drift = [], 0.0
    for _ in xs:
        drift = 0.9 * drift + rng.gauss(0, 5)
        noise.append(1000 + drift)
    _, low, high = ols_interval(xs, noise)
    if slope_verdict(low, high, 1050, 10000) != "flat":
        raise SystemExit(f"autocorrelated noise without a trend was not flat ({low}, {high})")
    print("soak slope selftest ok")


METRICS = (
    "ninfer_device_kv_used_pages",
    "ninfer_device_kv_capacity_pages",
    "ninfer_host_context_used_bytes",
    "ninfer_host_context_capacity_bytes",
)


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in METRICS:
            values[parts[0]] = float(parts[1])
    return values


def scrape_metrics():
    port = os.environ.get("NINFER_PORT", "8013")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            return parse_metrics(resp.read().decode("utf-8", "replace"))
    except OSError as error:
        return {"error": repr(error)}

MEDIA = pathlib.Path(os.environ.get(
    "NINFER_SOAK_MEDIA",
    pathlib.Path(__file__).resolve().parents[5] / "docs" / "verification" / "main-2026-10-03"))


def stream(messages, max_tokens, thinking):
    body = json.dumps(
        {
            "model": probe.MODEL,
            "messages": messages,
            "tools": probe.TOOLS,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "enable_thinking": thinking,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(probe.URL, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    last = None
    gaps = []
    final = None
    text = []
    reasoning = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("timings") or chunk.get("usage"):
                final = chunk
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if piece or delta.get("tool_calls") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    else:
                        gaps.append(now - last)
                    last = now
                    text.append(piece)
    total = time.perf_counter() - started
    timings = (final or {}).get("timings") or {}
    usage = (final or {}).get("usage") or {}
    gaps.sort()
    return {
        "ttft_ms": round(((first or time.perf_counter()) - started) * 1000.0, 1),
        "total_s": round(total, 3),
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "completion_tokens": usage.get("completion_tokens"),
        "engine_decode_tps": round(timings.get("predicted_per_second", 0.0), 1),
        "client_decode_tps": round(len(gaps) / sum(gaps), 1) if gaps and sum(gaps) > 0 else None,
        "gap_p50_ms": round(gaps[len(gaps) // 2] * 1000.0, 2) if gaps else None,
        "gap_p99_ms": round(gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] * 1000.0, 2) if gaps else None,
        "chunks": len(gaps) + (1 if first else 0),
        "text": "".join(text),
        "reasoning": "".join(reasoning),
    }


def image_message(rng):
    path = rng.choice(sorted(MEDIA.glob("*.png")))
    data = base64.b64encode(path.read_bytes()).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        ],
    }


def worker(index, deadline, pool, count, out, lock, errors):
    rng = random.Random(1000 + index)
    turn = 0
    session_no = 0
    while time.time() < deadline:
        session = probe.Session(f"soak{index}_{session_no}", pool, start=rng.randrange(len(pool)), count=count)
        session.add_reads(8000)
        limit = rng.randrange(40000, 90000)
        session_no += 1
        while session.tokens < limit and time.time() < deadline:
            turn += 1
            thinking = turn % 5 == 0
            if turn % 15 == 0:
                question = image_message(rng)
            else:
                question = {"role": "user", "content": f"Turn {turn}: summarise the last file in a few sentences."}
            messages = session.messages + [question]
            try:
                result = stream(messages, 700 if thinking else 200, thinking)
            except Exception as error:  # noqa: BLE001 - record and keep going
                with lock:
                    errors.append(repr(error))
                    out.write(json.dumps({"t": time.time(), "worker": index, "error": repr(error)}) + "\n")
                    out.flush()
                time.sleep(5)
                break
            record = {"t": round(time.time(), 1), "worker": index, "session": session.sid, "turn": turn,
                      "thinking": thinking, "image": turn % 15 == 0, "messages": len(messages), **result}
            reply = record.pop("text")
            thought = record.pop("reasoning", "")
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
            assistant = {"role": "assistant", "content": reply}
            if thought:
                assistant["reasoning_content"] = thought
            session.messages = messages + [assistant]
            body_chars = rng.randrange(1500, 5000)
            session.add_small_read(body_chars)
            session.tokens += body_chars // 4 + 300


def serve_pid():
    for proc in psutil.process_iter(["name"]):
        if proc.info["name"] == "ninfer-spo-serve.exe":
            return proc.pid
    return None


def wddm_usage(pid):
    counters = ["\\GPU Process Memory(*)\\Shared Usage", "\\GPU Process Memory(*)\\Dedicated Usage"]
    lines = subprocess.run(["typeperf", *counters, "-sc", "1"], capture_output=True, text=True).stdout.splitlines()
    rows = [line for line in lines if line.startswith('"')]
    if len(rows) < 2:
        return None, None
    header = [h.strip('"') for h in rows[0].split('","')]
    values = [v.strip('"') for v in rows[1].split('","')]
    shared = dedicated = 0.0
    for name, value in zip(header, values):
        if f"pid_{pid}_" not in name:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if name.endswith("Shared Usage"):
            shared += number
        elif name.endswith("Dedicated Usage"):
            dedicated += number
    return shared / 2**30, dedicated / 2**30


def monitor(deadline, path, stop):
    pynvml.nvmlInit()
    handle = None
    for i in range(pynvml.nvmlDeviceGetCount()):
        candidate = pynvml.nvmlDeviceGetHandleByIndex(i)
        if "RTX PRO 6000" in pynvml.nvmlDeviceGetName(candidate):
            handle = candidate
    with open(path, "a", encoding="utf-8") as out:
        while not stop.is_set():
            pid = serve_pid()
            record = {"t": round(time.time(), 1), "pid": pid}
            if pid:
                try:
                    proc = psutil.Process(pid)
                    mem = proc.memory_info()
                    record.update(rss_gib=round(mem.rss / 2**30, 3), private_gib=round(mem.private / 2**30, 3),
                                  handles=proc.num_handles(), threads=proc.num_threads())
                except psutil.Error as error:
                    record["proc_error"] = repr(error)
                shared, dedicated = wddm_usage(pid)
                record.update(wddm_shared_gib=round(shared, 3) if shared is not None else None,
                              wddm_dedicated_gib=round(dedicated, 3) if dedicated is not None else None)
            if handle is not None:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                record.update(nvml_used_gib=round(info.used / 2**30, 3),
                              power_w=round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1),
                              temp_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            samples = []
            deadline_sample = time.time() + 60
            while not stop.is_set() and time.time() < deadline_sample:
                sample = scrape_metrics()
                if "error" not in sample:
                    samples.append(sample)
                if stop.wait(10):
                    break
            def column(name):
                return [sample[name] for sample in samples if name in sample]
            kv = column("ninfer_device_kv_used_pages")
            host = column("ninfer_host_context_used_bytes")
            if kv:
                record.update(kv_floor=min(kv), kv_high=max(kv))
            if host:
                record.update(host_floor=min(host), host_high=max(host))
            if samples:
                record["kv_capacity"] = samples[-1].get("ninfer_device_kv_capacity_pages")
                record["host_capacity"] = samples[-1].get("ninfer_host_context_capacity_bytes")
            out.write(json.dumps(record) + "\n")
            out.flush()


def report_slopes(path):
    if not pathlib.Path(path).exists():
        print("no monitor samples")
        return 1
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    exit_code = 0
    for label, floor, high, capacity in (
        ("device_kv_used_pages", "kv_floor", "kv_high", "kv_capacity"),
        ("host_context_used_bytes", "host_floor", "host_high", "host_capacity"),
    ):
        points = [(index, row[floor]) for index, row in enumerate(rows) if row.get(floor) is not None]
        highs = [row[high] for row in rows if row.get(high) is not None]
        caps = [row[capacity] for row in rows if row.get(capacity)]
        if len(points) < 3:
            print(f"{label}: too few samples ({len(points)})")
            exit_code = 1
            continue
        slope, low, high_bound = ols_interval([p[0] for p in points], [p[1] for p in points])
        high_water = max(highs) if highs else points[-1][1]
        cap = caps[-1] if caps else 0
        verdict = slope_verdict(low, high_bound, high_water, cap)
        print(f"{label}: slope {slope:.4g} per minute, 95% [{low:.4g}, {high_bound:.4g}], "
              f"high-water {high_water:.4g} / capacity {cap:.4g}, {verdict}")
        if verdict == "leak":
            exit_code = 1
    return exit_code


def summarize(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    print(f"requests {len(ok)}, errors {len(errors)}")
    if not ok:
        return
    third = max(1, len(ok) // 3)
    for name, part in (("first third", ok[:third]), ("last third", ok[-third:])):
        warm = [r["ttft_ms"] for r in part if r["cache_n"]]
        gp99 = [r["gap_p99_ms"] for r in part if r["gap_p99_ms"] is not None]
        eng = [r["engine_decode_tps"] for r in part if r["engine_decode_tps"]]
        cli = [r["client_decode_tps"] for r in part if r["client_decode_tps"]]
        print(f"{name}: warm TTFT median {statistics.median(warm) if warm else None} ms, "
              f"gap p99 median {statistics.median(gp99) if gp99 else None} ms, "
              f"engine decode median {statistics.median(eng) if eng else None}, "
              f"client chunk rate median {statistics.median(cli) if cli else None}")


def detach(out):
    flags = 0x00000008 | 0x00000200 | 0x08000000
    log_path = pathlib.Path(out + ".run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child = [sys.executable, str(pathlib.Path(__file__).resolve())]
    child += [arg for arg in sys.argv[1:] if arg != "--detach"]
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("launch: " + " ".join(child) + "\n")
        log.flush()
        subprocess.Popen(child, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
    print(f"detached; log {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--selftest", action="store_true",
                        help="check the slope estimator on a flat series and a rising series, then exit")
    parser.add_argument("--detach", action="store_true",
                        help="relaunch with no console, appending to <out>.run.log, and return")
    args = parser.parse_args()
    slope_selftest()
    if args.selftest:
        return 0
    if not args.minutes or not args.out:
        parser.error("--minutes and --out are required unless --selftest")
    if args.detach:
        detach(args.out)
        return 0
    pool = probe.chunk_pool()
    count = probe.Counter()
    deadline = time.time() + args.minutes * 60
    stop = threading.Event()
    mon = threading.Thread(target=monitor, args=(deadline, args.out + ".monitor.jsonl", stop), daemon=True)
    mon.start()
    lock = threading.Lock()
    errors = []
    with open(args.out + ".requests.jsonl", "a", encoding="utf-8") as out:
        threads = [threading.Thread(target=worker, args=(i, deadline, pool, count, out, lock, errors)) for i in range(args.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    stop.set()
    mon.join(timeout=90)
    summarize(args.out + ".requests.jsonl")
    code = report_slopes(args.out + ".monitor.jsonl")
    if errors:
        print(f"{len(errors)} request errors; first: {errors[0]}")
        code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
