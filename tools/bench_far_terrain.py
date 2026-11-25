#!/usr/bin/env python3
"""Like-for-like far-terrain benchmark: Sunrise + Distant Sunrises vs a baseline Sunrise.

Both rows run the same seed, camera path and render distance through one Sunrise bench driver
(by default `tools/mod_pipeline_bench.py` in a Sunrise checkout, the driver used for
`/box/shared/sunrise/distant-horizons/bench-5185/`). While each row runs, this script samples the
game process for:

  * frame time p50 / p95 / p99 (ms), read from the driver's per-row JSON (`frame_ms` list, or the
    driver's own `p50_ms` / `p95_ms` / `p99_ms` keys),
  * server-thread CPU % (`/proc/<pid>/task/*/stat` of threads whose comm contains "Server"),
  * peak memory RSS (`VmHWM` of the process, summed with `/proc/<pid>/status`),
  * LOD load time: seconds from world ready until the mod reports `lod_ready_s` (driver JSON), or,
    when the driver does not report it, the time until the log prints `--lod-ready-marker`.

The result is written as JSON and compared against a reference file with the same row layout
(default: the former built-in far-terrain / mod-pipeline run `bench-5185/after.json`). Only rows
and metrics present on both sides are compared; nothing is invented for a missing metric.

Usage:
  tools/bench_far_terrain.py --sunrise /path/to/sunrise --mod-dir . --seed 1234 \
      --camera 0,100,0 --distance 32 --runs 3 --label after
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_REFERENCE = Path("/box/shared/sunrise/distant-horizons/bench-5185/after.json")
ROWS = {"baseline": [], "distant-sunrises": ["--mod", "distantsunrises"]}
CLK_TCK = os.sysconf("SC_CLK_TCK")


def descendants(pid):
    """`pid` plus every live descendant (the driver launches the game as a child)."""
    out, frontier = [pid], [pid]
    while frontier:
        parent = frontier.pop()
        for task in Path(f"/proc/{parent}/task").glob("*/children"):
            for child in task.read_text().split():
                out.append(int(child))
                frontier.append(int(child))
    return out


def thread_cpu_ticks(pid):
    """{(pid, tid, comm): utime+stime ticks} for every thread of `pid`."""
    ticks = {}
    for stat in Path(f"/proc/{pid}/task").glob("*/stat"):
        try:
            raw = stat.read_text()
        except OSError:
            continue
        comm = raw[raw.index("(") + 1 : raw.rindex(")")]
        fields = raw[raw.rindex(")") + 2 :].split()
        ticks[(pid, int(stat.parent.name), comm)] = int(fields[11]) + int(fields[12])
    return ticks


def rss_hwm_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


class Sampler(threading.Thread):
    """Tracks server-thread CPU time and peak RSS of the driver's process tree."""

    def __init__(self, root_pid):
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.stop = threading.Event()
        self.first = {}
        self.last = {}
        self.peak_rss_kib = 0
        self.t0 = time.monotonic()
        self.t1 = self.t0

    def run(self):
        while not self.stop.is_set():
            rss = 0
            for pid in descendants(self.root_pid):
                rss += rss_hwm_kib(pid)
                for key, ticks in thread_cpu_ticks(pid).items():
                    self.first.setdefault(key, ticks)
                    self.last[key] = ticks
            self.peak_rss_kib = max(self.peak_rss_kib, rss)
            self.t1 = time.monotonic()
            self.stop.wait(0.25)

    def server_cpu_percent(self):
        wall = max(self.t1 - self.t0, 1e-9)
        ticks = sum(
            self.last[k] - self.first[k] for k in self.last if "server" in k[2].lower()
        )
        return 100.0 * (ticks / CLK_TCK) / wall


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def frame_stats(row):
    frames = row.get("frame_ms")
    if frames:
        return {
            "p50_ms": percentile(frames, 0.50),
            "p95_ms": percentile(frames, 0.95),
            "p99_ms": percentile(frames, 0.99),
        }
    return {k: row[k] for k in ("p50_ms", "p95_ms", "p99_ms") if k in row}


def run_row(args, name, run_index):
    out_json = Path(args.out_dir) / f"{args.label}-{name}-{run_index}.json"
    cmd = [
        sys.executable,
        str(Path(args.sunrise) / args.driver),
        "--label", f"{args.label}-{name}-{run_index}",
        "--runs", "1",
        "--seed", str(args.seed),
        "--camera", args.camera,
        "--distance", str(args.distance),
        "--output", str(out_json),
    ]
    if args.offscreen:
        cmd.append("--offscreen")
    for extra in ROWS[name]:
        cmd.append(extra)
    if name != "baseline":
        cmd += ["--mod-dir", str(Path(args.mod_dir).resolve())]
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=args.sunrise, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    sampler = Sampler(proc.pid)
    sampler.start()
    lod_ready_wall = None
    for line in proc.stdout:
        if lod_ready_wall is None and args.lod_ready_marker in line:
            lod_ready_wall = time.monotonic() - started
    rc = proc.wait()
    sampler.stop.set()
    sampler.join()
    if rc != 0:
        raise SystemExit(f"driver failed for row {name} run {run_index}: exit {rc}")
    data = json.loads(out_json.read_text())
    rows = [r for run in data["runs"] for r in run]
    row = next((r for r in rows if r.get("row") == name), rows[-1])
    result = {
        "row": name,
        **frame_stats(row),
        "server_cpu_percent": sampler.server_cpu_percent(),
        "rss_mib": sampler.peak_rss_kib / 1024.0,
    }
    ready = row.get("ready_s")
    lod = row.get("lod_ready_s")
    if lod is not None and ready is not None:
        result["lod_load_s"] = max(lod - ready, 0.0)
    elif lod_ready_wall is not None and ready is not None:
        result["lod_load_s"] = max(lod_ready_wall - ready, 0.0)
    return result


METRICS = ["p50_ms", "p95_ms", "p99_ms", "server_cpu_percent", "rss_mib", "lod_load_s"]


def medians(results):
    out = {}
    for name in ROWS:
        mine = [r for r in results if r["row"] == name]
        out[name] = {
            m: statistics.median(r[m] for r in mine if m in r)
            for m in METRICS
            if any(m in r for r in mine)
        }
    return out


def reference_medians(path):
    data = json.loads(Path(path).read_text())
    rows = [r for run in data["runs"] for r in run]
    out = {}
    for row in rows:
        stats = out.setdefault(row["row"], {})
        for key, value in frame_stats(row).items():
            stats.setdefault(key, []).append(value)
    return {
        row: {k: statistics.median(v) for k, v in stats.items()} for row, stats in out.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sunrise", required=True, help="Sunrise checkout containing the bench driver")
    ap.add_argument("--driver", default="tools/mod_pipeline_bench.py")
    ap.add_argument("--mod-dir", default=".", help="Distant Sunrises checkout to load")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--camera", default="0,100,0", help="x,y,z of the fixed camera")
    ap.add_argument("--distance", type=int, default=32, help="render distance in chunks")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="far-terrain")
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--out-dir", default="bench-out")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--lod-ready-marker", default="distantsunrises: lod ready")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    # Alternate rows per run so thermal / cache drift hits both sides equally.
    for i in range(args.runs):
        for name in ROWS:
            results.append(run_row(args, name, i))
    summary = medians(results)
    reference = reference_medians(args.reference) if Path(args.reference).exists() else {}
    report = {
        "label": args.label,
        "seed": args.seed,
        "camera": args.camera,
        "distance_chunks": args.distance,
        "runs": results,
        "medians": summary,
        "reference": {"path": args.reference, "medians": reference},
    }
    Path(args.out_dir, f"{args.label}.json").write_text(json.dumps(report, indent=2))

    print("| metric | baseline | distant-sunrises | reference (former built-in) |")
    print("| --- | --- | --- | --- |")
    ref_row = reference.get("mod, warm") or reference.get("mod, cold") or {}
    for m in METRICS:
        b = summary["baseline"].get(m)
        d = summary["distant-sunrises"].get(m)
        r = ref_row.get(m)
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"| {m} | {fmt(b)} | {fmt(d)} | {fmt(r)} |")


if __name__ == "__main__":
    main()
