#!/usr/bin/env python3
"""Measure a real Sunrise single-player world, with and without Distant Sunrises.

Linux with /proc and a dedicated X display. Builds nothing: use a trusted
cargo-sunrise-mod run app stage, and sunrise-ctl from the matching game release.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

MOD = "distantsunrises"
PACKAGE_DIR = "sunrise-native-mods"
RECORD = "native-mods.json"
GAME = "sunrise-client"
WORLD = "far-terrain"
PLAYER = "FarTerrainBench"
FRAME_TIMES = ("p50_ms", "p95_ms", "p99_ms")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def file_hashes(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def native_stage(app, release, sdk):
    package = app / PACKAGE_DIR / MOD
    with (package / "sunrise-mod.toml").open("rb") as file:
        manifest = tomllib.load(file)
    if set((app / PACKAGE_DIR).iterdir()) != {package} or manifest["id"] != MOD:
        raise ValueError(f"stage must contain only the {MOD} native package")
    commit = release["commit"]
    build = (commit, release["version_id"])
    if (sdk["build"]["commit"], sdk["build"]["game"]) != build:
        raise ValueError("game release and SDK metadata name different builds")
    if (manifest["native"]["sunrise"], manifest["native"]["game"]) != build:
        raise ValueError("Distant Sunrises package is pinned to a different game build")
    libraries = json.loads((app / RECORD).read_text())["libraries"]
    by_id = {lib["mod"]: lib for lib in libraries}
    if len(by_id) != len(libraries) or MOD not in by_id or by_id.keys() - {MOD, "sunrise_patch"}:
        raise ValueError("native record must name Distant Sunrises and at most the SDK patch library")
    omitted = {RECORD, PACKAGE_DIR}
    for lib in libraries:
        name = lib["file"]
        if name in omitted or not re.fullmatch(r"lib[^/]+\.so", name):
            raise ValueError(f"invalid staged native library: {name}")
        identity = lib["identity"].split()
        if identity[:2] != [commit, lib["mod"]] or identity[3] != release["rust"]:
            raise ValueError(f"{name} does not match the release commit/Rust ABI")
        if sha256(app / name) != lib["sha256"]:
            raise ValueError(f"{name} does not match its native record hash")
        omitted.add(name)
    if by_id[MOD]["identity"].split()[2] != manifest["version"]:
        raise ValueError("native library version does not match the staged manifest")
    for item in app.iterdir():
        if item.name not in omitted | {GAME} and not (
            item.is_file() and re.fullmatch(r"lib[^/]+\.so(?:\.\d+)*", item.name)
        ):
            raise ValueError(f"unexpected SDK app entry: {item.name}")
    return by_id[MOD], omitted


def server_snapshot(pid):
    sampled = time.monotonic()
    threads = []
    for stat in Path(f"/proc/{pid}/task").glob("*/stat"):
        try:
            line = stat.read_text()
        except FileNotFoundError:  # An unrelated thread can exit during enumeration.
            continue
        end = line.rindex(")")
        if line[line.index("(") + 1 : end] == "Server thread":
            fields = line[end + 2 :].split()  # Starts at field 3 (state).
            threads.append((int(stat.parent.name), int(fields[19]), int(fields[11]) + int(fields[12])))
    if len(threads) != 1:
        raise ValueError(f"expected one Server thread in game PID {pid}, found {len(threads)}")
    tid, start, ticks = threads[0]
    return {"tid": tid, "start_ticks": start, "cpu_ticks": ticks, "monotonic_s": sampled}


def scene(ctl, args, modded):
    state, status = ctl("state"), ctl("status")
    x, y, z, yaw, pitch = args.camera
    if (
        state["screen"] != "playing" or state["paused"] or state["gamemode"] != "spectator"
        or state["dimension"] != "minecraft:overworld" or state["player"]["position"] != [x, y, z]
        or state["player"]["yaw"] != yaw or state["player"]["pitch"] != pitch
        or state["weather"]["raining"] or state["weather"]["thundering"]
        or state["time"]["day_time"] != 6000 or status["render_distance"] != args.distance
        or status["server"]["address"] != "local" or not status["server"]["loaded"]
        or status["renderer"]["size"] != [1280, 720]
    ):
        raise ValueError("game did not reach the requested fixed single-player scene; see controls.jsonl")
    ds = None
    if modded:
        reply = ctl("command", "ds")
        match = re.match(
            r"Distant Sunrises \[ON\] dim=minecraft:overworld radius=(\d+)ch sections=(\d+) ",
            next(iter(reply["chat"]), ""),
        )
        if not reply["success"] or match is None:
            raise ValueError("the staged Distant Sunrises mod did not answer /ds with an enabled status")
        ds = {"radius_chunks": int(match[1]), "active_store_sections": int(match[2])}
    return {
        "near": {"loaded_chunks": status["chunks"], "pending_chunks": status["pending_chunks"],
                 "resident_sections": status["renderer"]["resident_sections"]},
        "sunrise_lod": status["lod"],
        "renderer": {k: status["renderer"][k] for k in (
            "gpu", "driver", "backend", "size", "present_mode", "resolution_scale", "dynamic_resolution",
        )},
        "distant_sunrises": ds,
    }


def run_row(args, inputs, root, out, lib, omitted, modded):
    out.mkdir()
    app, game_dir = root / "app", root / "game"
    staged_app = inputs / "app"
    shutil.copytree(staged_app, app, ignore=lambda path, _: omitted if not modded and Path(path) == staged_app else ())
    resources = root / "resources"
    shutil.copytree(inputs / "resources", resources)
    game_dir.mkdir()
    options = f"renderDistance:{args.distance}\nresourcePacks:[]\nlang:en_us\n"
    (game_dir / "options.txt").write_text(options)
    (out / "options.txt").write_text(options)
    socket = root / "ctl.sock"
    # Do not inherit account/session, SUNRISE_*, caches, LD_PRELOAD or player environment.
    env = {k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ}
    for key in ("HOME", "SUNRISE_HOME", "SUNRISE_LAUNCHER_HOME", "XDG_DATA_HOME",
                "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
        folder = root / key.lower()
        folder.mkdir(mode=0o700)
        env[key] = str(folder)
    env.update(LD_LIBRARY_PATH=str(app), SUNRISE_NO_AUDIO="1", SUNRISE_LAUNCHER_NO_SELF_UPDATE="1")
    command = [str(app / GAME), "--game-dir", str(game_dir), "--resources", str(resources),
               "--world", WORLD, "--seed", str(args.seed), "--name", PLAYER, "--creative",
               "--no-pause", "--width", "1280", "--height", "720", "--render-distance",
               str(args.distance), "--control-socket", str(socket)]
    write_json(out / "launch.json", command)
    with (out / "game.log").open("w") as log, (out / "controls.jsonl").open("w") as trace:
        def ctl(*words, timeout=30):
            words = tuple(map(str, words))
            result = subprocess.run(
                [str(args.ctl), "--socket", str(socket), "--timeout", str(timeout), *words],
                cwd=game_dir, env=env, capture_output=True, text=True, timeout=timeout + 5,
            )
            trace.write(json.dumps({"command": words, "exit": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr}) + "\n")
            trace.flush()
            result.check_returncode()
            return json.loads(result.stdout)

        launched = time.monotonic()
        proc = subprocess.Popen(command, cwd=game_dir, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            while not socket.is_socket():
                if proc.poll() is not None or time.monotonic() - launched > 300:
                    raise ValueError("Sunrise did not open its control socket; see game.log")
                time.sleep(0.1)
            hello = ctl("hello")
            ctl("wait-ready", 300, timeout=310)
            ready_s = time.monotonic() - launched
            maps = Path(f"/proc/{proc.pid}/maps").read_text()
            mappings = [line for line in maps.splitlines() if line.endswith(f"/{lib['file']}")]
            (out / "native-mappings.txt").write_text("\n".join(mappings) + "\n")
            if (modded and not any(line.endswith(f" {app / lib['file']}") for line in mappings)
                    or not modded and mappings):
                raise ValueError("game's mapped native library does not match the requested row")
            if not modded and ctl("command", "ds")["success"]:
                raise ValueError("the baseline unexpectedly answered /ds")
            for cmd in ("gamemode spectator", "gamerule minecraft:advance_time false",
                        "time set noon", "weather clear"):
                if not ctl("command", cmd)["success"]:
                    raise ValueError(f"game refused {cmd}")
            ctl("tp", *args.camera[:3])
            ctl("look", *args.camera[3:])
            ctl("wait", 20)
            ctl("wait-ready", 300, timeout=310)
            time.sleep(args.settle)
            before = scene(ctl, args, modded)
            start = server_snapshot(proc.pid)
            frames = ctl("bench", args.seconds, "raw", timeout=args.seconds + 30)
            end = server_snapshot(proc.pid)
            if (start["tid"], start["start_ticks"]) != (end["tid"], end["start_ticks"]):
                raise ValueError("Server thread changed during the sample")
            ticks = end["cpu_ticks"] - start["cpu_ticks"]
            elapsed = end["monotonic_s"] - start["monotonic_s"]
            hz = os.sysconf("SC_CLK_TCK")
            if ticks < 0 or elapsed <= 0 or hz <= 0:
                raise ValueError("invalid /proc Server thread CPU sample")
            if not frames["frames"] or len(frames["frame_ms"]) != frames["frames"] or not all(
                math.isfinite(frames[k]) and frames[k] > 0 for k in FRAME_TIMES
            ):
                raise ValueError("Sunrise bench returned no usable frame-time sample")
            after = scene(ctl, args, modded)
            if modded and not after["distant_sunrises"]["active_store_sections"]:
                raise ValueError("no active Distant Sunrises terrain after measurement")
            ctl("screenshot", out / "terrain.png")
            ctl("save")
            ctl("quit")
            if proc.wait(timeout=120) != 0:
                raise ValueError("Sunrise did not shut down cleanly; cache measurement is unavailable")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            latest = Path(env["SUNRISE_HOME"]) / "logs/latest.log"
            if latest.is_file():
                shutil.copy2(latest, out / "latest.log")

    save = game_dir / "saves" / WORLD
    if not (save / "level.dat").is_file():
        raise ValueError("scratch save missing after shutdown")
    regions = [p.stat() for p in (save / "distant_sunrises").rglob("*.dhoct") if p.is_file()]
    if bool(regions) != modded:
        raise ValueError("flushed .dhoct cache does not match the requested row")
    write_json(out / "frames.json", frames)
    row = {
        "row": "distant-sunrises" if modded else "baseline", "evidence": out.name,
        "hello": hello, "ready_s": ready_s, "coverage_start": before, "coverage_end": after,
        "frame_count": frames["frames"], **{k: frames[k] for k in FRAME_TIMES},
        "server_cpu_percent": 100 * ticks / hz / elapsed,
        "server_cpu_sample": {"start": start, "end": end, "clock_ticks_per_second": hz},
        "lod_cache": {"files": len(regions), "apparent_bytes": sum(s.st_size for s in regions),
                      "allocated_bytes": sum(s.st_blocks * 512 for s in regions)},
    }
    write_json(out / "row.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True, help="staged SDK run/app directory, not a source .smod")
    parser.add_argument("--ctl", type=Path, required=True, help="sunrise-ctl from the exact game release")
    parser.add_argument("--release", type=Path, required=True, help="matching verified sunrise-release.json")
    parser.add_argument("--sdk", type=Path, required=True, help="matching verified sdk.json")
    parser.add_argument("--resources", type=Path, required=True, help="prepared resources (read-only input)")
    parser.add_argument("--out", type=Path, required=True, help="new directory for private evidence")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--camera", type=int, nargs=5, default=(-525, 105, 64, 0, 18),
                        metavar=("X", "Y", "Z", "YAW", "PITCH"))
    parser.add_argument("--distance", type=int, default=4, help="near render distance in chunks")
    parser.add_argument("--settle", type=float, default=15, help="seconds after the fixed camera is ready")
    parser.add_argument("--seconds", type=float, default=20, help="seconds of sunrise-ctl bench per row")
    parser.add_argument("--runs", type=int, default=1, help="number of fresh baseline/mod pairs")
    args = parser.parse_args()
    if not Path("/proc/self/task").is_dir() or not os.environ.get("DISPLAY"):
        parser.error("run on Linux with /proc and a dedicated X display")
    if (not all(math.isfinite(v) for v in (args.settle, args.seconds))
            or not -90 <= args.camera[4] <= 90 or not -(2**63) <= args.seed < 2**63
            or not 2 <= args.distance <= 128 or not 0 < args.seconds <= 86400
            or args.settle < 0 or args.runs < 1):
        parser.error("invalid scene, seed, distance or measurement window")
    for attr in ("app", "ctl", "release", "sdk", "resources", "out"):
        setattr(args, attr, getattr(args, attr).resolve())
    temp = Path("/tmp").resolve()  # Keep the Unix control socket path short.
    if any(path.is_relative_to(p) for path in (args.out, temp) for p in (args.app, args.resources)):
        parser.error("output and temporary directory must not be inside supplied inputs")
    args.out.mkdir(mode=0o700, parents=True)

    with tempfile.TemporaryDirectory(prefix="ds-bench-", dir=temp) as scratch:
        inputs = Path(scratch) / "inputs"
        for key in ("app", "resources"):
            shutil.copytree(getattr(args, key), inputs / key)
        if not (inputs / "resources/sunrise-resources.json").is_file():
            raise ValueError("prepared resources must contain sunrise-resources.json")
        ctl = inputs / "sunrise-ctl"
        shutil.copy2(args.ctl, ctl)
        args.ctl = ctl
        release = json.loads(args.release.read_text())
        sdk = json.loads(args.sdk.read_text())
        lib, omitted = native_stage(inputs / "app", release, sdk)
        fingerprints = {
            "release": release, "sdk_build": sdk["build"], "native_mod": lib,
            "omitted_from_baseline_app": sorted(omitted),
            "release_sha256": sha256(args.release), "sdk_sha256": sha256(args.sdk),
            "control_sha256": sha256(ctl), "app_files_sha256": file_hashes(inputs / "app"),
            "resources_files_sha256": file_hashes(inputs / "resources"),
            "seed": args.seed, "camera": args.camera, "near_distance_chunks": args.distance,
            "settle_s": args.settle, "bench_s": args.seconds, "host": list(os.uname()),
            "starting_save": "fresh world; empty LOD cache, options and offline player isolated per row",
        }
        write_json(args.out / "inputs.json", fingerprints)
        pairs = []
        for i in range(args.runs):
            pair = {}
            for modded in (bool(i % 2), not bool(i % 2)):
                row = "distant-sunrises" if modded else "baseline"
                name = f"{i + 1}-{row}"
                root = Path(scratch) / name
                pair[row] = run_row(args, inputs, root, args.out / name, lib, omitted, modded)
                shutil.rmtree(root)
            baseline, mod = pair["baseline"], pair["distant-sunrises"]
            samples = ("coverage_start", "coverage_end")
            if baseline["hello"] != mod["hello"] or any(
                baseline[key]["renderer"] != mod[key]["renderer"] for key in samples
            ):
                raise ValueError("baseline and mod rows reported different game or renderer settings")
            pair["near_counts_match"] = {key: baseline[key]["near"] == mod[key]["near"] for key in samples}
            pairs.append(pair)
        write_json(args.out / "report.json", {
            "pairs": pairs,
            "comparison": "fixed-time cost only; no equal far coverage or speedup claimed",
            "unavailable": {"equal_far_coverage": "baseline has no Distant Sunrises terrain",
                            "lod_load_s": "no shared far-terrain readiness target"},
        })
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
