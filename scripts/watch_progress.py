#!/usr/bin/env python3
"""Live progress bar for a run_suite matrix: per-entry rounds, percent, and ETA.

Reads the server's events.jsonl of each entry's run directory, which is appended per round while
the run is in flight -- metrics.parquet and summary.json only appear at the end, so they are
useless for watching. Stdlib only, so it runs under the host's plain python3 without uv.

    python3 scripts/watch_progress.py                                   # 30s refresh
    python3 scripts/watch_progress.py 10                                # 10s refresh
    python3 scripts/watch_progress.py 30 configs/experiments_other.yaml
"""

from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO / "configs" / "experiments_dawid_skene_50.yaml"
TAIL_BYTES = 200_000


def entry_names(matrix: Path) -> list[str]:
    """Matrix entry names, in file order -- run_suite runs them in exactly that sequence."""
    return re.findall(r"^\s*-\s*name:\s*(\S+)", matrix.read_text(), flags=re.MULTILINE)


def queued_rounds(matrix: Path) -> int:
    """Rounds to assume for an entry whose run directory does not exist yet.

    Read from the matrix's own base profiles rather than assumed, so the ETA is right from the
    first refresh instead of only after the first run directory appears -- the gap between a
    50-round and a 200-round matrix is six hours of pending work.
    """
    for profile in re.findall(r"^\s*base_profile:\s*(\S+)", matrix.read_text(), flags=re.MULTILINE):
        config = REPO / "configs" / f"{profile}.yaml"
        match = config.exists() and re.search(r"num_server_rounds:\s*(\d+)", config.read_text())
        if match:
            return int(match.group(1))
    return 0


def tail_lines(path: Path, limit: int = TAIL_BYTES) -> list[str]:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - limit))
        # The first line is probably cut mid-record when the file is larger than the window.
        return stream.read().decode("utf-8", "replace").splitlines()[1:]


def first_timestamp(path: Path) -> float | None:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                return float(json.loads(line)["ts"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return None


def progress(run_dir: Path) -> tuple[int, float | None, float | None]:
    """(latest round, first event ts, latest event ts) across this run's attempts."""
    events = sorted(run_dir.glob("attempts/*/events.jsonl"))
    if not events:
        return 0, None, None
    latest_round, started, last_ts = 0, first_timestamp(events[0]), None
    for path in events:
        for line in tail_lines(path):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial final line while the writer is mid-append
            if isinstance(record.get("round"), int):
                latest_round = max(latest_round, record["round"])
            if isinstance(record.get("ts"), (int, float)):
                last_ts = record["ts"]
    return latest_round, started, last_ts


def total_rounds(run_dir: Path) -> int:
    match = re.search(r"num_server_rounds:\s*(\d+)", (run_dir / "resolved_config.yaml").read_text())
    return int(match.group(1)) if match else 0


def bar(done: int, total: int, width: int = 34) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


def clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def render(matrix: Path) -> str:
    lines = [f"{time.strftime('%H:%M:%S')}  {matrix.name}", ""]
    done_total = pending_total = 0
    rates: list[float] = []
    queued = queued_rounds(matrix)

    for name in entry_names(matrix):
        matches = glob.glob(str(REPO / "artifacts" / "runs" / f"ssfl-s*-{name}-*"))
        if not matches:
            lines.append(f"  {name:<14} queued")
            pending_total += queued
            continue
        run_dir = Path(matches[0])
        total = total_rounds(run_dir)
        done, started, last_ts = progress(run_dir)
        complete = (run_dir / "summary.json").exists()
        done_total += done
        pending_total += max(0, total - done)

        if complete:
            accuracy = json.loads((run_dir / "summary.json").read_text())
            value = accuracy.get("final_centralized_metrics", {}).get("accuracy")
            suffix = f"done   accuracy {value:.4f}" if value is not None else "done"
        elif done and started and last_ts and last_ts > started:
            rate = (last_ts - started) / done
            rates.append(rate)
            stale = time.time() - last_ts
            suffix = f"{rate:5.0f}s/round  eta {clock(rate * (total - done))}"
            if stale > 600:
                suffix += f"  (!) no events for {clock(stale)}"
        else:
            suffix = "starting"
        lines.append(f"  {name:<14} {bar(done, total)}  {100 * done / total if total else 0:5.1f}%  {suffix}")

    rate = sum(rates) / len(rates) if rates else 0.0
    overall = done_total + pending_total
    lines += [
        "",
        f"  {'overall':<14} {bar(done_total, overall)}  "
        f"{100 * done_total / overall if overall else 0:5.1f}%"
        + (f"  eta {clock(rate * pending_total)}" if rate else ""),
    ]
    return "\n".join(lines)


def main() -> None:
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    matrix = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MATRIX
    while True:
        print(f"\033[2J\033[H{render(matrix)}\n\n  refreshing every {interval:.0f}s -- Ctrl-C to stop", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
