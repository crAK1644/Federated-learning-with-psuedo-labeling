#!/usr/bin/env bash
#
# One command for the controlled target-pair experiment (follow-up plan stages 5 and 8).
#
# Everything before the matrix is a pre-flight check, and every one exists because failing after
# eleven hours of GPU time costs more than failing in the first second. The matrix is resumable, so
# re-running this after an interruption picks up at the first unfinished arm.
#
#   uv run flwr run . --run-config 'profile="controlled_pair_smoke" algorithm="ssfl" scenario=4' \
#     --federation-config 'num-supernodes=89' --stream          # ~2 min, run this first
#   bash scripts/run_controlled_pair.sh                          # ~12 h
#
# The smoke is not optional on a host that has never run this experiment: it is the only cheap check
# that the scenario-4 roots load, that 89 supernodes fit in memory, and that the audit payload the
# offline ledger reads is the one the run writes.
#
# Expects an RTX 3090 or equivalent: configs/controlled_pair.yaml requests 0.125 GPU per ClientApp
# actor with 8 concurrent, the split measured to saturate a 24 GB card for this model.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MATRIX="$REPO_ROOT/configs/experiments_controlled_pair.yaml"
LOG_DIR="$REPO_ROOT/artifacts/logs"
RUN_LOG="$LOG_DIR/controlled_pair.log"
STATUS_FILE="$LOG_DIR/controlled_pair.status"
PYTHON="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT" || exit 1
mkdir -p "$LOG_DIR"

fail() { printf 'PREFLIGHT FAILED: %s\n' "$1" | tee "$STATUS_FILE"; exit 1; }

# Ordered cheapest and most portable first, so everything except the card can be verified from a
# machine that does not have one.

# 1. The interpreter. .venv is a symlink to a space-free path on this host; a fresh `uv sync` blows
#    that away and every `flwr run` then dies with an opaque "no such table: fab".
[[ -x "$PYTHON" ]] || fail "no interpreter at $PYTHON (see CLAUDE.md on the .venv symlink)"
case "$(readlink "$REPO_ROOT/.venv" || echo "$REPO_ROOT/.venv")" in
  *\ *) fail "the venv path contains a space; Flower's Alembic config will find zero migrations" ;;
esac

# 2. The two data roots, each with the scenario-4 assignment the arms load. Checking for the
#    directory is not enough: a root prepared without --target-classes looks complete and is
#    missing exactly the file this experiment needs.
for root in artifacts/data-balanced artifacts/data-specialist; do
  [[ -f "$REPO_ROOT/$root/scenarios/4.json" ]] ||
    fail "$root has no scenario 4; re-run prepare_data with --target-classes 1 2"
done

# 3. Disk. Checkpoints dominate at roughly 26 MB per checkpointed round, and controlled_pair.yaml
#    checkpoints every round (same as the proven 200-round profiles: mid-run resume is not wired,
#    so checkpoints are the only record of a partial run). That is ~1.3 GB per arm, ~8 GB for the
#    matrix, plus room for Ray's object spill.
avail_gb="$(df -g "$REPO_ROOT" | awk 'NR==2 {print $4}')"
[[ "${avail_gb:-0}" -ge 20 ]] || fail "only ${avail_gb}G free; the matrix needs headroom"

# 4. Memory. Scenario 4 is 89 clients and every one of them participates in every round -- there is
#    no participation fraction to turn down -- so the simulation holds 89 ClientApp processes at
#    once. A bare `import torch` plus this app measures 327 MB resident, which is ~29 GB before any
#    training data is loaded. A 16 GB host does not survive round 1: it dies during Ray actor
#    startup, and `flwr run` still exits 0, leaving a run directory with nothing but run_start in
#    events.jsonl. Checked here because that failure is silent everywhere else.
if [[ -r /proc/meminfo ]]; then
  ram_gb="$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)"
else
  ram_gb="$(( $(sysctl -n hw.memsize) / 1073741824 ))"
fi
[[ "${ram_gb:-0}" -ge 32 ]] ||
  fail "${ram_gb}G RAM; 89 concurrent ClientApps need ~29 GB resident, 32 GB minimum"

# 5. The card, last. `device: cuda` in the profile is a request, not a guarantee -- without this
#    check the matrix silently trains on CPU and takes days instead of hours.
"$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' ||
  fail "no CUDA device visible to torch"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RAY_local_fs_capacity_threshold=0.98
export RAY_DEDUP_LOGS=0

printf 'RUNNING: matrix started %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" | tee "$STATUS_FILE"
printf 'watch progress with: python3 scripts/watch_progress.py\n'

"$PYTHON" -m ssfl.experiments.run_suite --matrix "$MATRIX" --resume >>"$RUN_LOG" 2>&1
matrix_exit=$?

if [[ $matrix_exit -ne 0 ]]; then
  printf 'FAILED: matrix exit=%s, see %s\n' "$matrix_exit" "$RUN_LOG" | tee "$STATUS_FILE"
  exit "$matrix_exit"
fi

# The ledger scores against sealed open-set labels, which is why it is offline rather than part of
# the run. `-m` and not a path: it imports scripts.ds_headroom. No --data: round_metrics reads the
# root out of each run's own resolved_config.yaml, so the balanced and specialist arms are each
# scored against the labels they actually trained on.
printf 'SCORING: building per-round ledgers %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" | tee "$STATUS_FILE"
for run_dir in "$REPO_ROOT"/artifacts/runs/*controlled_*; do
  [[ -d "$run_dir" ]] || continue
  "$PYTHON" -m scripts.round_metrics "$run_dir" >>"$RUN_LOG" 2>&1 ||
    printf 'warning: ledger failed for %s\n' "$(basename "$run_dir")" | tee -a "$RUN_LOG"
done

# Pre-registered criteria. Non-zero here is a real answer about the experiment, not a broken run.
"$PYTHON" -m scripts.controlled_pair_criteria --runs "$REPO_ROOT/artifacts/runs" | tee -a "$RUN_LOG"
criteria_exit=${PIPESTATUS[0]}

if [[ $criteria_exit -eq 0 ]]; then
  printf 'SUCCESS: every pre-registered criterion passed %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" |
    tee "$STATUS_FILE"
else
  printf 'COMPLETE: matrix finished, criteria not all met (see above) %s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" | tee "$STATUS_FILE"
fi

exit 0
