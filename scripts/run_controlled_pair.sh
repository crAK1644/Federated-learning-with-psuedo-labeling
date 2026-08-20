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
#   bash scripts/run_controlled_pair.sh                          # ~12 h, seed 2023
#   bash scripts/run_controlled_pair.sh configs/experiments_controlled_pair_seed2024.yaml 2024
#
# The smoke is not optional on a host that has never run this experiment: it is the only cheap check
# that the scenario-4 roots load, that 89 supernodes fit in memory, and that the audit payload the
# offline ledger reads is the one the run writes.
#
# Expects an RTX 3090 or equivalent: configs/controlled_pair.yaml requests 0.125 GPU per ClientApp
# actor with 8 concurrent, the split measured to saturate a 24 GB card for this model. The host also
# needs 32 GB of RAM and 20 GB of free disk -- both are gated below, for reasons that are written
# next to each gate.
#
# On a fresh host, before any of the above:
#
#   git clone -b dawid-skene-validation <this repo> && cd Federated-learning-with-psuedo-labeling
#   uv sync                                  # clone into a path with no spaces; see CLAUDE.md
#   cp -r /path/to/nbaiot-csvs data          # the 89 raw N-BaIoT CSVs, flat
#   uv run python -m ssfl.data.prepare_data --input data --output artifacts/data --seed 2023
#
# then the two controlled roots, which the data gate below prints in full if they are missing.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Optional args: a different matrix (the seed replicates) and the seed to score it at. The seed is
# not derived from the filename -- criteria run against whichever seed is named, and a wrong guess
# there is a silently mixed comparison.
MATRIX="${1:-$REPO_ROOT/configs/experiments_controlled_pair.yaml}"
SEED="${2:-}"
LOG_DIR="$REPO_ROOT/artifacts/logs"
MATRIX_TAG="$(basename "$MATRIX" .yaml)"
RUN_LOG="$LOG_DIR/$MATRIX_TAG.log"
STATUS_FILE="$LOG_DIR/$MATRIX_TAG.status"
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
  [[ -f "$REPO_ROOT/$root/scenarios/4.json" ]] || {
    printf 'PREFLIGHT FAILED: %s has no scenarios/4.json. Prepare both roots first:\n' "$root" |
      tee "$STATUS_FILE"
    cat <<'PREP'

  uv run python -m ssfl.data.prepare_data --input data --output artifacts/data-balanced \
      --seed 2023 --target-classes 1 2 --target-specialization 0.0
  uv run python -m ssfl.data.prepare_data --input data --output artifacts/data-specialist \
      --seed 2023 --target-classes 1 2 --target-specialization 1.0

Same seed, same splits, same scaler; only the client assignment manifest differs. Both need the
raw N-BaIoT CSVs under ./data. Roughly ten minutes each.
PREP
    exit 1
  }
done

# 3. Disk. Checkpoints dominate at roughly 26 MB per checkpointed round, and controlled_pair.yaml
#    checkpoints every round (same as the proven 200-round profiles: mid-run resume is not wired,
#    so checkpoints are the only record of a partial run). That is ~1.3 GB per arm, ~8 GB for the
#    matrix, plus room for Ray's object spill. Flower also provisions a fresh ~500 MB virtualenv
#    per run under ~/.flwr/runtime-envs and never reaps it -- another ~3 GB across the six arms,
#    on whichever filesystem $HOME lives on.
# `df -Pk` and not `df -g`: -g is a BSD flag that GNU df rejects, and this script is written on a
# Mac to run on a Linux box. -P also stops GNU df wrapping long device names onto a second line.
avail_gb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print int($4 / 1048576)}')"
[[ "${avail_gb:-0}" -ge 20 ]] ||
  fail "only ${avail_gb}G free; the matrix needs headroom (\`rm -rf ~/.flwr/runtime-envs\` if that is where it went)"

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
"$PYTHON" -m scripts.controlled_pair_criteria --runs "$REPO_ROOT/artifacts/runs" \
  ${SEED:+--seed "$SEED"} | tee -a "$RUN_LOG"
criteria_exit=${PIPESTATUS[0]}

if [[ $criteria_exit -eq 0 ]]; then
  printf 'SUCCESS: every pre-registered criterion passed %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" |
    tee "$STATUS_FILE"
else
  printf 'COMPLETE: matrix finished, criteria not all met (see above) %s\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" | tee "$STATUS_FILE"
fi

exit 0
