#!/usr/bin/env bash
# Watch the newest run in artifacts/runs: prints the current round every N seconds, then dumps
# the headline metrics once summary.json appears.
#
# Usage:
#   scripts/watch_run.sh              # newest run, 30s interval
#   scripts/watch_run.sh 10           # newest run, 10s interval
#   scripts/watch_run.sh 30 <run_dir> # a specific run

set -uo pipefail

INTERVAL="${1:-30}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${2:-}"

while [ -z "$RUN" ]; do
  RUN=$(ls -td "$REPO_ROOT"/artifacts/runs/*/ 2>/dev/null | head -1)
  [ -n "$RUN" ] || { echo "waiting for a run directory to appear..."; sleep "$INTERVAL"; }
done
echo "watching $RUN"

# ponytail: poll the event log rather than tailing it -- the round number is the only thing
# wanted, and a re-glob each tick also survives the attempt directory changing on a resume.
while [ ! -f "$RUN/summary.json" ]; do
  round=$(grep -ho '"round": *[0-9]*' "$RUN"attempts/*/events.jsonl 2>/dev/null | tail -1 | tr -dc 0-9)
  echo "$(date +%H:%M:%S)  round ${round:-<none yet>}"
  sleep "$INTERVAL"
done

echo
echo "=== finished ==="
cd "$REPO_ROOT" && uv run python - "$RUN" <<'PY'
import sys, json
import numpy as np
import pandas as pd

run = sys.argv[1]
s = json.load(open(f"{run}/summary.json"))
m = pd.read_parquet(f"{run}/metrics.parquet")
last = m.iloc[-1]

print(f"{s['run_id']}  ({s['final_round']} rounds)")
print(f"  accuracy         {last['accuracy']:.4f}   (best {m['accuracy'].max():.4f} "
      f"@ round {int(m.loc[m['accuracy'].idxmax(), 'round'])})")
print(f"  macro precision  {last['macro_precision']:.4f}")
print(f"  macro F1         {last['macro_f1']:.4f}")

# The gafgyt.tcp / gafgyt.udp pair (classes 4 and 5) is the one the normalisation work targets.
p = pd.read_parquet(f"{run}/per_class_metrics.parquet")
pair = p[(p["round"] == p["round"].max()) & (p["class"].isin([4, 5]))]
print("\nflooding pair:")
for _, r in pair.iterrows():
    name = {4: "gafgyt.tcp", 5: "gafgyt.udp"}[r["class"]]
    print(f"  {name}  precision {r['precision']:.4f}  recall {r['recall']:.4f}  f1 {r['f1']:.4f}")

z = np.load(f"{run}/confusion_matrices.npz")
c = z[max(z.keys(), key=lambda k: int(k.split("_")[1]))]
print(f"  confusion  tcp->[tcp {int(c[4, 4])}, udp {int(c[4, 5])}]  "
      f"udp->[tcp {int(c[5, 4])}, udp {int(c[5, 5])}]")
PY
