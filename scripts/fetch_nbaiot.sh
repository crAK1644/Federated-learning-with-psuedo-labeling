#!/usr/bin/env bash
# Fetch and unpack the N-BaIoT dataset into the nested layout ssfl.data.discovery expects:
#
#   data/device_info.csv
#   data/<DeviceName>/benign_traffic.csv
#   data/<DeviceName>/gafgyt_attacks/{combo,junk,scan,tcp,udp}.csv
#   data/<DeviceName>/mirai_attacks/{ack,scan,syn,udp,udpplain}.csv   (7 of 9 devices)
#
# The UCI archive ships the attack CSVs inside per-device RAR files and carries no device_info.csv,
# so both are handled here. Idempotent: re-running skips work that is already done.
#
# Usage:
#   scripts/fetch_nbaiot.sh                 # download, then unpack into data/
#   scripts/fetch_nbaiot.sh --zip PATH      # reuse an already-downloaded archive
#   scripts/fetch_nbaiot.sh --out DIR       # unpack somewhere other than data/

set -euo pipefail

URL="https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/data"
ZIP=""
MIN_FREE_GIB=15

while [ $# -gt 0 ]; do
  case "$1" in
    --zip) ZIP="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ponytail: bsdtar is the RAR reader -- libarchive has RAR support built in and ships with macOS,
# so this avoids depending on unar/unrar or the rarfile package just to unpack 17 archives.
command -v bsdtar >/dev/null || { echo "bsdtar not found (needed to read the .rar attack files)" >&2; exit 1; }

free_gib=$(df -g "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
if [ "$free_gib" -lt "$MIN_FREE_GIB" ]; then
  echo "only ${free_gib} GiB free, need ${MIN_FREE_GIB}; refusing to unpack" >&2
  exit 1
fi

mkdir -p "$OUT"

if [ -z "$ZIP" ]; then
  ZIP="$OUT/.nbaiot.zip"
  [ -f "$ZIP" ] || curl -fL --retry 3 -o "$ZIP" "$URL"
fi

# Devices are identified only by directory name in the archive; discovery needs numeric ids.
# Sorted order puts the two six-class devices (Ennio_Doorbell, Samsung_SNH_1011_N_Webcam) at
# positions 3 and 7, matching device_class_map in the shipped dataset_manifest.json.
if [ ! -f "$OUT/device_info.csv" ] || [ ! -d "$OUT/Danmini_Doorbell" ]; then
  unzip -q -o "$ZIP" -x "__MACOSX/*" -d "$OUT"
fi

for rar in "$OUT"/*/[gm]*_attacks.rar; do
  [ -e "$rar" ] || continue
  dest="${rar%.rar}"
  mkdir -p "$dest"
  bsdtar -xf "$rar" -C "$dest"
  rm -f "$rar"
done

{
  echo "DeviceID,DeviceName"
  i=0
  for d in "$OUT"/*/; do
    name="$(basename "$d")"
    i=$((i + 1))
    echo "$i,$name"
  done
} > "$OUT/device_info.csv"

found=$(find "$OUT" -name '*.csv' -not -name device_info.csv -not -name 'demonstrate_structure.csv' | wc -l | tr -d ' ')
echo "unpacked into $OUT -- $found source CSVs (expected 89)"
[ "$found" -eq 89 ] || { echo "unexpected file count" >&2; exit 1; }
