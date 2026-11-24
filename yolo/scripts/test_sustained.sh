#!/usr/bin/env bash
# Hardware test: sustained run with a viewer decoding the stream throughout.
# Writes viewer.json, tegrastats.log and a short clip into the experiment folder.
# Usage: scripts/test_sustained.sh [SECONDS=300] [yolo_stream args...]
set -u
cd "$(dirname "$0")/.."
py=${PYTHON:-.venv/bin/python}
secs=${1:-300}; [ $# -gt 0 ] && shift
before=$(ls -d experiments/*/ 2>/dev/null | wc -l)
"$py" -m yolo_stream --label sustained "$@" --duration "$secs" &
app=$!
exp=""
for _ in $(seq 1 120); do
  if [ "$(ls -d experiments/*/ 2>/dev/null | wc -l)" -gt "$before" ]; then
    exp=$(ls -dt experiments/*/ | head -1)
    grep -q "processing started" "$exp/run.log" 2>/dev/null && break
  fi
  kill -0 $app 2>/dev/null || { echo "app exited during start-up"; exit 1; }
  sleep 1
done
tegrastats --interval 5000 > "${exp}tegrastats.log" 2>/dev/null &
stats=$!
python3 scripts/rtsp_probe.py --seconds $((secs - 25)) > "${exp}viewer.json" &
probe=$!
sleep 5
scripts/record_stream.sh "${exp}stream_sample.mp4" 10   # second, simultaneous viewer
wait $probe
wait $app
code=$?
kill $stats 2>/dev/null
echo "viewer: $(cat "${exp}viewer.json")"
echo "app exit code: $code   experiment: $exp"
exit $code
