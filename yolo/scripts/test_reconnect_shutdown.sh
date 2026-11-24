#!/usr/bin/env bash
# Hardware test: viewer connect, disconnect, reconnect, then SIGINT shutdown
# and immediate camera reuse. Results go to <experiment>/reconnect_shutdown.txt.
# Usage: scripts/test_reconnect_shutdown.sh [yolo_stream args...]
set -u
cd "$(dirname "$0")/.."
py=${PYTHON:-.venv/bin/python}
before=$(ls -d experiments/*/ 2>/dev/null | wc -l)
"$py" -m yolo_stream --label reconnect-shutdown "$@" &
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
out="${exp}reconnect_shutdown.txt"
say() { echo "$@" | tee -a "$out"; }
say "date: $(date -Is)"
say "viewer 1:             $(python3 scripts/rtsp_probe.py --seconds 8)"
sleep 3
say "viewer 2 (reconnect): $(python3 scripts/rtsp_probe.py --seconds 8)"
sleep 1
say "viewer 3 (reconnect): $(python3 scripts/rtsp_probe.py --seconds 8)"
t0=$(date +%s.%N)
kill -INT $app
wait $app
code=$?
t1=$(date +%s.%N)
say "SIGINT -> exit: $(echo "$t1 - $t0" | bc) s, exit code $code"
if timeout 20 gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 num-buffers=30 ! \
     'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! fakesink >/dev/null 2>&1; then
  say "camera reopened immediately after shutdown: yes (30 frames captured)"
else
  say "camera reopened immediately after shutdown: NO"
fi
echo "experiment: $exp"
[ "$code" -eq 0 ]
