#!/usr/bin/env bash
# Run the app and, once it is processing, attach a local RTSP viewer that
# records REC_SECONDS of the stream into the experiment folder.
# Usage: scripts/run_with_viewer.sh REC_SECONDS [yolo_stream args...]
# Environment: PYTHON (default .venv/bin/python)
set -u
cd "$(dirname "$0")/.."
rec=${1:?usage: run_with_viewer.sh REC_SECONDS [yolo_stream args...]}; shift
py=${PYTHON:-.venv/bin/python}
before=$(ls -d experiments/*/ 2>/dev/null | wc -l)
"$py" -m yolo_stream "$@" &
app=$!
trap 'kill -INT $app 2>/dev/null' INT TERM
exp=""
for _ in $(seq 1 120); do
  if [ "$(ls -d experiments/*/ 2>/dev/null | wc -l)" -gt "$before" ]; then
    exp=$(ls -dt experiments/*/ | head -1)
    grep -q "processing started" "$exp/run.log" 2>/dev/null && break
  fi
  kill -0 $app 2>/dev/null || break
  sleep 1
done
if kill -0 $app 2>/dev/null && [ -n "$exp" ]; then
  scripts/record_stream.sh "${exp}stream_sample.mp4" "$rec"
fi
wait $app
code=$?
echo "app exit code: $code   experiment: $exp"
exit $code
