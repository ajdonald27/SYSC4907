#!/usr/bin/env bash
# Act as an RTSP viewer: record the stream for N seconds to an MP4 without
# re-encoding. Used as evidence that the stream is playable and for sample video.
# Usage: scripts/record_stream.sh OUT.mp4 [SECONDS] [URL]
set -u
out=${1:?usage: record_stream.sh OUT.mp4 [SECONDS] [URL]}
secs=${2:-10}
url=${3:-rtsp://127.0.0.1:8554/detections}
timeout -s INT "$secs" gst-launch-1.0 -e -q \
  rtspsrc location="$url" protocols=tcp latency=200 ! \
  rtph264depay ! h264parse ! mp4mux ! filesink location="$out"
if [ -s "$out" ]; then
  frames=$(gst-discoverer-1.0 -v "$out" 2>/dev/null | grep -E 'Duration|Width|Height|Frame rate' | tr -s ' ' | tr '\n' ';')
  echo "recorded $(stat -c %s "$out") bytes to $out: $frames"
else
  echo "FAILED: no video recorded from $url" >&2
  exit 1
fi
