# RATS radar node

Runs on the PYNQ-Z2 (live radar) or on a laptop (replaying recordings):

frame → preprocess → range FFT (numpy or FPGA) → range-Doppler → detect and track (plus blade channel) → classify → UDP messages

| File | What it does |
|---|---|
| `config.py` | Every setting: radar (S_system), FFT scaling, detection, classifier thresholds, network |
| `dsp.py` | Preprocessing (identical to `golden_fft.py`), numpy range FFT, range-Doppler map |
| `pl_fft.py` | Range FFT in the FPGA through AXI DMA. Same output as the numpy version |
| `detect.py` | Strongest moving target, sub-bin range, speed, Doppler spread, single-target tracking |
| `classify.py` | Rule-based drone / person / unknown |
| `messages.py` | UDP JSON messages with sequence number, timestamp and CRC-32 checksum (FR11) |
| `sources.py` | Live radar (Infineon SDK) or replay of `record.py` recordings |
| `node.py` | The main loop, with CSV logging and status messages |
| `listener.py` | Test receiver. Counts malformed, lost and stale messages (FR12) |
| `sim.py`, `selftest.py` | Synthetic data and a self-test that needs no radar |
| `blade_det.py` | Blade channel: second detector for small, hovering drones (`BLADE` in `config.py`) |
| `blade_eval.py` | Scores the blade channel and the whole node on recordings, with `--set` to try settings |
| `diag.py`, `blade.py` | Quick looks at one recording around a given range (moving energy, blade energy, ADC headroom) |
| `clock_sync.py` | Run on the Mac: sets the PYNQ clock so stale checks work |

## Before you run it

1. In `config.py`, set `FFT_SHIFT` to the `total_shift` value in `tv/scaling.json`.
2. Set `dest_host` to the machine that runs the listener: your Mac's address on the PYNQ link (`192.168.2.1`), or later the Jetson.
3. Leave `RANGE_SCALE_MEASURED = 1.0` until S_system range calibration is done.

## Mac (replay)

Run these from this folder:

```
arch -x86_64 ~/radar-venv/bin/python selftest.py
arch -x86_64 ~/radar-venv/bin/python listener.py          # in a second Terminal window
arch -x86_64 ~/radar-venv/bin/python node.py --source replay ../characterization/recordings/<S_system file>.npz --log run.csv
```

## PYNQ (live)

Copy the folder to the board first: `scp -r radar_node xilinx@192.168.2.99:~/`. Then:

```
cd ~/radar_node
python3 selftest.py
python3 node.py --source live --dest 192.168.2.1:5005 --log run.csv
```

The listener runs on the Mac. If messages don't arrive, allow incoming connections for Python in the macOS firewall.

## Switching to the FPGA range FFT

Do this once `golden_fft.py compare` passes on the board:

```
python3 node.py --source live --fft pl --overlay ~/overlays/radar_fft.bit --dest 192.168.2.1:5005
```

The PL FFT passed the golden check on Nov 19 (48.9 to 67.6 dB error SNR). Add `--per-chirp` if the design doesn't have the TLAST counter yet.

## Messages

Every message has the fields `type`, `source`, `seq`, `timestamp`, `payload` and `checksum`.

- **`timestamp`** is the Unix time the frame was captured on the PYNQ.
- **`checksum`** is a CRC-32 of the message without its checksum field, serialized as compact JSON with sorted keys.

There are two message types.

**`radar_detection`**, one per frame (20 per second). Payload:

- `detected`, `range_m`, `speed_m_s` (approaching is negative, `null` when only the blade channel sees the target), `snr_db`
- `class` (drone, person, unknown or none), `confidence`
- `track_id` (1000 for blade-channel detections), `frame`, `latency_ms`
- `detector`: `main`, `blade` or `main+blade`
- `features`: `flash`, `flash_hz`, `kurtosis`, `doppler_spread_m_s`, `blade_state`, `blade_excess_db`, `blade_excess_hi_db`, `blade_hits`

**`radar_status`**, about once a second. Payload:

- `state`: initializing, running, disconnected, fault or stopped
- `fps`, `frames`, `sdk_errors`, `skipped_frames` (dropped because processing fell behind), `fft_backend`, `proc_ms_mean`, `proc_ms_max`, `detail`

Full field list: `../docs/radar_interface.md`.

## What is tested

**Self-test on synthetic data** (`selftest.py`, ALL PASS on the PYNQ, Nov 25):

- Preprocessing matches `golden_fft.py` (when it is next to `radar_node`).
- Person range within 0.01 m and speed within 0.01 m/s, classified as a person with no blade frames.
- Strong spinning blades at 2 m and 5 m classified as a drone.
- Weak drone (3.7 dB blade excess) missed by the main detector and found by the blade channel at 2.49 m.
- No drone labels on a walker, a walker leaving the scene, or 10 s of empty scene.
- Empty room gives no detections. Message checks (checksum, lost, stale, malformed) work.

**Live on the PYNQ with the PL FFT** (Nov 19 to 25):

- 20.00 fps for 5 minutes, 0 SDK errors, 1 skipped frame, 38.3 ms mean processing.
- 2-minute network run after `clock_sync.py`: 0 lost, 0 malformed, 0 stale.
- Sub-250 g drone hovering at 2 m: labelled drone in 82 to 97 % of frames. Not detected at 3 m.
- 60 s of walking in and out of the beam: 0 false drone labels.

**Limits:**

- Single target only. A drone at two or three times the range of a person within 1.5 m is hidden while the person is there.
- Small drones only to about 2.5 m. Range reads about 0.45 m long until calibrated.
- Stale checks need the PYNQ and receiver clocks synced: run `clock_sync.py` after each boot.

## Blade channel (small drones, added Nov 25)

A sub-249 g drone is too weak for the main detector (15 dB in one frame). Its blades still raise
the fast-Doppler energy at its range by about 3 dB over the empty scene when averaged over 0.5 s.
`blade_det.py` looks for that and reports `class: drone`, `detector: blade`, range, and
`speed_m_s: null` (a hovering body echo is buried in ground clutter). Settings: `BLADE` in `config.py`.

- The empty scene is learned during the first ~3.5 s of a run: keep the area in front clear.
  Or load one: `--blade-bg <empty .npz or .npy>`, save one with `--blade-save bg.npy`, turn off with `--no-blade`.
- Tune on recordings: `python3 blade_eval.py --bg <empty.npz> <drone and walk .npz files> --at 2.5`
  (`--set thr_db=1.0` etc. tries a setting without editing `config.py`).
- `diag.py` and `blade.py`: quick looks at one recording around a given range.
- Expected range: 2-3 m for a sub-249 g drone. Longer windows (`window_frames` 20-40) may stretch it.
