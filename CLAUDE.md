# RATS (SYSC4907 capstone) - notes for Claude

Project: Radar-Assisted Vision Tracking System. Detect and classify small drones at 5-15 m indoors.
The radar processing unit is a PYNQ-Z2. The vision/fusion unit is a Jetson.
AJ and Jayven own the radar subsystem.

## Radar
- Infineon DEMO BGT60UTR11AIP: 60 GHz FMCW, 1 TX / 1 RX (no angle), real-valued 12-bit ADC.
- Radar SDK 3.6.4 (`ifxradarsdk`), firmware 2.6.0. Wheels are in ~/radar_sdk/python_wheels:
  macOS universal2, linux_armv7l (for the PYNQ-Z2), win_amd64.
- SDK frames are signed floats (about +/-0.14) with shape (rx, chirps, samples).
  Recordings are saved as float32 .npz.
- The radar connects to the PYNQ-Z2 over USB. The Jetson CANNOT host it: there is no aarch64 SDK build.

## Running Python on this Mac
- ALWAYS run `arch -x86_64 ~/radar-venv/bin/python ...` (an x86_64 venv under Rosetta).
- Plain `python` is aliased to the system 3.9: do not use it.
- Shell shortcuts in ~/.zshrc: `rec` (record.py), `ana` (analyze.py), `live` (live_view.py).
  `$RATS` points to radarReferenceModel/characterization.
- Only one program can use the radar at a time. The Radar Fusion GUI runs in Parallels Windows.

## Scripts (radarReferenceModel/characterization)
- presets.py: A_range and B_microdoppler. HP cutoff is 20 kHz, because 80 kHz hid targets
  closer than about 3 m.
- sdk_check.py, record.py (`--show`), analyze.py (`show` / `calib`; auto background = newest empty_*),
  live_view.py (live range view, see below), synth.py (synthetic data).
- analyze.py sets the Agg backend only when run directly; forcing it on import broke live_view's window.
- recordings/, *.png and *.npz are gitignored.

## live_view.py
- Must be run by the user in their own Terminal. A Claude background shell has no display.
- Recommended for 5-15 m: `live --chirps 256 --if-gain 40 --max-range 16`.
  256 chirps is the most that fits in the 100 ms frame (about +9 dB). Watch the "ADC peak" readout for saturation.
- Two modes: background subtraction, and "Moving only" (Doppler FFT, which also reports speed). Both average N frames (`--avg`, default 10).
- Buttons: True dist box, Save reading, Moving only, Averaging, Re-learn room, Quit.
  Each save appends a row to recordings/live_readings.csv (true_m, range_m, snr_dB, snr_spread_dB, speed_m_s, ...).
- The Doppler speed sign is not verified on hardware yet: approaching should read negative.

## Proposal targets (SYSC4907_RATS_Proposal_Final.pdf)
- NFR1: detect at >= 5 m, with about 10 m as the target and about 15 m as the maximum.
- FR2: range within +/-1 m and speed within +/-2 m/s.
- NFR2: drone vs. walking person correct in >= 75 % of trials. NFR3: report within 1 s.
- Decision gate: drone at 2/5/10 m. If the propeller signature shows at 5 m, use micro-Doppler;
  otherwise use motion/trajectory features.

## Status (2026-11-07)
- Python talks to the radar on the Mac, and live_view works.
- The radar tracked a hand-held baking sheet from 1 to 4.5 m. That target is much stronger than a drone,
  and the distance is below NFR1's 5 m, so this does not yet show 5-15 m performance.
- Next: a tape-measured distance test with live_view logging. Use a person in Moving only mode at 2/5/8/10 m,
  then a can on a stick at 5/10 m. Compare the baseline, --chirps 256, and --if-gain 40.
  After that: T1-T7, then the drone decision-gate session.
- Nov 20: golden model plus live PYNQ acquisition. Dec 4: range FFT in the PYNQ PL via DMA.

## Preferences
- Keep instructions concise.
- Generated documents use Times New Roman: headings 12 pt bold, body 11 pt.
