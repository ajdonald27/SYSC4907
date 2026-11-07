# RATS radar scripts

- `sdk_check.py`: checks that the SDK imports and the radar answers.
- `presets.py`: chirp presets A (range) and B (micro-Doppler); run it to print derived specs.
- `record.py`: records frames to `recordings/*.npz` (`--benchmark` measures fps only).
- `analyze.py show <file>`: range, speed, micro-Doppler and blade-flash plots plus a summary.
- `analyze.py calib calib.csv`: range error vs tape distance (CSV: `file,true_m,background`).
- `live_view.py`: live distance view with the empty room removed (keys: b = relearn background, m = moving only, q = quit).
- `synth.py`: fake recordings for testing without the radar.

Setup: `pip install <ifxradarsdk wheel> numpy matplotlib` (armv7l wheel on the PYNQ).

```
python record.py --preset A_range --label reflector --distance 2 --seconds 5
python analyze.py show recordings/<file>.npz --background recordings/<empty>.npz
```
