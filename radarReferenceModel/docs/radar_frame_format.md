# Radar frame format (S_system)

The one radar configuration used for all live processing. The FPGA design, `golden_fft.py`, and
`radar_node/config.py` all assume exactly this frame. Frozen November 19, 2026.

## Radar configuration

| Setting | Value |
|---|---|
| Sensor | Infineon BGT60UTR11AIP (1 TX, 1 RX), USB kit on the PYNQ-Z2 |
| Sweep | 60.0 to 60.6 GHz (B = 600 MHz) |
| ADC | 1 MS/s, 128 samples per chirp (chirp sampled for 128 µs) |
| Chirp repetition time | 150 µs |
| Chirps per frame | 256 |
| Frame period | 50 ms (20 frames/s) |
| Filters | High-pass 20 kHz, low-pass 500 kHz |
| IF gain, TX power | 33 dB, level 31 (maximum) |

| Derived | Value |
|---|---|
| Range per bin | c / 2B = 0.25 m |
| Usable range bins | 64 (real input), 0 to 16 m |
| Minimum useful range | about 0.7 m (high-pass filter and TX leakage) |
| Unambiguous speed | ±λ / 4Tc = ±8.3 m/s (λ = 5.0 mm) |
| Speed per Doppler bin | λ / 2MTc = 0.065 m/s |

## 1. Frame from the SDK

`float [256 chirps][128 samples]`, receiver 1, values normalized by the SDK to about ±1.
Recordings (`record.py`) store frames as `float32 [n_frames, 256, 128]` in an `.npz` with
`t` (seconds from start) and `meta` (JSON: preset, label, distance, notes, frame rate, SDK errors).

## 2. Preprocessing (ARM, identical in `golden_fft.py` and `radar_node/dsp.py`)

For each chirp:

1. Subtract the mean of its 128 samples (DC removal).
2. Multiply by the Hann window table `round(hanning(128) * 32767) / 32767`.
3. Scale by 32767, round, and clip to int16 (Q1.15).

## 3. DMA input to the PL

- One 32-bit word per sample: bits 15:0 = real sample (int16), bits 31:16 = imaginary = 0.
- Chirp-major order: chirp 0 samples 0 to 127, then chirp 1, and so on.
- 256 x 128 = 32,768 words per frame, sent in one AXI DMA transfer (MM2S).

## 4. FFT core configuration

- Xilinx FFT LogiCORE, 128 points, fixed point, pipelined streaming, natural-order output.
- 16-bit input and phase factors, scaled arithmetic, convergent rounding.
- Configuration word written through AXI GPIO: `0x2B` = forward transform (bit 0) plus a
  scaling schedule of one bit in each of the first three stages (bits 8:1 = `0b00010101`).
- Total scaling 2^-3 (`FFT_SHIFT = 3` in `config.py`, `total_shift` in `tv/scaling.json`),
  chosen from real recordings with about 12 dB of headroom.

## 5. DMA output from the PL

- 32,768 words per frame (S2MM): bits 15:0 = real, bits 31:16 = imaginary, both int16.
- Natural order per chirp: equals FFT(x) / 8 of each preprocessed chirp.
- `axis_tlast_gen` asserts TLAST once per 32,768 words, so one transfer carries the whole frame.
- Only bins 0 to 63 of each chirp are used (real input).

Golden check (`golden_fft.py compare`): pass when the error SNR is at least 40 dB and the peak
bin agrees in at least 99 % of chirps. Result on the board (Nov 19): 48.9 to 67.6 dB, worst
error 3.2 LSB, scale x1.

## 6. After the range FFT (ARM)

1. Static scene removal: subtract the mean over the 256 chirps in each range bin.
2. Hann window across chirps, 256-point FFT per range bin, fftshift: range-Doppler map
   `[256 Doppler bins, 64 range bins]`.
3. Speed axis: `fftshift(fftfreq(256, 150e-6)) * λ / 2`. Approaching targets are negative.
4. Range axis: bin k = k x 0.25 m (`RANGE_SCALE_MEASURED = 1.0`, not calibrated yet).

Timing on the PYNQ (Nov 19): range FFT in the PL 3.9 ms per frame (pack 0.4, DMA and FFT 1.8,
unpack 1.1), range-Doppler 8.2 ms, whole chain 34.7 ms mean.
