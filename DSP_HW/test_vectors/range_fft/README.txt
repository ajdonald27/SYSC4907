Range-FFT test vectors (S_system), from 20250503_S_system_walk_pynq_nam_180851.npz

Each frame is 256 chirps x 128 samples, chirp-major (chirp 0 samples 0..127, then chirp 1 ...).
input_*.hex     one 32-bit DMA word per line: real int16 in [15:0], imag 0 in [31:16]
expected_*.hex  ideal FFT output words: real int16 in [15:0], imag int16 in [31:16]
*.npy           the same data for Python (input int16; expected complex float, before rounding)
input_tone      chirp m = cosine at bin (m % 60) + 2; chirp 255 = impulse.
                Expected peaks at bins (m % 60) + 2 and 128 - that; the impulse gives a flat spectrum.
window_q15      the 128-point Hann window table (Q1.15) applied before the FFT (on the ARM for now).

FFT IP: forward, 128-point, scaled, natural order. Total scaling = 2^-3
(largest unscaled output in the data = 43827; headroom after scaling = 15.5 dB).
Set SCALE_SCH so the per-stage shifts add up to 3, putting the shifts in the early stages first
(check the bit layout for Pipelined Streaming I/O in the FFT IP product guide, PG109).

Check the hardware output with:  python golden_fft.py compare expected_f0.npy <pl_output.npy>
(pl_output may be uint32 words or complex values, shape 256 x 128)
