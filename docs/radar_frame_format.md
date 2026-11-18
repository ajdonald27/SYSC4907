# Radar frame format for the FPGA (S_system, v1, 2026-11-19)

## Radar
- Sweep: 60.0-60.6 GHz.
- Sampling: 128 samples per chirp at 1 MS/s.
- Chirps: 150 us chirp period, 256 chirps per frame.
- Frame rate: 20 fps (a 38.4 ms burst every 50 ms).
- Channels: 1 RX, real-valued samples.
- Range: bin k = k x 0.25 m; bins 0-63 are useful, max about 16 m.
- Speed: max 8.3 m/s, resolution 0.065 m/s. Approaching reads negative.

## ARM side (Python, before DMA)
1. Take one frame from the SDK: float32 array of shape (256, 128).
2. Remove DC: subtract each chirp's mean.
3. Window: multiply by a 128-point Hann window.
4. Convert to int16 Q1.15: round(x * 32767), clipped.

## ARM to PL (AXI DMA MM2S to the FFT IP)
- One DMA transfer per frame: 256 x 128 = 32768 words of 32 bits (128 KB).
- Word layout: [15:0] = sample (int16), [31:16] = 0 (imaginary part).
- TLAST on the last word of the frame.

## FFT IP (Xilinx FFT, forward)
- 128-point, fixed size, pipelined streaming.
- 16-bit fixed-point input, scaled.
- Natural-order output.
- Scaling schedule: set with the golden model on Nov 20.

## PL to ARM (S2MM)
- 32768 complex words per frame: [15:0] = real, [31:16] = imaginary (int16 each).
- The ARM keeps bins 0-63 of each chirp, then runs the Doppler FFT (256-point, numpy) and detection.

## Throughput
- About 655k samples/s, or about 2.6 MB/s each way. Small for AXI DMA at 100 MHz.
