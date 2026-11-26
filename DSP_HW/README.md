# RATS FPGA: range FFT overlay for the PYNQ-Z2

| File | What it is |
|---|---|
| `build_radar_fft.tcl` | Builds the whole Vivado project and block design, then makes the bitstream |
| `axis_tlast_gen.v` | Small Verilog block: TLAST every 32768 words, so the DMA gets one whole frame per transfer |
| `test_fft_overlay.py` | Runs on the PYNQ: loopback test, or FFT test against the golden test vectors |
| `overlays/` | Built bitstreams and hardware handoff files (`radar_fft.*`, `radar_loopback.*`), ready to copy to the PYNQ |
| `test_vectors/range_fft/` | Golden test vectors from `golden_fft.py` (inputs, expected outputs, `scaling.json`) |

Frame format and word layout: `../radarReferenceModel/docs/radar_frame_format.md`.

## Results on the board (November 19, 2026)

| Test | Result |
|---|---|
| Loopback, one 32,768-word frame | PASS, 1.29 ms round trip |
| FFT vs golden, tone and three recorded frames | PASS, error SNR 48.9 to 67.6 dB, worst 3.2 LSB, peak bins agree 99.6 to 100 %, scale x1 |
| Range FFT step in the live node | 3.9 ms per frame (DMA and FFT 1.8 ms) vs 6.4 ms in NumPy |
| Build time | About 25 minutes per design (Vivado 2024.1 in Parallels) |

**Design:**

- Zynq PS at 100 MHz.
- AXI DMA in simple mode with 32-bit streams.
- Xilinx FFT: 128-point, scaled, pipelined, natural order, non-real-time.
- `axis_tlast_gen` between the FFT and the DMA.
- AXI GPIO that sets the FFT's forward/scaling word from Python.

The Tcl script picks the newest installed version of each IP, so it runs on 2024.1 or a newer Vivado. Use 2024.1 to match the PYNQ v3.1.1 image.

## 1. Build the bitstreams (Windows / Parallels)

1. Copy this folder to `C:\radar\fpga`. Use a short path with no spaces; Vivado dislikes long paths.
2. Open Vivado. At the bottom, open the **Tcl Console**.
3. Build the loopback design first. This is the bring-up test:

   ```
   cd C:/radar/fpga
   set MODE loopback
   source build_radar_fft.tcl
   ```

4. When it prints `RATS: done in … min`, you have `out/radar_loopback.bit` and `out/radar_loopback.hwh`. Note the build time.
5. Build the FFT design:

   ```
   set MODE fft
   source build_radar_fft.tcl
   ```

   You get `out/radar_fft.bit` and `out/radar_fft.hwh`.

You can also skip the GUI. In the "Vivado 2024.1 Tcl Shell", run this (it builds the FFT design):

```
cd C:/radar/fpga
vivado -mode batch -source build_radar_fft.tcl
```

**If it stops with an error,** copy the first red ERROR line and send it. The likely spots are an IP setting name that changed between Vivado versions, or the block automation step.

## 2. Copy the files to the PYNQ (Mac)

Get `out/` from Windows to the Mac through a Parallels shared folder. Then:

```
ssh xilinx@192.168.2.99 "mkdir -p ~/overlays"
scp out/radar_loopback.bit out/radar_loopback.hwh out/radar_fft.bit out/radar_fft.hwh test_fft_overlay.py xilinx@192.168.2.99:~/overlays/
scp -r /Users/ajdonald/Desktop/SYSC4907/SYSC4907/radarReferenceModel/characterization/tv xilinx@192.168.2.99:~/
```

The `.bit` and `.hwh` for each design must have the same base name and sit in the same folder.

## 3. Test on the PYNQ (Jupyter terminal)

```
cd /home/xilinx/overlays
python3 test_fft_overlay.py --bit /home/xilinx/overlays/radar_loopback.bit
python3 test_fft_overlay.py --bit /home/xilinx/overlays/radar_fft.bit --tv /home/xilinx/tv
```

- **Loopback:** expect `LOOPBACK PASS` and a round trip of a few milliseconds.
- **FFT:** expect `PASS` for the `tone` frame and each `f0`, `f1`, `f2` frame. That means error SNR of 40 dB or more and matching peak bins.

The script prints hints when a test fails:

- **Peaks are right but the size is off by about 2^k:** the scaling bits are read differently than the model assumes. Rerun with the `--shift` value it suggests, then put that value in `radar_node/config.py` as `FFT_SHIFT`.
- **Most peaks differ:** the word layout, output order or FFT direction is wrong. Send the output.

## 4. Use it in the radar node

Once the FFT test passes:

```
cd /home/xilinx/radar_node
python3 node.py --source live --fft pl --overlay /home/xilinx/overlays/radar_fft.bit --dest 192.168.2.1:5005 --seconds 60
```

Measured: 34.7 ms mean processing per frame with the PL FFT, against 37.4 ms with NumPy.
