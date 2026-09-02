# Streaming FPGA Architecture

- Initial Draft: 07/21/2026
- Latest Revision: 07/21/2026
- Authors: AJ Donald & Jayven Larsen
- Status: Initial Draft

---

# 1. Purpose

The purpose of this document is to define the proposed streaming FPGA architecture for the radar signal-processing subsystem.

The architecture converts digitized radar samples into a list of detected targets containing Doppler frequency, signal magnitude, and estimated radial velocity.

This is a preliminary design and will continue to evolve as the radar module, ADC, FPGA board, fixed-point formats, and system interfaces are selected and refined.

---

# 2. Current Processing Chain

The current floating-point Python Golden Reference Model (GRM) performs the following operations:

1. Radar sample acquisition and/or CSV loading
2. DC offset removal
3. Digital FIR low-pass filtering
4. Hamming-window multiplication
5. Fast Fourier Transform (FFT)
6. Magnitude calculation
7. Peak detection
8. Quadratic peak interpolation
9. Radial velocity estimation

---

# 3. Proposed FPGA Data Flow

## 3.1 Signal Acquisition Chain

```
Radar Module
    ↓
Analog Front End
    ↓
ADC
    ↓
ADC Interface
```

## 3.2 FPGA Processing Chain

```
ADC Interface
    ↓
Input Sample Formatter
    ↓
DC Offset Removal
    ↓
FIR Low-Pass Filter
    ↓
Frame Controller
    ↓
Hamming Window
    ↓
FFT
    ↓
Magnitude Calculation
    ↓
Peak Detection
    ↓
Frequency Refinement
    ↓
Velocity Estimation
    ↓
Detection Packet Generator
    ↓
ARM Processor
```

---

# 4. Streaming Architecture Overview

The FPGA processing subsystem will use a streaming architecture wherever possible.

Radar samples will enter the FPGA sequentially and pass through each processing block in order. Each block performs a specific operation before forwarding the result to the next stage.

The internal processing pipeline is expected to use an AXI4-Stream style interface consisting of:

- `tdata` – Sample or processing result
- `tvalid` – Indicates valid data
- `tready` – Indicates the downstream block is ready to receive data
- `tlast` – Indicates the final sample of a processing frame

Using a streaming architecture allows multiple processing blocks to operate simultaneously, increasing throughput while minimizing memory usage.

---

# 5. Preliminary FPGA / ARM Processing Split

The initial design proposes that the FPGA perform the computationally intensive DSP operations, while the ARM processor is responsible for system control and communication.

### FPGA

- Sample formatting
- DC offset removal
- FIR filtering
- Hamming window
- FFT
- Magnitude calculation
- Peak detection
- Detection packet generation

### ARM Processor

- FPGA configuration
- Radar system control
- Reading detection packets
- Velocity calculation (initial implementation)
- Communication with the sensor fusion subsystem
- Debugging and visualization

---

# 6. Open Design Decisions

The following items remain to be finalized:

- Radar module selection (HB100)
- ADC selection (Look into : ADS7066IRTER)
- FPGA platform configuration (zybo 20)
- FFT size (probably 1024, if FPGA is capable, do bigger. 4096)
- Fixed-point data widths (need to research)
- Communication interfaces (SPI, UART and AXI interface)
- Final FPGA / ARM task partition (Distinguish tasks between ARM)
