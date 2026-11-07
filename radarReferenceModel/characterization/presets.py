"""
RATS radar characterization - chirp presets for the Infineon DEMO BGT60UTR11AIP.

The BGT60UTR11AIP has 1 TX / 1 RX and a real-valued 12-bit ADC, so a range FFT of
N samples gives N/2 usable range bins.

Key formulas (printed by `python presets.py`):
    range resolution   dR   = c / (2 * B)
    max range          Rmax = (N / 2) * dR             (Nyquist of the IF signal)
    max IF frequency   fIF  = 2 * Rmax * S / c = fs/2  (S = B / T_ramp, T_ramp = N / fs)
    wavelength         lam  = c / f_center (~5 mm at 60 GHz)
    max speed          vmax = lam / (4 * Tc)           (Tc = chirp repetition time)
    speed resolution   dv   = lam / (2 * M * Tc)       (M = chirps per frame)

If the SDK rejects a value (e.g. a filter cutoff it does not support), change it
here, re-run `python presets.py` to see the new derived numbers, and write the
change in the checklist.
"""

C = 299_792_458.0

PRESETS = {
    # A - range / calibration preset. Fine range resolution, long reach, slow chirps.
    #     Use for: reflector calibration, range resolution, noise floor, SNR vs range.
    "A_range": dict(
        start_frequency_Hz=60.0e9,
        end_frequency_Hz=61.0e9,      # B = 1 GHz -> dR = 15 cm
        sample_rate_Hz=1_000_000,     # 1 MS/s
        num_samples=256,              # T_ramp = 256 us -> Rmax = 128 * 0.15 = 19.2 m
        chirp_repetition_time_s=300e-6,
        num_chirps=32,
        frame_repetition_time_s=0.1,  # 10 frames/s
        tx_power_level=31,
        if_gain_dB=33,
        lp_cutoff_Hz=500_000,
        hp_cutoff_Hz=20_000,     # 2 m sits at ~52 kHz IF; 80 kHz HPF suppressed it
    ),
    # B - micro-Doppler preset. Short chirps (high vmax), long frames (fine Doppler),
    #     frame period close to the frame length (little dead time between frames).
    #     Use for: fan / propeller signatures, walking person, thrown ball, the
    #     classification dataset, and the PYNQ throughput benchmark.
    "B_microdoppler": dict(
        start_frequency_Hz=60.0e9,
        end_frequency_Hz=60.3e9,      # B = 300 MHz -> dR = 0.5 m (FR2 needs +/-1 m)
        sample_rate_Hz=1_000_000,
        num_samples=64,               # T_ramp = 64 us -> Rmax = 32 * 0.5 = 16 m
        chirp_repetition_time_s=80e-6,  # vmax ~ 15.6 m/s, 12.5 kHz slow-time rate
        num_chirps=512,               # frame = 41 ms -> dv ~ 0.06 m/s, ~24 Hz line spacing
        frame_repetition_time_s=0.05,  # 20 frames/s, ~82 % duty
        tx_power_level=31,
        if_gain_dB=33,
        lp_cutoff_Hz=500_000,
        hp_cutoff_Hz=20_000,     # keep close targets (fan at 1.5 m) above the HPF
    ),
}


def derived(p):
    B = p["end_frequency_Hz"] - p["start_frequency_Hz"]
    fc = 0.5 * (p["end_frequency_Hz"] + p["start_frequency_Hz"])
    N, fs, M, Tc = p["num_samples"], p["sample_rate_Hz"], p["num_chirps"], p["chirp_repetition_time_s"]
    T_ramp = N / fs
    S = B / T_ramp
    lam = C / fc
    dR = C / (2 * B)
    d = dict(
        bandwidth_MHz=B / 1e6,
        wavelength_mm=lam * 1e3,
        ramp_time_us=T_ramp * 1e6,
        slope_MHz_per_us=S / 1e12,
        range_resolution_m=dR,
        max_range_m=(N / 2) * dR,
        if_at_10m_kHz=2 * 10 * S / C / 1e3,
        max_speed_m_s=lam / (4 * Tc),
        speed_resolution_m_s=lam / (2 * M * Tc),
        slow_time_rate_kHz=1 / Tc / 1e3,
        frame_length_ms=M * Tc * 1e3,
        frame_rate_Hz=1 / p["frame_repetition_time_s"],
        duty_cycle_pct=100 * M * Tc / p["frame_repetition_time_s"],
        samples_per_frame=N * M,
        raw_rate_kS_s=N * M / p["frame_repetition_time_s"] / 1e3,
    )
    problems = []
    if Tc <= T_ramp:
        problems.append("chirp_repetition_time must be longer than the ramp time")
    if M * Tc >= p["frame_repetition_time_s"]:
        problems.append("frame_repetition_time must be longer than num_chirps * Tc")
    if 2 * d["max_range_m"] * S / C > p["lp_cutoff_Hz"] * 1.05:
        problems.append(
            f"IF at max range ({2 * d['max_range_m'] * S / C / 1e3:.0f} kHz) is above the "
            f"low-pass cutoff ({p['lp_cutoff_Hz'] / 1e3:.0f} kHz): far targets will be attenuated")
    if 2 * 2.0 * S / C < p["hp_cutoff_Hz"]:
        problems.append(
            f"a target at 2 m gives {2 * 2.0 * S / C / 1e3:.0f} kHz IF, below the high-pass cutoff "
            f"({p['hp_cutoff_Hz'] / 1e3:.0f} kHz): close targets will be suppressed")
    return d, problems


def to_sdk_config(p):
    """Build the Radar SDK sequence config (SDK 3.x Python API)."""
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
    return FmcwSimpleSequenceConfig(
        frame_repetition_time_s=p["frame_repetition_time_s"],
        chirp_repetition_time_s=p["chirp_repetition_time_s"],
        num_chirps=p["num_chirps"],
        tdm_mimo=False,
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=p["start_frequency_Hz"],
            end_frequency_Hz=p["end_frequency_Hz"],
            sample_rate_Hz=p["sample_rate_Hz"],
            num_samples=p["num_samples"],
            rx_mask=1,   # UTR11AIP: one receiver
            tx_mask=1,   # one transmitter
            tx_power_level=p["tx_power_level"],
            lp_cutoff_Hz=p["lp_cutoff_Hz"],
            hp_cutoff_Hz=p["hp_cutoff_Hz"],
            if_gain_dB=p["if_gain_dB"],
        ),
    )


if __name__ == "__main__":
    for name, p in PRESETS.items():
        d, problems = derived(p)
        print(f"\n=== {name} ===")
        for k, v in d.items():
            print(f"  {k:24s} {v:10.3f}")
        for msg in problems:
            print("  WARNING:", msg)
