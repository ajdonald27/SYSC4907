"""
RATS radar node - all settings in one place.

RADAR must stay identical to S_system in characterization/presets.py and to
docs/radar_frame_format.md (the FPGA expects exactly this frame).
"""
import numpy as np

# ---- radar (frozen system configuration S_system, 2026-11-19) ----
RADAR = dict(
    start_frequency_Hz=60.0e9, end_frequency_Hz=60.6e9, sample_rate_Hz=1_000_000,
    num_samples=128, chirp_repetition_time_s=150e-6, num_chirps=256,
    frame_repetition_time_s=0.05, tx_power_level=31, if_gain_dB=33,
    lp_cutoff_Hz=500_000, hp_cutoff_Hz=20_000,
)

C = 299_792_458.0
N = RADAR["num_samples"]            # samples per chirp = range FFT size (128)
M = RADAR["num_chirps"]             # chirps per frame = Doppler FFT size (256)
NB = N // 2                         # useful range bins (real input)
TC = RADAR["chirp_repetition_time_s"]
BW = RADAR["end_frequency_Hz"] - RADAR["start_frequency_Hz"]
FC = 0.5 * (RADAR["end_frequency_Hz"] + RADAR["start_frequency_Hz"])
LAMBDA = C / FC
RANGE_RES = C / (2 * BW)            # 0.25 m

# Range calibration: measured = RANGE_SCALE_MEASURED x true. Preset A read about 1.056;
# re-measure with S_system (calibration task) and set it here. 1.0 = no correction.
RANGE_SCALE_MEASURED = 1.0
RANGE_AXIS = np.arange(NB) * RANGE_RES / RANGE_SCALE_MEASURED
VEL_AXIS = np.fft.fftshift(np.fft.fftfreq(M, TC)) * LAMBDA / 2   # m/s; approaching = negative

# ---- range FFT ----
# Total scaling of the range FFT (2**-FFT_SHIFT), the same value as golden_fft.py's
# tv/scaling.json and the FFT IP's scaling schedule. Copy your number here.
FFT_SHIFT = 3

# ---- detection and tracking ----
DETECT = dict(
    min_range_m=0.7,       # below this the 20 kHz high-pass filter and TX leakage dominate
    max_range_m=15.5,
    snr_db=15.0,           # moving-energy peak above the median; 12 let a 12-14 dB clutter blob through (Nov 19 live test)
    gate_m=1.5,            # keep following the current track within +/- this distance
    max_misses=10,         # frames without a detection before the track is dropped (0.5 s)
)

# ---- classifier (rule-based v1; thresholds are first guesses until the drone session) ----
CLASSIFY = dict(
    history=10,                  # frames considered (0.5 s)
    blade_spread_m_s=6.0,        # blades: Doppler spread at least this wide (limbs stay under ~4 m/s)
    blade_kurtosis=4.0,          # ...and a bursty (flashing) fast-Doppler envelope
    flash_vmin_m_s=1.0,          # envelope uses only Doppler faster than this
    flash_band_hz=(60.0, 1000.0),  # flash-rate estimate range (reported, not used to decide)
    drone_min_blade_frames=3,    # blade frames in at least this many of the recent frames -> drone
    person_spread_m_s=(0.6, 5.0),  # walking person: real walk (Nov 18) 90th pct 4.1, 99th 4.7, max 5.2 m/s
    person_max_speed_m_s=3.0,
    person_min_speed_m_s=0.2,    # median |speed| over the history: something that barely moves is not a walker
    min_frames=3,                # frames with a detection before any class is given
)

# ---- blade channel: second detector for small, hovering drones (Nov 25 outdoor test) ----
# A sub-249 g drone is ~3 dB above the empty-scene blade-band energy at 2.5 m when averaged over
# 0.5 s, far below DETECT['snr_db']. Blade echoes alias over the whole +/-8.3 m/s span, so the
# excess shows up above 5 m/s too; a walker's limbs stay under ~5 m/s (that's the "tilt" test).
# First guesses from one recording: tune with blade_eval.py.
BLADE = dict(
    enabled=True,
    min_range_m=1.0,           # ground and TX leakage dominate closer in
    max_range_m=6.0,
    v_lo_m_s=1.5,              # band used: |v| >= 1.5 m/s (wind / slow clutter live below)
    v_hi_m_s=5.0,              # "high" band |v| >= 5 m/s (above a walker's limbs)
    window_frames=10,          # energy averaged over this many frames (0.5 s); 20-40 for more range
    learn_frames=60,           # empty-scene profile learned over this many frames (3 s) at start
    thr_db=1.0,                # full-band excess over the empty scene (Nov 25: drone at 2.5 m +1.1 to +2.9 dB)
    thr_hi_db=0.0,             # high band: real blades (Nov 25) only +0.5 dB there; 0.5 cut hits in half
    max_tilt_db=3.0,           # full minus high-band excess: drone 0.3-2.4 dB (Nov 25), walker 4-14 dB (5-min run)
    persist_frames=20,         # look back this many frames (1 s) ...
    persist_min=12,            # ... a drone needs a blade hit in at least this many,
    persist_gate_m=0.5,        # all within +/- this distance of their median range
    mask_gate_m=1.0,           # ignore blade hits within this distance of the main tracker's target...
    mask_hold_frames=30,       # ...and for 1.5 s after it is lost (a walker leaving left 'drone' streaks)
    mask_multipath_below_m=1.5,  # a target this close also echoes at 2x and 3x its range: mask those too
                                 # (walker at 0.9 m -> 'drone' at 2.9 m, Nov 25 v3 walk test)
    bg_alpha=0.005,            # slow empty-scene update when nothing is there (~10 s time constant)
    person_override_hi_db=2.0,  # blade channel beats a "person" label only with this much high-band excess
)

# ---- network ----
NET = dict(
    dest_host="127.0.0.1",   # Jetson / Mac running the listener
    dest_port=5005,
    source="radar-pynq",
    status_period_s=1.0,
)
