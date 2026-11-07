"""
RATS - 30-second "is the radar alive on this machine?" check.
Run on the Mac, in Windows (Parallels) or on the PYNQ-Z2:   python sdk_check.py
"""
import platform
import sys

print("Python  :", sys.version.split()[0], "|", platform.machine(), "|", platform.platform())
try:
    import numpy as np
    from ifxradarsdk import get_version_full
    from ifxradarsdk.fmcw import DeviceFmcw
    print("SDK     :", get_version_full())
except Exception as e:
    sys.exit(f"SDK import FAILED: {type(e).__name__}: {e}\n"
             "-> wrong wheel for this platform/Python, or a missing shared library. See checklist step 1.")

try:
    with DeviceFmcw() as dev:
        print("Sensor  :", dev.get_sensor_type())
        frame = np.asarray(dev.get_next_frame()[0])   # default config
        print("Frame   :", frame.shape, "(rx, chirps, samples)  min/max", float(frame.min()), float(frame.max()))
        print("OK - radar is talking to this machine.")
except Exception as e:
    sys.exit(f"Device FAILED: {type(e).__name__}: {e}\n"
             "-> cable/port, USB passthrough (Parallels), or permissions (Linux: try sudo).")
