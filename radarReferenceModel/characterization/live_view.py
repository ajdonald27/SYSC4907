"""
RATS - live, easy-to-read range view.

It learns the empty room first, then shows ONLY what changed, and labels the strongest
new thing with its distance. Walls, desk and the radar's own leakage disappear.

    python live_view.py                    # preset A_range, radar connected
    python live_view.py --preset B_microdoppler
    python live_view.py --synthetic        # no radar: fake target moving 1 -> 4 m
    python live_view.py --chirps 128 --if-gain 40 --max-range 16   # more reach for 5-15 m (NFR1)

--chirps: more chirps per frame = more integration gain (32 -> 128 is ~+6 dB in "moving only").
--if-gain: receiver gain; raise it until the "ADC peak" readout nears saturation, then back off.
--avg: average the curve and readout over the last N frames (default 10 = 1 s at 10 fps).

Buttons under the plot (keyboard shortcut in brackets):
    True dist (m)       click, type the tape-measure distance (it stays until you change it)
    Save reading  [space]  append the averaged reading + true distance to recordings/live_readings.csv
    Moving only   [m]   hide anything standing still (people/fan/drone stay)
    Averaging     [a]   toggle averaging on/off
    Re-learn room [b]   step out of view first, then click
    Quit          [q]
"""
import argparse
import collections
import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from presets import PRESETS, to_sdk_config
from analyze import range_fft, db

C = 299_792_458.0
BG_FRAMES = 20


def frame_source(preset, p, synthetic):
    if synthetic:
        import synth
        t0 = time.time()
        while True:
            d = 1.0 + (time.time() - t0) * 0.3 % 3.0
            adc, _, _ = synth.synth("reflector", preset, d, 1.5 * p["frame_repetition_time_s"], 0, 0,
                                    seed=int(time.time() * 10) % 1000)
            time.sleep(p["frame_repetition_time_s"])
            yield adc[0].astype(np.float64)
    else:
        from ifxradarsdk.fmcw import DeviceFmcw
        with DeviceFmcw() as dev:
            dev.set_acquisition_sequence(dev.create_simple_sequence(to_sdk_config(p)))
            while True:
                yield np.asarray(dev.get_next_frame()[0])[0].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="A_range", choices=PRESETS.keys())
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--chirps", type=int, help="chirps per frame (preset default if omitted)")
    ap.add_argument("--if-gain", type=int, help="IF gain in dB (preset default if omitted)")
    ap.add_argument("--max-range", type=float, default=16.0, help="x-axis limit in m (default 16)")
    ap.add_argument("--avg", type=int, default=10, help="frames to average (default 10; 1 = off)")
    a = ap.parse_args()
    a.avg = max(1, a.avg)

    p = dict(PRESETS[a.preset])
    if a.chirps:
        p["num_chirps"] = a.chirps
    if a.if_gain is not None:
        p["if_gain_dB"] = a.if_gain
    frame_len = p["num_chirps"] * p["chirp_repetition_time_s"]
    if frame_len >= 0.9 * p["frame_repetition_time_s"]:
        raise SystemExit(f"{p['num_chirps']} chirps take {frame_len * 1e3:.0f} ms; frame period is "
                         f"{p['frame_repetition_time_s'] * 1e3:.0f} ms. Use fewer chirps.")
    if a.synthetic and a.chirps:
        print("note: --chirps is ignored with --synthetic")
    B = p["end_frequency_Hz"] - p["start_frequency_Hz"]
    rng = np.arange(p["num_samples"] // 2) * C / (2 * B)
    valid = rng >= 0.4                       # ignore the radar's own leakage
    lam = C / (0.5 * (p["start_frequency_Hz"] + p["end_frequency_Hz"]))
    print(f"preset {a.preset}: {p['num_chirps']} chirps, IF gain {p['if_gain_dB']} dB, "
          f"range bin {rng[1]:.2f} m, max range {rng[-1]:.1f} m")

    src = frame_source(a.preset, p, a.synthetic)
    state = {"bg": None, "learn": BG_FRAMES, "acc": [], "moving": False, "quit": False,
             "avg_on": a.avg > 1, "save": False, "reading": None}
    hist = collections.deque(maxlen=a.avg)   # per-frame (linear power profile, speed per bin)
    csv_path = os.path.join("recordings", "live_readings.csv")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_line, = ax.plot(rng, np.full_like(rng, -100), color="0.75", label="everything the radar sees")
    inst_line, = ax.plot(rng, np.full_like(rng, -100), color="C0", lw=0.8, alpha=0.3)  # this frame only
    new_line, = ax.plot(rng, np.full_like(rng, -100), color="C0", lw=2, label="new since background")
    marker = ax.axvline(0, color="r", ls="--")
    label = ax.text(0.5, 0.92, "", transform=ax.transAxes, ha="center", fontsize=16, color="r")
    ax.axvspan(5, 15, color="C2", alpha=0.08, label="NFR1 band 5-15 m")
    for d in (5, 10):
        ax.axvline(d, color="C2", ls=":", lw=1)
    adc_text = ax.text(0.01, 0.02, "", transform=ax.transAxes, fontsize=9, color="0.4")
    ax.set(xlabel="distance from radar (m)", ylabel="strength (dB)",
           xlim=(0, min(a.max_range, rng[-1])), ylim=(-120, -20))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ---- control panel: buttons + "true distance" box under the plot ----
    fig.set_size_inches(11, 6.5)
    fig.subplots_adjust(bottom=0.25)
    status = fig.text(0.5, 0.13, "Type the tape-measure distance in the box, then click 'Save reading'.",
                      ha="center", fontsize=10, color="0.25")
    ax_true = fig.add_axes([0.10, 0.03, 0.10, 0.06])
    tb_true = TextBox(ax_true, "True dist (m) ", initial="")
    b_save = Button(fig.add_axes([0.22, 0.03, 0.15, 0.06]), "Save reading", color="#cfe8cf", hovercolor="#a8d8a8")
    b_move = Button(fig.add_axes([0.39, 0.03, 0.17, 0.06]), "")
    b_avg = Button(fig.add_axes([0.58, 0.03, 0.15, 0.06]), "")
    b_bg = Button(fig.add_axes([0.75, 0.03, 0.12, 0.06]), "Re-learn room")
    b_quit = Button(fig.add_axes([0.89, 0.03, 0.07, 0.06]), "Quit", color="#f2c9c9", hovercolor="#e8a8a8")

    def refresh_buttons():
        b_move.label.set_text("Moving only: " + ("ON" if state["moving"] else "OFF"))
        b_avg.label.set_text(f"Averaging: {a.avg}" if state["avg_on"] else "Averaging: OFF")

    def do_save():
        state["save"] = True

    def do_move():
        state["moving"] = not state["moving"]
        hist.clear()                             # don't mix the two modes in one average
        refresh_buttons()

    def do_avg():
        state["avg_on"] = not state["avg_on"]
        refresh_buttons()

    def do_bg():
        state.update(bg=None, learn=BG_FRAMES, acc=[])
        hist.clear()

    def do_quit():
        state["quit"] = True

    b_save.on_clicked(lambda _: do_save())
    b_move.on_clicked(lambda _: do_move())
    b_avg.on_clicked(lambda _: do_avg())
    b_bg.on_clicked(lambda _: do_bg())
    b_quit.on_clicked(lambda _: do_quit())
    # (no on_submit save: TextBox also "submits" when you click away, which would double-save)
    refresh_buttons()

    def on_key(e):                               # keyboard shortcuts (ignored while typing in the box)
        if tb_true.capturekeystrokes:
            return
        actions = {"b": do_bg, "m": do_move, "a": do_avg, " ": do_save, "q": do_quit}
        if e.key in actions:
            actions[e.key]()
    fig.canvas.mpl_connect("key_press_event", on_key)

    for frame in src:
        if state["quit"] or not plt.fignum_exists(fig.number):
            break
        X = range_fft(frame[None])[0]                       # [chirps, bins] complex
        raw = db(np.sqrt((np.abs(X) ** 2).mean(axis=0)))
        raw_line.set_ydata(raw)
        adc_text.set_text(f"ADC peak {np.abs(frame).max():.3f}  |  {p['num_chirps']} chirps, "
                          f"IF gain {p['if_gain_dB']} dB")
        speed = None

        if state["learn"] > 0:                              # learning the empty room
            state["acc"].append(X.mean(axis=0))
            state["learn"] -= 1
            if state["learn"] == 0:
                state["bg"] = np.mean(state["acc"], axis=0)
            label.set_text(f"Learning empty room... stay out of view ({state['learn']})")
            new_line.set_ydata(np.full_like(rng, -120))
        else:
            if state["moving"]:
                # Remove everything static, then Doppler FFT across chirps: a moving target's
                # energy adds up coherently in one Doppler bin (integration gain grows with chirps).
                M = X.shape[0]
                Y = X - X.mean(axis=0, keepdims=True)
                D = np.fft.fftshift(np.fft.fft(Y * np.hanning(M)[:, None], axis=0), axes=0) / np.sqrt(M)
                P = np.abs(D) ** 2
                P[M // 2] = 0                               # zero-Doppler bin = static leftovers
                pw = P.max(axis=0)
                spd_bins = (P.argmax(axis=0) - M // 2) * lam / (2 * M * p["chirp_repetition_time_s"])
                mode = "moving only (Doppler)"
            else:
                Y = X - state["bg"][None, :]                # remove the empty room
                pw = (np.abs(Y) ** 2).mean(axis=0)
                spd_bins = None
                mode = "new since background"
            hist.append((pw, spd_bins))

            # Average in linear power (not dB), over the last N frames.
            n_used = len(hist) if state["avg_on"] else 1
            recent = list(hist)[-n_used:]
            new = 10 * np.log10(np.mean([h[0] for h in recent], axis=0) + 1e-24)
            inst_line.set_ydata(10 * np.log10(pw + 1e-24))
            inst_line.set_visible(n_used > 1)
            new_line.set_ydata(new)
            new_line.set_label(mode + (f", avg {n_used} frames" if n_used > 1 else ""))

            k = np.where(valid)[0][np.argmax(new[valid])]
            above = new[k] - np.median(new[valid])
            # frame-to-frame spread of the SNR at this range bin (how steady the reading is)
            per_frame = [10 * np.log10(h[0][k] / np.median(h[0][valid]) + 1e-24) for h in recent]
            spread = float(np.std(per_frame)) if n_used > 1 else 0.0
            # the bin's speed history includes noise frames, so use the median (robust to outliers)
            speed = float(np.median([h[1][k] for h in recent])) if state["moving"] else None
            marker.set_xdata([rng[k], rng[k]])

            pm = f" ± {spread:.0f}" if n_used > 1 else ""
            spd = f", {speed:+.2f} m/s" if speed is not None else ""
            if above >= 15:
                label.set_text(f"Target at {rng[k]:.2f} m{spd}   ({above:.0f}{pm} dB above noise)")
            else:
                label.set_text(f"Nothing clear yet (best {rng[k]:.2f} m, only {above:.0f}{pm} dB)")
            state["reading"] = dict(time=time.strftime("%H:%M:%S"), mode=mode, frames=n_used,
                                    chirps=p["num_chirps"], if_gain_dB=p["if_gain_dB"],
                                    range_m=round(float(rng[k]), 2), snr_dB=round(float(above), 1),
                                    snr_spread_dB=round(spread, 1),
                                    speed_m_s=None if speed is None else round(speed, 2),
                                    detected=bool(above >= 15))
            ax.legend(loc="upper right")

        if state["save"]:
            state["save"] = False
            r = state["reading"]
            if r is None:
                status.set_text("Nothing to save yet: still learning the empty room.")
            else:
                true_m = tb_true.text.strip()
                try:
                    true_m = round(float(true_m), 2) if true_m else ""
                except ValueError:
                    true_m = ""
                r = dict(r, true_m=true_m)
                state["n_saved"] = state.get("n_saved", 0) + 1
                err = f", error {r['range_m'] - true_m:+.2f} m" if true_m != "" else ""
                status.set_text(f"Saved #{state['n_saved']}: radar {r['range_m']:.2f} m, {r['snr_dB']:.0f} dB"
                                f" (true {true_m if true_m != '' else '?'} m{err})  -> {csv_path}")
                os.makedirs("recordings", exist_ok=True)
                new_file = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    if new_file:
                        w.writeheader()
                    w.writerow(r)
                print("saved:", ", ".join(f"{k}={v}" for k, v in r.items()))
        fig.canvas.draw_idle()
        plt.pause(0.001)


if __name__ == "__main__":
    main()
