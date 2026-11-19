"""
Step 3a: simulate the sensors a real flight computer would see.

- Barometer: altitude + noise + transonic error (shock waves over the static port)
- Accelerometer: true acceleration + noise + small constant bias
  (it cannot feel gravity in free fall, so it measures a + g; we subtract g in software)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_v1 import G0, atmosphere
from sim_v2 import simulate

RNG = np.random.default_rng(42)
BARO_NOISE = 2.0      # m, 1-sigma
ACC_NOISE = 0.5       # m/s^2, 1-sigma
ACC_BIAS = 0.15       # m/s^2, calibration error


def transonic_error(mach):
    # Illustrative: pressure disturbance peaks around Mach 0.9 and reads as fake altitude
    return -120 * np.exp(-((mach - 0.9) / 0.06) ** 2)


def make_sensors(truth):
    t, h, v = truth[:, 0], truth[:, 1], truth[:, 2]
    a = np.gradient(v, t)
    mach = np.array([abs(vi) / atmosphere(max(hi, 0))[1] for hi, vi in zip(h, v)])
    baro = h + transonic_error(mach) + RNG.normal(0, BARO_NOISE, len(t))
    acc = a + ACC_BIAS + RNG.normal(0, ACC_NOISE, len(t))
    return baro, acc, a, mach


if __name__ == "__main__":
    truth = simulate(lambda t, h, v: 0.0)
    t, h, v = truth[:, 0], truth[:, 1], truth[:, 2]
    baro, acc, a_true, mach = make_sensors(truth)
    dt = t[1] - t[0]

    # Naive method 1: velocity = change in baro altitude / time
    v_from_baro = np.gradient(baro, dt)
    # Naive method 2: integrate the accelerometer
    v_from_acc = np.cumsum(acc) * dt
    h_from_acc = np.cumsum(v_from_acc) * dt

    print(f"Baro-derived velocity error (std): {np.std(v_from_baro - v):.0f} m/s")
    print(f"Accel-integrated altitude error at apogee: {h_from_acc[-1] - h[-1]:.0f} m")
    print(f"Worst transonic baro error: {np.min(baro - h):.0f} m")

    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    ax[0].plot(t, baro, lw=0.6, label="Barometer reading")
    ax[0].plot(t, h, "k", lw=1.2, label="True altitude")
    ax[0].set_ylabel("Altitude [m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Barometer: noisy, with a fake dip near Mach 0.9")
    ins = ax[0].inset_axes([0.55, 0.12, 0.42, 0.38])
    m = (t > 2.5) & (t < 5.5)
    ins.plot(t[m], baro[m] - h[m], lw=0.6); ins.set_title("Baro error, t=2.5-5.5 s", fontsize=7)
    ins.tick_params(labelsize=6)

    ax[1].plot(t, v_from_baro, lw=0.4, label="From differentiating baro")
    ax[1].plot(t, v_from_acc, lw=1.2, label="From integrating accelerometer")
    ax[1].plot(t, v, "k", lw=1.2, label="True velocity")
    ax[1].set_ylim(-300, 600); ax[1].set_ylabel("Velocity [m/s]"); ax[1].legend(fontsize=8)
    ax[1].set_title("Velocity: baro is far too noisy, accelerometer drifts")

    ax[2].plot(t, h_from_acc - h, label="Accelerometer altitude error")
    ax[2].set_ylabel("Error [m]"); ax[2].set_xlabel("Time [s]"); ax[2].legend(fontsize=8)
    ax[2].set_title("Accelerometer drift grows with time")
    for x in ax: x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step3a_sensors.png", dpi=130)
