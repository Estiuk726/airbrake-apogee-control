"""
Step 3b: Kalman filter fusing accelerometer + barometer.

State x = [altitude h, velocity v, accelerometer bias b]
PREDICT (every step): use the accelerometer to move the state forward
    v_new = v + (a_meas - b) * dt
    h_new = h + v * dt + 0.5 * (a_meas - b) * dt^2
    b_new = b
UPDATE (when we trust the baro): nudge the state toward the barometer,
    by an amount the filter computes from how uncertain each side is (Kalman gain K).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_v1 import atmosphere
from sim_v2 import simulate
from sensors import make_sensors, BARO_NOISE, ACC_NOISE


class AltitudeKF:
    def __init__(self, dt, gating=False, mach_lockout=False):
        self.dt, self.gating, self.mach_lockout = dt, gating, mach_lockout
        self.x = np.zeros(3)
        self.P = np.diag([1.0, 0.1, 0.5 ** 2])           # initial uncertainty
        self.F = np.array([[1, dt, -0.5 * dt**2],
                           [0, 1, -dt],
                           [0, 0, 1]])
        self.B = np.array([0.5 * dt**2, dt, 0])
        q = (ACC_NOISE * 1.5) ** 2                          # trust accel slightly less than its spec
        self.Q = q * np.outer(self.B, self.B) + np.diag([0, 0, 1e-7])
        self.H = np.array([[1.0, 0, 0]])
        self.R = BARO_NOISE ** 2
        self.rejected = 0

    def step(self, acc_meas, baro):
        # Predict
        self.x = self.F @ self.x + self.B * acc_meas
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Decide whether to use the barometer
        if self.mach_lockout:
            mach_est = abs(self.x[1]) / atmosphere(max(self.x[0], 0))[1]
            if mach_est > 0.75:
                self.rejected += 1
                return self.x.copy()
        y = baro - self.H @ self.x                          # innovation: baro vs prediction
        S = self.H @ self.P @ self.H.T + self.R
        if self.gating and y**2 / S > 9:                    # more than 3 sigma away
            self.rejected += 1
            return self.x.copy()
        # Update
        K = self.P @ self.H.T / S
        self.x = self.x + (K * y).ravel()
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x.copy()


def run(truth, baro, acc, **kw):
    dt = truth[1, 0] - truth[0, 0]
    kf = AltitudeKF(dt, **kw)
    est = np.array([kf.step(a, b) for a, b in zip(acc, baro)])
    return est, kf.rejected


if __name__ == "__main__":
    truth = simulate(lambda t, h, v: 0.0)
    t, h, v = truth[:, 0], truth[:, 1], truth[:, 2]
    baro, acc, _, _ = make_sensors(truth)

    variants = {
        "Plain Kalman": dict(),
        "Kalman + 3-sigma gating": dict(gating=True),
        "Kalman + Mach lockout": dict(mach_lockout=True),
    }
    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for name, kw in variants.items():
        est, rej = run(truth, baro, acc, **kw)
        eh, ev = est[:, 0] - h, est[:, 1] - v
        print(f"{name:26s} max alt err {np.max(abs(eh)):5.1f} m | "
              f"max vel err {np.max(abs(ev)):5.1f} m/s | "
              f"bias est {est[-1,2]:.3f} (true 0.150) | baro rejected {rej}")
        ax[0].plot(t, eh, label=name); ax[1].plot(t, ev, label=name)
        ax[2].plot(t, est[:, 2], label=name)
    ax[0].set_ylabel("Altitude error [m]"); ax[0].set_title("Filter altitude error")
    ax[1].set_ylabel("Velocity error [m/s]"); ax[1].set_title("Filter velocity error")
    ax[2].axhline(0.15, c="k", ls="--", lw=1, label="True bias")
    ax[2].set_ylabel("Bias [m/s²]"); ax[2].set_title("Filter learns the accelerometer bias")
    ax[2].set_xlabel("Time [s]")
    for a in ax: a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/step3b_kalman.png", dpi=130)
