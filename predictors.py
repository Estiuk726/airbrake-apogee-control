"""
Step 4a: apogee predictors (used during the coast, after burnout).

1. Energy only (no drag):      h + v^2 / (2g)
2. Analytic with drag:         h + (m / 2k) * ln(1 + k v^2 / (m g)),   k = 0.5 * rho * Cd * A
   (exact solution for constant k; we use rho and Cd at the current point)
3. Onboard forward simulation: step the equations of motion ahead until v = 0
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_v1 import AREA, DRY_MASS, BURN_TIME, G0, atmosphere, cd
from sim_v2 import simulate, A_BRAKE_MAX, CD_BRAKE


def pred_energy(h, v):
    return h + v**2 / (2 * G0)


def pred_analytic(h, v, u=0.0):
    rho, a_s = atmosphere(max(h, 0))
    k = 0.5 * rho * (cd(v / a_s) * AREA + CD_BRAKE * u * A_BRAKE_MAX)
    return h + DRY_MASS / (2 * k) * np.log(1 + k * v**2 / (DRY_MASS * G0))


def pred_forward(h, v, u=0.0, dt=0.05):
    while v > 0:
        rho, a_s = atmosphere(max(h, 0))
        k = 0.5 * rho * (cd(v / a_s) * AREA + CD_BRAKE * u * A_BRAKE_MAX)
        acc = -G0 - k * v * v / DRY_MASS
        # semi-implicit Euler, cheap enough for a flight computer
        v_new = v + acc * dt
        if v_new <= 0:                      # interpolate the last partial step
            h += v * v / (2 * (-acc)); break
        h += 0.5 * (v + v_new) * dt
        v = v_new
    return h


if __name__ == "__main__":
    truth = simulate(lambda t, h, v: 0.0)
    t, h, v = truth[:, 0], truth[:, 1], truth[:, 2]
    true_apogee = h.max()
    coast = (t > BURN_TIME) & (v > 0)
    tc = t[coast][::10]; hc = h[coast][::10]; vc = v[coast][::10]
    preds = {
        "Energy only": [pred_energy(a, b) for a, b in zip(hc, vc)],
        "Analytic with drag": [pred_analytic(a, b) for a, b in zip(hc, vc)],
        "Forward simulation": [pred_forward(a, b) for a, b in zip(hc, vc)],
    }
    print(f"True apogee: {true_apogee:.0f} m")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in preds.items():
        err = np.array(p) - true_apogee
        print(f"{name:20s} error at burnout {err[0]:+6.0f} m | at t=10 s "
              f"{err[np.argmin(abs(tc-10))]:+5.0f} m | at t=20 s {err[np.argmin(abs(tc-20))]:+4.0f} m")
        ax.plot(tc, err, label=name)
    ax.axhline(0, c="k", lw=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Time of prediction [s]"); ax.set_ylabel("Prediction error [m] (log scale)")
    ax.set_title("Apogee prediction error during the coast")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/step4a_predictors.png", dpi=130)
