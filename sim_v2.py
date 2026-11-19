"""
Step 2: add an airbrake.

The airbrake adds extra drag:  D_brake = 0.5 * rho * v^2 * CD_BRAKE * (u * A_BRAKE_MAX)
where u = deployment fraction, 0 (closed) to 1 (fully open).
It is only allowed after burnout (opening flaps under thrust is risky).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_v1 import AREA, BURN_TIME, G0, thrust, mass, atmosphere, cd

A_BRAKE_MAX = 0.008   # m^2, total flap area when fully open (placeholder: 3 flaps ~ 5x5.3 cm)
CD_BRAKE = 1.2        # flat plate facing the flow


def simulate(deploy_rule, dt=0.01):
    """deploy_rule(t, h, v) -> u in [0, 1]"""
    t, h, v = 0.0, 0.0, 0.0
    log = []

    def acc(t, h, v):
        rho, a_s = atmosphere(max(h, 0))
        u = deploy_rule(t, h, v) if t > BURN_TIME else 0.0
        drag = 0.5 * rho * v * abs(v) * (cd(abs(v) / a_s) * AREA + CD_BRAKE * u * A_BRAKE_MAX)
        return (thrust(t) - drag) / mass(t) - G0, u

    while True:
        # RK4
        a1, u = acc(t, h, v)
        k1 = (v, a1)
        k2 = (v + dt/2*k1[1], acc(t+dt/2, h+dt/2*k1[0], v+dt/2*k1[1])[0])
        k3 = (v + dt/2*k2[1], acc(t+dt/2, h+dt/2*k2[0], v+dt/2*k2[1])[0])
        k4 = (v + dt*k3[1],   acc(t+dt,   h+dt*k3[0],   v+dt*k3[1])[0])
        log.append((t, h, v, u))
        h += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        t += dt
        if v < 0 and t > BURN_TIME:
            break
    return np.array(log)


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))

    # Experiment A: constant deployment level from burnout to apogee
    print("A) Constant deployment after burnout")
    levels = [0, 0.25, 0.5, 0.75, 1.0]
    apogees = []
    for u0 in levels:
        d = simulate(lambda t, h, v, u0=u0: u0)
        apogees.append(d[:, 1].max())
        print(f"   {int(u0*100):3d}% open -> apogee {apogees[-1]:.0f} m")
        ax[0].plot(d[:, 0], d[:, 1], label=f"{int(u0*100)}% open: {apogees[-1]:.0f} m")
    ax[0].axhline(3000, c="k", ls="--", lw=1); ax[0].text(0.5, 3030, "Target 3000 m")
    ax[0].set_title("A) Airbrake held open from burnout to apogee")
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("Altitude [m]"); ax[0].legend(fontsize=8)

    # Experiment B: fully open for only 3 seconds, starting at different times
    print("B) Fully open for 3 s, starting at different times")
    base = apogees[0]
    starts = np.arange(4.5, 24, 1.0)
    losses = []
    for s in starts:
        d = simulate(lambda t, h, v, s=s: 1.0 if s <= t < s + 3 else 0.0)
        losses.append(base - d[:, 1].max())
    for s, l in zip(starts[::3], losses[::3]):
        print(f"   open at t={s:4.1f} s -> removes {l:.0f} m")
    ax[1].bar(starts, losses, width=0.8)
    ax[1].set_title("B) Height removed by a 3 s full-open pulse, by start time")
    ax[1].set_xlabel("Pulse start time [s]"); ax[1].set_ylabel("Apogee reduction [m]")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step2_airbrake.png", dpi=130)
