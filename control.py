"""
Step 4b: first closed-loop flight.

Loop (controller at 20 Hz, physics at 100 Hz):
  truth -> sensors -> Kalman filter -> apogee predictor -> controller -> flap servo -> truth

Controller (proportional on predicted overshoot):
  predicted = forward-sim apogee assuming flaps CLOSED from now on
  u_cmd = clip(KP * (predicted - TARGET), 0, 1)
When the prediction reaches the target, u -> 0 and the rocket coasts exactly to it.

Servo: flaps can't jump instantly; max rate = 1 / SERVO_FULL_TRAVEL per second.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_v1 import AREA, BURN_TIME, G0, thrust, mass, atmosphere, cd
from sim_v2 import A_BRAKE_MAX, CD_BRAKE
from sensors import transonic_error, BARO_NOISE, ACC_NOISE, ACC_BIAS
from kalman import AltitudeKF
from predictors import pred_forward

TARGET = 3000.0
KP = 1 / 150            # full open when predicted overshoot >= 150 m
SERVO_FULL_TRAVEL = 0.5 # s, closed -> fully open
DT = 0.01
CTRL_EVERY = 5          # 100 Hz / 5 = 20 Hz controller


def accel(t, h, v, u):
    rho, a_s = atmosphere(max(h, 0))
    drag = 0.5 * rho * v * abs(v) * (cd(abs(v) / a_s) * AREA + CD_BRAKE * u * A_BRAKE_MAX)
    return (thrust(t) - drag) / mass(t) - G0


def fly(controller=True, kp=KP, seed=1):
    rng = np.random.default_rng(seed)
    kf = AltitudeKF(DT, mach_lockout=True)
    t, h, v, u, u_cmd = 0.0, 0.0, 0.0, 0.0, 0.0
    log = []
    i = 0
    while True:
        # --- physics (RK4, flap position held during the step) ---
        k1 = (v, accel(t, h, v, u))
        k2 = (v + DT/2*k1[1], accel(t+DT/2, h+DT/2*k1[0], v+DT/2*k1[1], u))
        k3 = (v + DT/2*k2[1], accel(t+DT/2, h+DT/2*k2[0], v+DT/2*k2[1], u))
        k4 = (v + DT*k3[1],   accel(t+DT,   h+DT*k3[0],   v+DT*k3[1],   u))
        a_true = k1[1]
        h += DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); v += DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); t += DT
        # --- sensors ---
        mach = abs(v) / atmosphere(max(h, 0))[1]
        baro = h + transonic_error(mach) + rng.normal(0, BARO_NOISE)
        acc_m = a_true + ACC_BIAS + rng.normal(0, ACC_NOISE)
        # --- estimator ---
        h_est, v_est, _ = kf.step(acc_m, baro)
        # --- controller ---
        pred = np.nan
        if controller and t > BURN_TIME + 0.2 and v_est > 0 and i % CTRL_EVERY == 0:
            pred = pred_forward(h_est, v_est, u=0.0, dt=0.1)
            u_cmd = float(np.clip(kp * (pred - TARGET), 0, 1))
        if v_est <= 0:
            u_cmd = 0.0
        # --- servo rate limit ---
        max_step = DT / SERVO_FULL_TRAVEL
        u += np.clip(u_cmd - u, -max_step, max_step)
        log.append((t, h, v, h_est, v_est, u, pred))
        i += 1
        if v < 0 and t > BURN_TIME:
            break
    return np.array(log)


if __name__ == "__main__":
    off = fly(controller=False)
    on = fly(controller=True)
    print(f"No airbrake:   apogee {off[:,1].max():.1f} m")
    print(f"With control:  apogee {on[:,1].max():.1f} m  (target {TARGET:.0f})")

    t = on[:, 0]
    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    ax[0].plot(off[:, 0], off[:, 1], "--", c="grey", label=f"No airbrake: {off[:,1].max():.0f} m")
    ax[0].plot(t, on[:, 1], label=f"Controlled: {on[:,1].max():.1f} m")
    ax[0].axhline(TARGET, c="k", lw=1, ls=":"); ax[0].set_ylabel("Altitude [m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Closed-loop airbrake flight")
    m = ~np.isnan(on[:, 6])
    ax[1].plot(t[m], on[m, 6], ".", ms=2, label="Predicted apogee (flaps closed from now)")
    ax[1].axhline(TARGET, c="k", lw=1, ls=":"); ax[1].set_ylabel("Prediction [m]"); ax[1].legend(fontsize=8)
    ax[2].plot(t, on[:, 5] * 100); ax[2].set_ylabel("Flap opening [%]"); ax[2].set_xlabel("Time [s]")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step4b_closed_loop.png", dpi=130)
