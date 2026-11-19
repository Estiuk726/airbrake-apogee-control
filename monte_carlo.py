"""
Step 5: Monte Carlo robustness test.

Each flight draws a random "real world". The flight computer does NOT know these values:
it still uses the nominal model (nominal drag, mass, standard atmosphere).

Uncertainty                         Distribution
Rocket body drag (Cd scale)         normal, mean 1, 7% sigma
Motor total impulse (scale)         normal, mean 1, 3% sigma
Dry mass                            normal, 18 kg, 0.3 kg sigma
Airbrake flap Cd                    normal, 1.2, 10% sigma
Accelerometer bias                  normal, 0, 0.2 m/s^2 sigma
Day temperature vs standard         uniform, -10 to +15 K
Pad temperature sensor error        normal, 0, 1 K sigma   (used to calibrate the baro)
Transonic baro error size           uniform, 60-180 m, centred at Mach 0.85-0.95
Servo full-travel time              uniform, 0.4-0.7 s

Not modelled yet (1-D sim): wind and launch angle. That needs a 2-D/3-D sim, a later upgrade.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sim_v1 import AREA, DRY_MASS, PROP_MASS, BURN_TIME, G0, THRUST_T, THRUST_F, cd
from sim_v2 import A_BRAKE_MAX, CD_BRAKE
from sensors import BARO_NOISE, ACC_NOISE
from kalman import AltitudeKF
from predictors import pred_forward

TARGET, KP, DT, CTRL_EVERY = 3000.0, 1 / 150, 0.01, 5
N = 1000
TOTAL_IMPULSE = np.trapezoid(THRUST_F, THRUST_T)


def draw(rng):
    return dict(
        drag=rng.normal(1, 0.07), impulse=rng.normal(1, 0.03),
        dry=rng.normal(DRY_MASS, 0.3), cd_brake=rng.normal(CD_BRAKE, 0.12),
        bias=rng.normal(0, 0.2), dT=rng.uniform(-10, 15), pad_T_err=rng.normal(0, 1),
        trans_amp=rng.uniform(60, 180), trans_mach=rng.uniform(0.85, 0.95),
        servo=rng.uniform(0.4, 0.7),
    )


def fly(p, controller, seed):
    rng = np.random.default_rng(seed)
    T0 = 288.15 + p["dT"]                        # real ground temperature today
    T0_est = T0 + p["pad_T_err"]                 # what the pad sensor tells the computer

    def atm_true(h):
        T = T0 - 0.0065 * h
        pr = 101325 * (T / T0) ** 5.2561
        return pr, pr / (287.05 * T), np.sqrt(1.4 * 287.05 * T)

    def thrust(t):
        return p["impulse"] * np.interp(t, THRUST_T, THRUST_F, right=0.0)

    def mass(t):
        if t >= BURN_TIME:
            return p["dry"]
        ts = np.linspace(0, t, 30)
        return p["dry"] + PROP_MASS * (1 - np.trapezoid(np.interp(ts, THRUST_T, THRUST_F), ts) / TOTAL_IMPULSE)

    def accel(t, h, v, u):
        _, rho, a_s = atm_true(max(h, 0))
        drag = 0.5 * rho * v * abs(v) * (p["drag"] * cd(abs(v) / a_s) * AREA + p["cd_brake"] * u * A_BRAKE_MAX)
        return (thrust(t) - drag) / mass(t) - G0

    kf = AltitudeKF(DT, mach_lockout=True)
    t = h = v = u = u_cmd = 0.0
    i, h_max, u_sum = 0, 0.0, 0.0
    while True:
        k1 = (v, accel(t, h, v, u))
        k2 = (v + DT/2*k1[1], accel(t+DT/2, h+DT/2*k1[0], v+DT/2*k1[1], u))
        k3 = (v + DT/2*k2[1], accel(t+DT/2, h+DT/2*k2[0], v+DT/2*k2[1], u))
        k4 = (v + DT*k3[1],   accel(t+DT,   h+DT*k3[0],   v+DT*k3[1],   u))
        a_true = k1[1]
        h += DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); v += DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); t += DT
        h_max = max(h_max, h)
        if v < 0 and t > BURN_TIME:
            break
        if not controller:
            continue
        # sensors: baro converts pressure to altitude using the pad-measured temperature
        pr, _, a_s = atm_true(max(h, 0))
        h_baro = T0_est / 0.0065 * (1 - (pr / 101325) ** (1 / 5.2561))
        h_baro += -p["trans_amp"] * np.exp(-((abs(v) / a_s - p["trans_mach"]) / 0.06) ** 2)
        h_baro += rng.normal(0, BARO_NOISE)
        acc_m = a_true + p["bias"] + rng.normal(0, ACC_NOISE)
        h_est, v_est, _ = kf.step(acc_m, h_baro)
        if t > BURN_TIME + 0.2 and v_est > 0 and i % CTRL_EVERY == 0:
            pred = pred_forward(h_est, v_est, u=0.0, dt=0.1)
            u_cmd = float(np.clip(KP * (pred - TARGET), 0, 1))
        if v_est <= 0:
            u_cmd = 0.0
        step = DT / p["servo"]
        u += np.clip(u_cmd - u, -step, step)
        u_sum += u * DT
        i += 1
    return h_max, u_sum


def one(k):
    rng = np.random.default_rng(1000 + k)
    p = draw(rng)
    ap_off, _ = fly(p, controller=False, seed=k)
    ap_on, effort = fly(p, controller=True, seed=k)
    return p, ap_off, ap_on, effort


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(one, range(N))
    params = [r[0] for r in res]
    off = np.array([r[1] for r in res]); on = np.array([r[2] for r in res])
    miss = on - TARGET
    lo, hi = np.percentile(miss, [2.5, 97.5])
    print(f"Flights: {N}")
    print(f"No airbrake: mean {off.mean():.0f} m, spread (1 sigma) {off.std():.0f} m, "
          f"range {off.min():.0f}-{off.max():.0f} m")
    print(f"Controlled:  mean miss {miss.mean():+.1f} m, 1 sigma {miss.std():.1f} m")
    print(f"95% of flights land between {lo:+.0f} m and {hi:+.0f} m of target")
    for tol in (10, 25, 50):
        print(f"   within +/-{tol} m: {np.mean(abs(miss) <= tol)*100:.1f}%")
    print(f"Flights that could not reach target even with brake closed: {np.sum(off < TARGET)}")

    # Sensitivity: which uncertainty drives the miss?
    names = {"drag": "Body drag", "impulse": "Motor impulse", "dry": "Dry mass",
             "cd_brake": "Flap Cd", "bias": "Accel bias", "dT": "Day temperature",
             "pad_T_err": "Pad temp sensor", "trans_amp": "Transonic error",
             "servo": "Servo speed"}
    corr = {names[k]: np.corrcoef([q[k] for q in params], miss)[0, 1] for k in names}
    order = sorted(corr, key=lambda k: abs(corr[k]))
    print("Correlation of each uncertainty with the miss:")
    for k in reversed(order):
        print(f"   {k:16s} {corr[k]:+.2f}")
    np.savez("figures/mc_results.npz", off=off, on=on, miss=miss)

    fig, ax = plt.subplots(3, 1, figsize=(7, 11))
    ax[0].hist(off, bins=40, color="grey", alpha=0.7, label="No airbrake")
    ax[0].hist(on, bins=40, color="C0", label="Controlled")
    ax[0].axvline(TARGET, c="k", ls=":")
    ax[0].set_title(f"{N} simulated flights: where they land"); ax[0].set_xlabel("Apogee [m]")
    ax[0].legend(fontsize=8)
    ax[1].hist(miss, bins=40, color="C0")
    ax[1].axvspan(lo, hi, color="C1", alpha=0.15, label=f"95% band: {lo:+.0f} to {hi:+.0f} m")
    ax[1].set_title("Controlled flights: miss distance (zoomed)"); ax[1].set_xlabel("Apogee - target [m]")
    ax[1].legend(fontsize=8)
    ax[2].barh(order, [corr[k] for k in order], color=["C3" if corr[k] < 0 else "C2" for k in order])
    ax[2].set_title("What drives the miss? (correlation with miss)"); ax[2].set_xlim(-1, 1)
    ax[2].axvline(0, c="k", lw=1)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step5_monte_carlo.png", dpi=130)
