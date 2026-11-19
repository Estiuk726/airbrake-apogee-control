"""
Step 7: 2-D flight with wind and launch rail angle.

Truth model (3-DOF point mass, x = downrange, z = up):
  - On the rail: the rocket moves along the rail direction (RAIL_TILT from vertical).
  - Off the rail: a stable rocket points into the relative wind (zero angle of attack),
    so thrust acts along the air-relative velocity and drag acts against it.
  - Wind blows along +x (downrange, the way the rail is tilted) or -x (back over the pad), and grows with height:
        w(z) = w_ref * (max(z, 2) / 10) ** (1/7)
  - Mach and drag use AIRSPEED, not ground speed.

Flight computer:
  - Kalman filter on altitude (baro + vertical accelerometer from the IMU), unchanged.
  - Gyro gives pitch angle (angle of the rocket from horizontal) with 0.5 deg noise.
  - Three guidance versions:
      A. 1-D predictor, fixed drag model (steps 4-5)
      B. 1-D predictor + body/flap drag learning (step 6)
      C. Tilt-aware: uses gyro pitch to recover airspeed, 2-D predictor, drag learning
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sim_v1 import AREA, DRY_MASS, PROP_MASS, BURN_TIME, G0, THRUST_T, THRUST_F, atmosphere, cd
from sim_v2 import A_BRAKE_MAX, CD_BRAKE
from sensors import BARO_NOISE, ACC_NOISE
from kalman import AltitudeKF
from monte_carlo import draw, TOTAL_IMPULSE, TARGET, KP, DT, CTRL_EVERY

RAIL_LEN = 6.0                  # m
RAIL_TILT = np.radians(5.0)     # from vertical, tilted downrange (+x)
GYRO_NOISE = np.radians(0.5)
FORGET, PRIOR_SD, V_MIN, LEARN_FROM = 0.995, 0.15, 40.0, 0.3
N = 400


def wind(z, w_ref):
    return w_ref * (max(z, 2.0) / 10.0) ** (1 / 7)


def pred_1d(h, v, s=1.0):
    """Vertical-only forward sim, flaps closed."""
    dt = 0.1
    while v > 0:
        rho, a = atmosphere(max(h, 0))
        acc = -G0 - 0.5 * rho * s * cd(v / a) * AREA * v * v / DRY_MASS
        vn = v + acc * dt
        if vn <= 0:
            h += v * v / (2 * -acc); break
        h += 0.5 * (v + vn) * dt; v = vn
    return h


def pred_2d(h, vz, vx, s=1.0):
    """Tilted forward sim in still air, flaps closed: drag along the velocity vector."""
    dt = 0.1
    while vz > 0:
        rho, a = atmosphere(max(h, 0))
        V = np.hypot(vx, vz)
        k = 0.5 * rho * s * cd(V / a) * AREA * V / DRY_MASS
        az = -G0 - k * vz; ax = -k * vx
        vzn = vz + az * dt
        if vzn <= 0:
            h += vz * vz / (2 * -az); break
        h += 0.5 * (vz + vzn) * dt; vz = vzn; vx += ax * dt
    return h


def fly(p, w_ref, mode, seed, ctrl=True, record=False):
    """mode: 'A', 'B' or 'C' (see top). Returns apogee, downrange at apogee, log."""
    rng = np.random.default_rng(seed)
    T0 = 288.15 + p["dT"]; T0_est = T0 + p["pad_T_err"]

    def atm_true(z):
        T = T0 - 0.0065 * z
        pr = 101325 * (T / T0) ** 5.2561
        return pr, pr / (287.05 * T), np.sqrt(1.4 * 287.05 * T)

    def mass(t):
        if t >= BURN_TIME:
            return p["dry"]
        ts = np.linspace(0, t, 30)
        return p["dry"] + PROP_MASS * (1 - np.trapezoid(np.interp(ts, THRUST_T, THRUST_F), ts) / TOTAL_IMPULSE)

    def deriv(t, s, u):
        x, z, vx, vz = s
        thr = p["impulse"] * np.interp(t, THRUST_T, THRUST_F, right=0.0)
        m = mass(t)
        on_rail = np.hypot(x, z) < RAIL_LEN and t < 2
        if on_rail:
            ex, ez = np.sin(RAIL_TILT), np.cos(RAIL_TILT)
            vax, vaz = vx, vz                                   # rail guides it; ignore wind on the rail
        else:
            vax, vaz = vx - wind(z, w_ref), vz                  # air-relative velocity
            Va = np.hypot(vax, vaz)
            ex, ez = (vax / Va, vaz / Va) if Va > 1e-3 else (np.sin(RAIL_TILT), np.cos(RAIL_TILT))
        Va = np.hypot(vax, vaz)
        _, rho, a_s = atm_true(max(z, 0))
        D = 0.5 * rho * Va * Va * (p["drag"] * cd(Va / a_s) * AREA + p["cd_brake"] * u * A_BRAKE_MAX)
        ax = (thr - D) * ex / m
        az = (thr - D) * ez / m - G0
        if on_rail:                                             # rail removes motion across it
            a_al = ax * ex + (az + G0) * ez - G0 * ez
            ax, az = a_al * ex, a_al * ez
            if t < 0.2 and a_al < 0:
                ax = az = 0.0
        return np.array([vx, vz, ax, az]), np.arctan2(ez, ex), Va / a_s

    kf = AltitudeKF(DT, mach_lockout=(mode != "C"))      # A, B lock out on vertical speed; C on airspeed
    st = np.zeros(4); t = u = u_cmd = 0.0
    i, h_max = 0, 0.0
    M = np.zeros((2, 2)); r = np.zeros(2); sb = 1.0; pred = np.nan
    log = []
    while True:
        k1, pitch, mach = deriv(t, st, u)
        k2, _, _ = deriv(t + DT/2, st + DT/2*k1, u)
        k3, _, _ = deriv(t + DT/2, st + DT/2*k2, u)
        k4, _, _ = deriv(t + DT, st + DT*k3, u)
        st = st + DT/6*(k1 + 2*k2 + 2*k3 + k4); t += DT
        x, z, vx, vz = st
        h_max = max(h_max, z)
        if vz < 0 and t > BURN_TIME:
            break
        # sensors
        pr, _, _ = atm_true(max(z, 0))
        h_baro = T0_est / 0.0065 * (1 - (pr / 101325) ** (1 / 5.2561))
        h_baro += -p["trans_amp"] * np.exp(-((mach - p["trans_mach"]) / 0.06) ** 2) + rng.normal(0, BARO_NOISE)
        az_m = k1[3] + p["bias"] + rng.normal(0, ACC_NOISE)          # vertical accel from the IMU
        pitch_m = pitch + rng.normal(0, GYRO_NOISE)
        skip = False
        if mode == "C":                                              # airspeed Mach from vertical speed and gyro pitch
            vz_prev = kf.x[1]
            mach_air = abs(vz_prev) / max(np.sin(pitch_m), 0.2) / atmosphere(max(kf.x[0], 0))[1]
            skip = mach_air > 0.75
        h_e, vz_e, b_e = kf.step(az_m, h_baro, skip_baro=skip)
        sp = max(np.sin(pitch_m), 0.2)
        V_e = vz_e / sp                                              # airspeed estimate from the gyro (mode C)

        # drag learning (modes B and C), same Bayesian least squares as step 6
        if mode in "BC" and ctrl and t > BURN_TIME + 0.05 and vz_e > V_MIN:
            rho, a_snd = atmosphere(max(h_e, 0))
            if mode == "B":                                          # pretends flight is vertical
                q = 0.5 * rho * vz_e**2 / DRY_MASS
                phi = np.array([q * cd(vz_e / a_snd) * AREA, q * CD_BRAKE * u * A_BRAKE_MAX])
            else:                                                    # vertical part of drag along the tilted path
                q = 0.5 * rho * V_e * vz_e / DRY_MASS
                phi = np.array([q * cd(V_e / a_snd) * AREA, q * CD_BRAKE * u * A_BRAKE_MAX])
            y = -(az_m - b_e) - G0
            M = FORGET * M + np.outer(phi, phi) / ACC_NOISE**2
            r = FORGET * r + phi * y / ACC_NOISE**2
            if t > BURN_TIME + LEARN_FROM:
                th = np.linalg.solve(M + np.eye(2) / PRIOR_SD**2, r + np.ones(2) / PRIOR_SD**2)
                sb = float(np.clip(th[0], 0.5, 2.0))

        if ctrl and t > BURN_TIME + 0.2 and vz_e > 0 and i % CTRL_EVERY == 0:
            if mode == "C":
                pred = pred_2d(h_e, vz_e, vz_e / np.tan(np.clip(pitch_m, 0.2, np.pi - 0.2)), sb)
            else:
                pred = pred_1d(h_e, vz_e, sb if mode == "B" else 1.0)
            u_cmd = float(np.clip(KP * (pred - TARGET), 0, 1))
        if vz_e <= 0:
            u_cmd = 0.0
        stp = DT / p["servo"]
        u += np.clip(u_cmd - u, -stp, stp)
        if record and i % 5 == 0:
            log.append((t, x, z, np.degrees(pitch), u, pred, h_e, vz_e, sb, np.degrees(pitch_m)))
        i += 1
    return h_max, st[0], np.array(log) if record else None


def nominal():
    p = draw(np.random.default_rng(1000))
    p.update(drag=1.0, impulse=1.0, dry=DRY_MASS, cd_brake=CD_BRAKE, bias=0.15, dT=0.0,
             pad_T_err=0.0, trans_amp=120.0, trans_mach=0.9, servo=0.5)
    return p


def one(k):
    rng = np.random.default_rng(5000 + k)
    p = draw(rng)
    w = rng.uniform(-10, 10)                     # m/s at 10 m; negative = blowing towards the launcher
    out = [fly(p, w, "A", k, ctrl=False)[0]]
    out += [fly(p, w, m, k)[0] for m in "ABC"]
    return p, w, out


if __name__ == "__main__":
    p0 = nominal()
    print("1) Uncontrolled flights, nominal rocket, rail tilted 5 deg downrange")
    fig, ax = plt.subplots(3, 1, figsize=(7, 11))
    for w in (10, 0, -10):
        ap, xr, lg = fly(p0, w, "A", 1, ctrl=False, record=True)
        lab = "no wind" if w == 0 else (f"{abs(w)} m/s wind blowing back over the pad" if w < 0 else f"{w} m/s wind blowing downrange")
        print(f"   wind {w:+3d} m/s: apogee {ap:5.0f} m, drift at apogee {xr:+5.0f} m")
        ax[0].plot(lg[:, 1], lg[:, 2], label=f"{lab}: {ap:.0f} m")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("Downrange [m]"); ax[0].set_ylabel("Altitude [m]")
    ax[0].set_title("Uncontrolled 2-D trajectories (wind measured at 10 m)"); ax[0].legend(fontsize=8)

    print("2) Controlled, nominal rocket, by wind speed (miss in m)")
    winds = np.arange(-12, 13, 2)
    res = {m: [] for m in "ABC"}
    for w in winds:
        for m in "ABC":
            res[m].append(fly(p0, w, m, 1)[0] - TARGET)
    names = {"A": "1-D, fixed drag", "B": "1-D + drag learning", "C": "Tilt-aware + drag learning"}
    for m in "ABC":
        print(f"   {names[m]:28s} " + " ".join(f"{v:+5.0f}" for v in res[m]))
        ax[1].plot(winds, res[m], "o-", ms=3, label=names[m])
    ax[1].axhline(0, c="k", lw=1); ax[1].set_xlabel("Wind at 10 m [m/s] (+ blows downrange, - blows back over the pad)")
    ax[1].set_ylabel("Miss [m]"); ax[1].set_title("Controlled miss vs wind, nominal rocket"); ax[1].legend(fontsize=8)

    print(f"3) Monte Carlo: {N} flights, random rocket errors + wind -10..+10 m/s")
    with Pool(2) as pool:
        mc = pool.map(one, range(N))
    misses = {name: np.array([r[2][j] for r in mc]) - TARGET for j, name in enumerate(["none", "A", "B", "C"])}
    for key, label in (("none", "No airbrake"), ("A", names["A"]), ("B", names["B"]), ("C", names["C"])):
        m = misses[key]; lo, hi = np.percentile(m, [2.5, 97.5])
        print(f"   {label:28s} mean {m.mean():+6.1f} m | 1 sigma {m.std():5.1f} m | 95% {lo:+5.0f}/{hi:+5.0f} m")
    ws = np.array([r[1] for r in mc])
    ok = misses["none"] >= 0
    print(f"   Flights that could not reach {TARGET:.0f} m even with flaps closed: {np.sum(~ok)} of {N} "
          f"(wind range {ws[~ok].min():+.1f} to {ws[~ok].max():+.1f} m/s)" if np.any(~ok) else "   All flights could reach the target")
    print("   Reachable flights only:")
    for key in "ABC":
        m = misses[key][ok]; lo, hi = np.percentile(m, [2.5, 97.5])
        print(f"   {names[key]:28s} mean {m.mean():+6.1f} m | 1 sigma {m.std():5.1f} m | 95% {lo:+5.0f}/{hi:+5.0f} m | within 25 m {np.mean(abs(m)<=25)*100:4.1f}%")
    for key in "ABC":
        print(f"   corr(|wind|, miss) {names[key]:28s} {np.corrcoef(abs(ws), misses[key])[0,1]:+.2f}")
    bins = np.linspace(-150, 80, 47)
    for key, c in (("A", "grey"), ("B", "C1"), ("C", "C0")):
        ax[2].hist(np.clip(misses[key][ok], -150, 80), bins=bins, alpha=0.55 if key != "C" else 0.8, color=c,
                   label=f"{names[key]}: 1 sigma {misses[key][ok].std():.0f} m")
    ax[2].set_xlabel("Apogee - target [m] (left bar = -150 m or worse)")
    np.savez("figures/step7_mc.npz", winds=ws, **{f"miss_{k}": v for k, v in misses.items()})
    ax[2].set_title(f"{ok.sum()} of {N} flights that could reach the target (wind + rocket errors)")
    ax[2].legend(fontsize=8)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step7_wind_2d.png", dpi=130)
