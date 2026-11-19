"""
Step 6: in-flight drag learning (adaptive apogee prediction).

During the coast, the only forces are gravity and drag, so
    measured drag deceleration   D_meas = -(a_meas - bias_est) - g
    model drag deceleration      D_nom  = 0.5 * rho * v^2 * (Cd(M) * A + Cd_b * u * A_b) / m_nominal
We fit one scale factor s with recursive least squares (with forgetting):
    s = sum(D_nom * D_meas) / sum(D_nom^2)
and the predictor multiplies all drag by s.
s absorbs body drag error, flap drag error, mass error and air density error together,
because the predictor only needs drag divided by mass in the real air.
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

N = 1000
FORGET = 0.995          # per accelerometer sample; older data slowly fades
LEARN_FROM = 0.3        # s after burnout before we trust the fit
V_MIN = 40.0            # m/s, below this drag is too small to measure well
PRIOR_SD = 0.15         # how far from nominal we expect each drag scale to be


def pred_scaled(h, v, s, dt=0.1):
    """Forward-sim apogee, flaps closed, body drag scaled by s."""
    while v > 0:
        rho, a_s = atmosphere(max(h, 0))
        k = 0.5 * rho * s * cd(v / a_s) * AREA
        acc = -G0 - k * v * v / DRY_MASS
        v_new = v + acc * dt
        if v_new <= 0:
            h += v * v / (2 * (-acc)); break
        h += 0.5 * (v + v_new) * dt
        v = v_new
    return h


def fly(p, adaptive, seed, record=False):
    rng = np.random.default_rng(seed)
    T0 = 288.15 + p["dT"]; T0_est = T0 + p["pad_T_err"]

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
    i, h_max = 0, 0.0
    M = np.zeros((2, 2)); r = np.zeros(2)
    s_est = f_est = 1.0
    log = []
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
        pr, _, a_s = atm_true(max(h, 0))
        h_baro = T0_est / 0.0065 * (1 - (pr / 101325) ** (1 / 5.2561))
        h_baro += -p["trans_amp"] * np.exp(-((abs(v) / a_s - p["trans_mach"]) / 0.06) ** 2)
        h_baro += rng.normal(0, BARO_NOISE)
        acc_m = a_true + p["bias"] + rng.normal(0, ACC_NOISE)
        h_est, v_est, b_est = kf.step(acc_m, h_baro)

        # --- drag learning (uses the flap position the computer commanded) ---
        if adaptive and t > BURN_TIME + 0.05 and v_est > V_MIN:
            rho, a_snd = atmosphere(max(h_est, 0))       # computer only knows the standard atmosphere
            phi_b = 0.5 * rho * v_est**2 * cd(v_est / a_snd) * AREA / DRY_MASS          # nominal body drag decel
            phi_f = 0.5 * rho * v_est**2 * CD_BRAKE * u * A_BRAKE_MAX / DRY_MASS         # nominal flap drag decel
            y = -(acc_m - b_est) - G0                                                    # measured drag decel
            if adaptive == 1:            # v1: one scale; flap drag assumed exactly nominal
                M[0, 0] = FORGET * M[0, 0] + phi_b * phi_b / ACC_NOISE**2
                r[0] = FORGET * r[0] + phi_b * (y - phi_f) / ACC_NOISE**2
            else:                        # v2: separate body and flap scales
                phi = np.array([phi_b, phi_f])
                M[:] = FORGET * M + np.outer(phi, phi) / ACC_NOISE**2
                r[:] = FORGET * r + phi * y / ACC_NOISE**2
            if t > BURN_TIME + LEARN_FROM:
                # Bayesian least squares with a prior of 1 +/- PRIOR_SD on each scale
                A = M + np.eye(2) / PRIOR_SD**2
                if adaptive == 1:
                    A[1, :] = A[:, 1] = 0; A[1, 1] = 1
                theta = np.linalg.solve(A, r + np.ones(2) / PRIOR_SD**2 * (np.array([1, 1]) if adaptive == 2 else np.array([1, 0])))
                s_est = float(np.clip(theta[0], 0.5, 2.0))
                f_est = float(np.clip(theta[1], 0.3, 3.0)) if adaptive == 2 else 1.0

        if t > BURN_TIME + 0.2 and v_est > 0 and i % CTRL_EVERY == 0:
            pred = pred_scaled(h_est, v_est, s_est if adaptive else 1.0)
            u_cmd = float(np.clip(KP * (pred - TARGET), 0, 1))
        if v_est <= 0:
            u_cmd = 0.0
        step = DT / p["servo"]
        u += np.clip(u_cmd - u, -step, step)
        if record and i % 5 == 0:
            log.append((t, s_est, f_est, u))
        i += 1
    return h_max, (np.array(log) if record else None)


def one(k):
    rng = np.random.default_rng(1000 + k)       # same random worlds as step 5
    p = draw(rng)
    return p, fly(p, 0, k)[0], fly(p, 1, k)[0], fly(p, 2, k)[0]


def stats(miss):
    lo, hi = np.percentile(miss, [2.5, 97.5])
    return miss.mean(), miss.std(), lo, hi, np.mean(abs(miss) <= 10)*100, np.mean(abs(miss) <= 25)*100


if __name__ == "__main__":
    # 1) Does the estimate converge? Single flights with known drag errors
    print("Drag learning on single flights (true body drag scale -> estimate at t=8 s, t=15 s):")
    base = draw(np.random.default_rng(1000))
    base.update(drag=1.0, dry=DRY_MASS, cd_brake=CD_BRAKE, bias=0.15, dT=0.0, pad_T_err=0.0)
    traces = {}
    for d, fc in ((0.85, 1.2), (1.0, 1.0), (1.15, 0.8)):
        q = dict(base, drag=d, cd_brake=CD_BRAKE * fc)
        ap, lg = fly(q, 2, 3, record=True)
        traces[(d, fc)] = lg
        at = lambda T, c: lg[np.argmin(abs(lg[:, 0] - T)), c]
        ap0, _ = fly(q, 0, 3); ap1, _ = fly(q, 1, 3)
        print(f"   true body {d:.2f} flap {fc:.2f} -> est body {at(8,1):.3f} flap {at(8,2):.3f} (t=8 s) | "
              f"miss fixed {ap0-TARGET:+6.1f} m, v1 {ap1-TARGET:+6.1f} m, v2 {ap-TARGET:+6.1f} m")

    # 2) Monte Carlo, same 1,000 worlds as step 5
    with Pool(2) as pool:
        res = pool.map(one, range(N))
    params = [r[0] for r in res]
    m_fix = np.array([r[1] for r in res]) - TARGET
    m_v1 = np.array([r[2] for r in res]) - TARGET
    m_ad = np.array([r[3] for r in res]) - TARGET
    for name, m in (("Fixed model (step 5)", m_fix), ("Learn body drag only", m_v1), ("Learn body + flap drag", m_ad)):
        mu, sd, lo, hi, w10, w25 = stats(m)
        print(f"{name:24s} mean {mu:+5.1f} m | 1 sigma {sd:4.1f} m | 95% {lo:+4.0f}/{hi:+4.0f} m | "
              f"within 10 m {w10:4.1f}% | within 25 m {w25:4.1f}%")
    names = {"drag": "Body drag", "pad_T_err": "Pad temp sensor", "dT": "Day temperature",
             "impulse": "Motor impulse", "dry": "Dry mass", "cd_brake": "Flap Cd"}
    print("Correlation with miss (fixed -> body only -> body + flap):")
    for k, n in names.items():
        x = [q[k] for q in params]
        print(f"   {n:16s} {np.corrcoef(x, m_fix)[0,1]:+.2f} -> {np.corrcoef(x, m_v1)[0,1]:+.2f} -> {np.corrcoef(x, m_ad)[0,1]:+.2f}")

    fig, ax = plt.subplots(3, 1, figsize=(7, 10))
    for j, ((d, fc), lg) in enumerate(traces.items()):
        ax[0].plot(lg[:, 0], lg[:, 1], c=f"C{j}", label=f"body (true {d:.2f})")
        ax[0].plot(lg[:, 0], lg[:, 2], c=f"C{j}", ls="--", label=f"flap (true {fc:.2f})")
        ax[0].axhline(d, ls=":", c=f"C{j}", lw=1); ax[0].axhline(fc, ls=":", c=f"C{j}", lw=1)
    ax[0].set_xlim(BURN_TIME, None); ax[0].set_ylim(0.6, 1.4)
    ax[0].set_title("Drag scales learned in flight (solid body, dashed flap, dotted truth)")
    ax[0].set_xlabel("Time [s]"); ax[0].legend(fontsize=7, ncol=3)
    bins = np.linspace(-80, 80, 41)
    ax[1].hist(m_fix, bins=bins, alpha=0.6, color="grey", label=f"Fixed model: 1 sigma {m_fix.std():.1f} m")
    ax[1].hist(m_v1, bins=bins, histtype="step", color="C3", lw=1.5, label=f"Body drag only: 1 sigma {m_v1.std():.1f} m")
    ax[1].hist(m_ad, bins=bins, alpha=0.8, color="C0", label=f"Body + flap drag: 1 sigma {m_ad.std():.1f} m")
    ax[1].set_title(f"{N} flights, same random worlds"); ax[1].set_xlabel("Apogee - target [m]"); ax[1].legend(fontsize=8)
    ax[2].scatter([q["drag"] for q in params], m_fix, s=4, c="grey", alpha=0.5, label="Fixed model")
    ax[2].scatter([q["drag"] for q in params], m_ad, s=4, c="C0", alpha=0.6, label="Body + flap drag")
    ax[2].axhline(0, c="k", lw=1); ax[2].set_xlabel("True body drag scale"); ax[2].set_ylabel("Miss [m]")
    ax[2].set_title("Drag error no longer tilts the result"); ax[2].legend(fontsize=8)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/step6_adaptive_drag.png", dpi=130)
