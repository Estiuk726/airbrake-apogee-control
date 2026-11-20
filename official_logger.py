"""
Step 8: score what the judges measure.

EuRoC scores apogee from the CATS Vega logger, which converts pressure to altitude assuming a
15 C ground temperature. This script asks: how far off is the LOGGED apogee when the team's
airbrake computer works in different altitude units?

  true   - flight computer aims for 3,000 m of true height (measured pad temperature)
  isa    - flight computer uses the common library formula (15 C, 1013.25 hPa) and aims for 3,000
  logger - flight computer aims for the pressure at which CATS reads 3,000 m

Outputs: results/official_logger.json and figures/step8_*.png
"""
import json, os
import numpy as np
from multiprocessing import Pool
from dataclasses import replace
from eclipse import (Rocket, World, Atmosphere, calibrate, fly, cats_logged, ALT_MODES)

N = 240
LABEL = {"true": "Aims for true 3,000 m", "isa": "Library formula, mixed into the filter", "logger": "Aims for the logger's 3,000 m"}
RK = None


def init():
    global RK
    RK = calibrate()


def closed_form():
    """Logger reading of a true 3,000 m, and true height needed for a logged 3,000 m."""
    rows = []
    for Tc in (5, 10, 15, 20, 25, 30, 35):
        for lapse in (0.0055, 0.0065, 0.0080):
            atm = Atmosphere(T0=Tc + 273.15, p0=101000.0, lapse=lapse)
            logged = cats_logged(atm.p(3000.0), atm.p0)
            lo, hi = 2000.0, 4000.0
            for _ in range(60):
                m = (lo + hi) / 2
                lo, hi = (m, hi) if cats_logged(atm.p(m), atm.p0) < 3000 else (lo, m)
            rows.append(dict(T=Tc, lapse=lapse * 1000, logged_of_true_3000=logged, true_for_logged_3000=m))
    return rows


def sweep_one(args):
    Tc, mode = args
    w = World(atm=Atmosphere(T0=Tc + 273.15, p0=101000.0), acc_bias=0.15)
    r = fly(RK, w, alt_mode=mode, seed=3)
    return Tc, mode, r["apogee"], r["logged"]


def draw_world(k):
    g = np.random.default_rng(9000 + k)
    return World(
        atm=Atmosphere(T0=g.uniform(12, 30) + 273.15, p0=g.uniform(99500, 102000), lapse=g.uniform(0.005, 0.008)),
        wind=g.uniform(-6, 6), drag=g.normal(1, 0.07), impulse=g.normal(1, 0.03), mass=g.normal(1, 0.015),
        flap_cd=g.normal(1, 0.10), acc_bias=g.normal(0, 0.2), pad_temp_err=g.normal(0, 1.0),
        trans_amp=g.uniform(60, 180), trans_mach=g.uniform(0.85, 0.95), servo=g.uniform(0.8, 1.4))


def mc_one(k):
    w = draw_world(k)
    closed = fly(RK, w, ctrl=False, seed=k)
    out = dict(k=k, T=w.atm.T0 - 273.15, lapse=w.atm.lapse * 1000, wind=w.wind,
               closed_true=closed["apogee"], closed_logged=closed["logged"])
    for m in ALT_MODES:
        r = fly(RK, w, alt_mode=m, seed=k)
        out[m + "_true"] = r["apogee"]; out[m + "_logged"] = r["logged"]
    return out


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    init()
    cf = closed_form()
    print("1) Logger reading of a TRUE 3,000 m (standard lapse 6.5 K/km):")
    for row in cf:
        if abs(row["lapse"] - 6.5) < 1e-6:
            print(f"   ground {row['T']:2d} C: logged {row['logged_of_true_3000']:6.0f} m | true height for a logged 3,000 m: {row['true_for_logged_3000']:6.0f} m")

    with Pool(2, initializer=init) as pool:
        sw = pool.map(sweep_one, [(T, m) for T in (5, 10, 15, 20, 25, 30, 35) for m in ALT_MODES])
        print("2) Nominal rocket, calm, perfect pad sensor - LOGGED miss by ground temperature:")
        for m in ALT_MODES:
            print(f"   {LABEL[m]:36s} " + "  ".join(f"{T:2d}C:{lg - 3000:+6.1f}" for T, mm, _, lg in sw if mm == m))
        mc = pool.map(mc_one, range(N))

    reach = [r for r in mc if r["closed_logged"] >= 3000]
    print(f"3) Monte Carlo: {N} flights, {len(reach)} could reach a logged 3,000 m with flaps closed")
    summary = {}
    for m in ALT_MODES:
        miss = np.array([r[m + "_logged"] - 3000 for r in reach])
        lo, hi = np.percentile(miss, [2.5, 97.5])
        Ts = np.array([r["T"] for r in reach])
        slope = np.polyfit(Ts, miss, 1)[0]
        summary[m] = dict(mean=float(miss.mean()), sd=float(miss.std()), p2_5=float(lo), p97_5=float(hi),
                          within10=float(np.mean(abs(miss) <= 10) * 100), within25=float(np.mean(abs(miss) <= 25) * 100),
                          slope_m_per_C=float(slope))
        print(f"   {LABEL[m]:36s} mean {miss.mean():+6.1f} m | 1 sigma {miss.std():5.1f} m | 95% {lo:+5.0f}/{hi:+5.0f} m | "
              f"within 25 m {np.mean(abs(miss) <= 25) * 100:4.1f}% | {slope:+.1f} m per C")
    json.dump(dict(rocket=dict(impulse=RK.impulse, burn=RK.burn, prop=RK.prop_mass, dry=RK.dry_mass,
                               diameter=RK.diameter, cd_sub=RK.cd_sub, flap_area=RK.flap_area),
                   closed_form=cf, sweep=[dict(T=a, mode=b, true=c, logged=d) for a, b, c, d in sw],
                   mc=mc, summary=summary, n=N, n_reach=len(reach)),
              open("results/official_logger.json", "w"), indent=1)
    print("saved results/official_logger.json")
