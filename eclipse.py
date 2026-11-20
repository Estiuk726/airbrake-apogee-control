"""
Eclipse-like rocket model and the EuRoC official altitude logger.

Public numbers used (2026 press coverage of BME Suborbitals' Eclipse):
    liftoff mass            33 kg
    rail exit speed         164 km/h  (45.6 m/s)
    maximum speed           970 km/h  (269.4 m/s)
    airbrake active         900 m to 3,000 m
    target apogee           3,000 m (EuRoC S3 category)
EuRoC launch conditions (System Requirements Document 2025):
    rail length 12 m, elevation 84 deg (6 deg from vertical)

Everything else is an ASSUMPTION, marked as such below. The motor is calibrated so the
model reproduces the published rail exit speed and maximum speed.

Official logger (CATS Vega firmware, flight_computer/src/control/data_processing.cpp):
    height(p) = -((p / 101250)^(1/5.257) - 1) * (15 + 273.15) / 0.0065
    logged altitude = height(p) - height(p_pad)
The ground temperature is fixed at 15 C whatever the weather.
"""
from dataclasses import dataclass, replace
import numpy as np
from kalman import AltitudeKF

G0, R_AIR, GAMMA = 9.80665, 287.05, 1.4
DT = 0.01

# ---------------------------------------------------------------- atmosphere
@dataclass(frozen=True)
class Atmosphere:
    """Real atmosphere on launch day: ground temperature, ground pressure, lapse rate."""
    T0: float = 288.15         # K
    p0: float = 101325.0       # Pa
    lapse: float = 0.0065      # K/m

    def T(self, h):
        return max(self.T0 - self.lapse * h, 200.0)

    def p(self, h):
        return self.p0 * (self.T(h) / self.T0) ** (G0 / (R_AIR * self.lapse))

    def rho(self, h):
        return self.p(h) / (R_AIR * self.T(h))

    def a(self, h):
        return np.sqrt(GAMMA * R_AIR * self.T(h))

    def height_of_pressure(self, p):
        """Inverse of p(h): the true height at which this atmosphere has pressure p."""
        return self.T0 / self.lapse * (1 - (p / self.p0) ** (R_AIR * self.lapse / G0))


# ---------------------------------------------------------------- official logger
CATS_P_INITIAL, CATS_T0_C, CATS_EXP = 101250.0, 15.0, 1 / 5.257


def cats_height(p):
    return -((p / CATS_P_INITIAL) ** CATS_EXP - 1) * (CATS_T0_C + 273.15) / 0.0065


def cats_logged(p, p_pad):
    """Altitude the EuRoC CATS Vega logger reports for pressure p (pad pressure p_pad)."""
    return cats_height(p) - cats_height(p_pad)


def cats_pressure_for(alt, p_pad):
    """Pressure at which the CATS logger reads `alt` metres above the pad."""
    h = alt + cats_height(p_pad)
    return CATS_P_INITIAL * (1 - h * 0.0065 / (CATS_T0_C + 273.15)) ** (1 / CATS_EXP)


def isa_library_height(p, p_pad):
    """The common 'library' pressure-altitude formula (44330 * (1 - (p/p0)^0.1903)), relative to the pad."""
    f = lambda q: 44330.8 * (1 - (q / 101325.0) ** 0.190263)
    return f(p) - f(p_pad)


# ---------------------------------------------------------------- rocket
SHAPE_T = np.array([0, 0.025, 0.125, 0.75, 0.9, 1.0])             # thrust curve shape (fraction of burn time)
SHAPE_F = np.array([0, 1.0, 0.931, 0.862, 0.276, 0.0])            # (fraction of peak thrust)
MACH = np.array([0, 0.5, 0.8, 0.95, 1.05, 1.2, 2.0])
CD_SHAPE = np.array([1.0, 0.978, 1.022, 1.333, 1.556, 1.378, 1.111])  # Cd(M) / Cd(subsonic)


@dataclass(frozen=True)
class Rocket:
    name: str = "Eclipse-like"
    liftoff_mass: float = 33.0      # kg, PUBLIC
    diameter: float = 0.140         # m, ASSUMPTION
    cd_sub: float = 0.45            # subsonic body drag coefficient, ASSUMPTION
    isp: float = 185.0              # s, ASSUMPTION (sets propellant mass from impulse)
    impulse: float = 0.0            # Ns, CALIBRATED to public speeds
    burn: float = 0.0               # s, CALIBRATED to public speeds
    flap_area: float = 0.0100       # m^2 fully open, ASSUMPTION
    flap_cd: float = 1.2            # ASSUMPTION
    servo: float = 0.5              # s full travel, ASSUMPTION
    gain_m: float = 150.0           # predicted overshoot that commands full open
    brake_floor: float = 900.0      # m, PUBLIC: airbrake active from 900 m
    rail_len: float = 12.0          # m, EuRoC
    rail_tilt_deg: float = 6.0      # deg from vertical, EuRoC

    @property
    def area(self):
        return np.pi * self.diameter ** 2 / 4

    @property
    def prop_mass(self):
        return self.impulse / (self.isp * G0)

    @property
    def dry_mass(self):
        return self.liftoff_mass - self.prop_mass

    def thrust(self, t):
        if t >= self.burn:
            return 0.0
        shape_imp = np.trapezoid(SHAPE_F, SHAPE_T)
        return self.impulse / (shape_imp * self.burn) * np.interp(t / self.burn, SHAPE_T, SHAPE_F)

    def mass(self, t):
        if t >= self.burn:
            return self.dry_mass
        ts = np.linspace(0, t, 25)
        burned = np.trapezoid([self.thrust(x) for x in ts], ts) / self.impulse
        return self.liftoff_mass - self.prop_mass * burned

    def cd(self, mach):
        return self.cd_sub * np.interp(mach, MACH, CD_SHAPE)


# ---------------------------------------------------------------- flight
@dataclass
class World:
    """What really happens on the day (the flight computer does not know these)."""
    atm: Atmosphere = Atmosphere()
    wind: float = 0.0           # m/s at 10 m, + blows downrange
    drag: float = 1.0           # body drag scale
    impulse: float = 1.0        # motor impulse scale
    mass: float = 1.0           # dry mass scale
    flap_cd: float = 1.0        # flap drag scale
    acc_bias: float = 0.15      # m/s^2
    pad_temp_err: float = 0.0   # K, error of the flight computer's pad temperature reading
    trans_amp: float = 120.0    # m, transonic baro dip (illustrative)
    trans_mach: float = 0.9
    servo: float = 1.0


def wind_at(z, w):
    return w * (max(z, 2.0) / 10.0) ** (1 / 7)


def _std_atm(h):   # the flight computer's own physics model of the air: standard atmosphere
    T = 288.15 - 0.0065 * h
    p = 101325 * (T / 288.15) ** 5.2559
    return p / (R_AIR * T), np.sqrt(GAMMA * R_AIR * T)


def _predict(h, vz, vx, s, rk: Rocket):
    """Flight computer's apogee predictor: tilted path, still air, flaps closed, body drag scaled by s."""
    dt = 0.1
    while vz > 0:
        rho, a = _std_atm(max(h, 0))
        V = np.hypot(vx, vz)
        k = 0.5 * rho * s * rk.cd(V / a) * rk.area * V / rk.dry_mass
        az = -G0 - k * vz
        vzn = vz + az * dt
        if vzn <= 0:
            return h + vz * vz / (2 * -az)
        h += 0.5 * (vz + vzn) * dt
        vz = vzn
        vx += -k * vx * dt
    return h


ALT_MODES = ("true", "isa", "logger")


def fly(rk: Rocket, w: World, *, ctrl=True, alt_mode="logger", target=3000.0, seed=0, record=False,
        fixed_u=None):
    """
    One 2-D flight. alt_mode is how the TEAM's flight computer turns pressure into altitude:
      "true"   - uses its measured pad temperature, aims for 3,000 m of true height
      "isa"    - common library formula (15 C, 1013.25 hPa), aims for 3,000 m in those units
      "logger" - uses its measured pad temperature for physics, but converts the target into the
                 pressure at which the CATS logger reads 3,000 m (i.e. aims for a pressure)
    Returns dict with true apogee, logged apogee (CATS), drift and an optional log.
    """
    rng = np.random.default_rng(seed)
    atm = w.atm
    tilt = np.radians(rk.rail_tilt_deg)
    p_pad = atm.p0
    # The flight computer's own conversion from pressure to altitude above the pad
    meas_atm = Atmosphere(T0=atm.T0 + w.pad_temp_err, p0=p_pad, lapse=0.0065)
    if alt_mode == "isa":
        to_alt = lambda p: isa_library_height(p, p_pad)
        tgt = target
    else:
        to_alt = meas_atm.height_of_pressure
        tgt = target if alt_mode == "true" else meas_atm.height_of_pressure(cats_pressure_for(target, p_pad))

    def deriv(t, s, u):
        x, z, vx, vz = s
        thr = w.impulse * rk.thrust(t)
        m = rk.mass(t) - rk.dry_mass * (1 - w.mass)
        on_rail = np.hypot(x, z) < rk.rail_len and t < 3
        if on_rail:
            ex, ez = np.sin(tilt), np.cos(tilt)
            vax, vaz = vx, vz
        else:
            vax, vaz = vx - wind_at(z, w.wind), vz
            V = np.hypot(vax, vaz)
            ex, ez = (vax / V, vaz / V) if V > 1e-3 else (np.sin(tilt), np.cos(tilt))
        V = np.hypot(vax, vaz)
        zz = max(z, 0)
        D = 0.5 * atm.rho(zz) * V * V * (w.drag * rk.cd(V / atm.a(zz)) * rk.area + rk.flap_cd * w.flap_cd * u * rk.flap_area)
        ax, az = (thr - D) * ex / m, (thr - D) * ez / m - G0
        if on_rail:
            al = ax * ex + az * ez
            if t < 0.3 and al < 0:
                al = 0.0
            ax, az = al * ex, al * ez
        return np.array([vx, vz, ax, az]), np.arctan2(ez, ex), V / atm.a(zz), on_rail

    kf = AltitudeKF(DT, mach_lockout=False)
    st = np.zeros(4)
    t = u = u_cmd = 0.0
    i = 0
    z_max, p_at_max, x_at_max = 0.0, p_pad, 0.0
    M = np.zeros((2, 2)); r = np.zeros(2); sb = 1.0
    v_rail = None; v_max = 0.0
    log = []
    while True:
        U = fixed_u if fixed_u is not None and t > rk.burn else (0.0 if fixed_u is not None else u)
        k1, pitch, mach, on_rail = deriv(t, st, U)
        k2 = deriv(t + DT / 2, st + DT / 2 * k1, U)[0]
        k3 = deriv(t + DT / 2, st + DT / 2 * k2, U)[0]
        k4 = deriv(t + DT, st + DT * k3, U)[0]
        st = st + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += DT
        x, z, vx, vz = st
        V = np.hypot(vx, vz); v_max = max(v_max, V)
        if v_rail is None and not on_rail and t > 0.05:
            v_rail = V
        if z > z_max:
            z_max, p_at_max, x_at_max = z, atm.p(z), x
        if (vz < 0 and t > rk.burn) or t > 120:
            break
        if not ctrl and not record:
            continue
        # sensors
        p_meas = atm.p(max(z, 0))
        h_baro = to_alt(p_meas)
        h_baro += -w.trans_amp * np.exp(-((mach - w.trans_mach) / 0.06) ** 2) + rng.normal(0, 2.0)
        az_m = k1[3] + w.acc_bias + rng.normal(0, 0.5)
        pitch_m = pitch + rng.normal(0, np.radians(0.5))
        sp = max(np.sin(pitch_m), 0.2)
        mach_est = abs(kf.x[1]) / sp / _std_atm(max(kf.x[0], 0))[1]
        h_e, vz_e, b_e = kf.step(az_m, h_baro, skip_baro=0.75 < mach_est < 1.2)
        V_e = vz_e / sp
        # in-flight drag learning (body and flap scales)
        if ctrl and t > rk.burn + 0.05 and vz_e > 40:
            rho, a = _std_atm(max(h_e, 0))
            q = 0.5 * rho * V_e * vz_e / rk.dry_mass
            phi = np.array([q * rk.cd(V_e / a) * rk.area, q * rk.flap_cd * u * rk.flap_area])
            y = -(az_m - b_e) - G0
            M = 0.995 * M + np.outer(phi, phi) / 0.25
            r = 0.995 * r + phi * y / 0.25
            if t > rk.burn + 0.3:
                th = np.linalg.solve(M + np.eye(2) / 0.15 ** 2, r + np.ones(2) / 0.15 ** 2)
                sb = float(np.clip(th[0], 0.5, 2.0))
        pred = np.nan
        if ctrl and t > rk.burn + 0.2 and vz_e > 0 and i % 5 == 0:
            pc = np.clip(pitch_m, 0.2, np.pi - 0.2)
            pred = _predict(h_e, vz_e, vz_e / np.tan(pc), sb, rk)
            u_cmd = float(np.clip((pred - tgt) / rk.gain_m, 0, 1)) if h_e >= rk.brake_floor else 0.0
        if ctrl and vz_e <= 0:
            u_cmd = 0.0
        if ctrl:
            step = DT / (rk.servo * w.servo)
            u += np.clip(u_cmd - u, -step, step)
        if record and i % 5 == 0:
            log.append((t, x, z, u, pred, h_e))
        i += 1
    return dict(apogee=z_max, logged=cats_logged(p_at_max, p_pad), drift=x_at_max,
                v_rail=v_rail, v_max=v_max, log=np.array(log) if record else None)


# ---------------------------------------------------------------- calibration
def calibrate(base: Rocket = Rocket(), v_rail_target=164 / 3.6, v_max_target=970 / 3.6):
    """Find impulse and burn time so the model matches the published rail exit and maximum speeds
    (standard day, no wind, airbrake closed)."""
    calm = World(acc_bias=0.0, trans_amp=0.0)
    burn, imp = 3.0, 8000.0
    for _ in range(30):
        rk = replace(base, impulse=imp, burn=burn)
        res = fly(rk, calm, ctrl=False)
        e_rail = res["v_rail"] / v_rail_target
        e_max = res["v_max"] / v_max_target
        # rail exit speed is set mainly by initial thrust (impulse / burn); max speed mainly by impulse
        imp_new = imp / e_max ** 1.2
        burn = burn * (e_rail ** 2) * (imp_new / imp)
        imp = imp_new
        if abs(e_rail - 1) < 2e-4 and abs(e_max - 1) < 2e-4:
            break
    return replace(base, impulse=imp, burn=burn)


if __name__ == "__main__":
    rk = calibrate()
    calm = World(acc_bias=0.0, trans_amp=0.0)
    c = fly(rk, calm, ctrl=False)
    o = fly(rk, calm, ctrl=False, fixed_u=1.0)
    print(f"Calibrated motor: {rk.impulse:.0f} Ns over {rk.burn:.2f} s "
          f"(avg thrust {rk.impulse / rk.burn:.0f} N), propellant {rk.prop_mass:.2f} kg, dry {rk.dry_mass:.2f} kg")
    print(f"Check: rail exit {c['v_rail'] * 3.6:.1f} km/h (public 164), max speed {c['v_max'] * 3.6:.1f} km/h (public 970), "
          f"max Mach {c['v_max'] / 340.3:.2f}")
    print(f"Standard day, no wind: apogee flaps closed {c['apogee']:.0f} m, flaps fully open after burnout {o['apogee']:.0f} m")
