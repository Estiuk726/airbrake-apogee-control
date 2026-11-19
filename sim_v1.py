"""
Step 1: 1-D vertical rocket flight simulator (no airbrake yet).

Forces on the rocket (positive = up):
    F = Thrust - Weight - Drag
    a = F / m

Placeholder rocket, roughly EuRoC 3 km class. Replace with real numbers later.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Rocket parameters (placeholders) ----------------
DIAMETER = 0.13                     # m
AREA = np.pi * (DIAMETER / 2) ** 2  # m^2, frontal area
DRY_MASS = 18.0                     # kg, rocket without propellant
PROP_MASS = 4.5                     # kg, propellant
BURN_TIME = 4.0                     # s
G0 = 9.80665

# Simple thrust curve: quick rise, slow tail-off (time [s], thrust [N])
THRUST_T = np.array([0.0, 0.1, 0.5, 3.0, 3.6, 4.0])
THRUST_F = np.array([0.0, 2900, 2700, 2500, 800, 0.0]) * 0.86
TOTAL_IMPULSE = np.trapezoid(THRUST_F, THRUST_T)


def thrust(t):
    return np.interp(t, THRUST_T, THRUST_F, right=0.0)


def mass(t):
    # Propellant burns in proportion to impulse delivered so far
    if t >= BURN_TIME:
        return DRY_MASS
    ts = np.linspace(0, t, 50)
    burned = np.trapezoid(thrust(ts), ts) / TOTAL_IMPULSE
    return DRY_MASS + PROP_MASS * (1 - burned)


# ---------------- Atmosphere (ISA, troposphere) ----------------
def atmosphere(h):
    T = 288.15 - 0.0065 * h                     # K
    p = 101325 * (T / 288.15) ** 5.2561         # Pa
    rho = p / (287.05 * T)                      # kg/m^3
    a = np.sqrt(1.4 * 287.05 * T)               # speed of sound, m/s
    return rho, a


# ---------------- Drag coefficient vs Mach ----------------
# Typical shape: flat when subsonic, sharp rise near Mach 1 (transonic)
MACH_PTS = np.array([0.0, 0.5, 0.8, 0.95, 1.05, 1.2, 2.0])
CD_PTS = np.array([0.45, 0.44, 0.46, 0.60, 0.70, 0.62, 0.50])


def cd(mach):
    return np.interp(mach, MACH_PTS, CD_PTS)


# ---------------- Equations of motion ----------------
def derivatives(t, state):
    h, v = state
    rho, a_sound = atmosphere(max(h, 0))
    mach = abs(v) / a_sound
    drag = 0.5 * rho * v * abs(v) * cd(mach) * AREA   # opposes motion
    m = mass(t)
    acc = (thrust(t) - drag) / m - G0
    return np.array([v, acc])


def rk4_step(t, state, dt):
    k1 = derivatives(t, state)
    k2 = derivatives(t + dt / 2, state + dt / 2 * k1)
    k3 = derivatives(t + dt / 2, state + dt / 2 * k2)
    k4 = derivatives(t + dt, state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(dt=0.01):
    t, state = 0.0, np.array([0.0, 0.0])
    log = [(t, *state, 0.0)]
    while True:
        new = rk4_step(t, state, dt)
        t += dt
        acc = (new[1] - state[1]) / dt
        state = new
        log.append((t, *state, acc))
        if state[1] < 0 and t > BURN_TIME:   # started falling: apogee passed
            break
    return np.array(log)


if __name__ == "__main__":
    data = simulate()
    t, h, v, a = data.T
    i = np.argmax(h)
    _, a0 = atmosphere(0)
    print(f"Total impulse: {TOTAL_IMPULSE:.0f} Ns")
    print(f"Apogee: {h[i]:.0f} m at t = {t[i]:.1f} s")
    print(f"Max velocity: {v.max():.0f} m/s (Mach {v.max()/a0:.2f})")
    print(f"Max acceleration: {a.max()/G0:.1f} g")

    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(t, h); ax[0].set_ylabel("Altitude [m]")
    ax[0].axhline(h[i], ls="--", c="grey"); ax[0].text(1, h[i]*0.93, f"Apogee {h[i]:.0f} m")
    ax[1].plot(t, v); ax[1].set_ylabel("Velocity [m/s]")
    ax[2].plot(t, a / G0); ax[2].set_ylabel("Acceleration [g]")
    ax[2].set_xlabel("Time [s]")
    for x in ax:
        x.axvline(BURN_TIME, ls=":", c="red"); x.grid(alpha=0.3)
    ax[0].set_title("Step 1: vertical flight, no airbrake (red line = burnout)")
    fig.tight_layout()
    fig.savefig("figures/step1_flight.png", dpi=130)
