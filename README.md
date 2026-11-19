# Airbrake Apogee Control

Simulation, state estimation and Monte Carlo validation of an airbrake controller for a competition rocket targeting a 3,000 m apogee.

**Result:** across 1,000 randomised flights, 95% landed within **-46 m to +36 m** of target (1-sigma 20 m), against a 1-sigma spread of 157 m without the airbrake.

![Monte Carlo result](figures/step5_monte_carlo.png)

## What it does

The full guidance chain of a flight computer, built step by step:

```
true flight -> barometer + accelerometer -> Kalman filter -> apogee predictor -> controller -> flap servo -> true flight
```

| Step | File | What it adds | Key result |
| --- | --- | --- | --- |
| 1 | `sim_v1.py` | 1-D flight sim: RK4, ISA atmosphere, Mach-dependent drag, mass loss | Uncontrolled apogee 3,503 m |
| 2 | `sim_v2.py` | Airbrake drag model, authority and timing study | Full flaps remove ~970 m; braking late is almost useless |
| 3a | `sensors.py` | Barometer noise and transonic error, accelerometer noise and bias | Baro-derived velocity error +/-149 m/s; accel drift 52 m |
| 3b | `kalman.py` | 3-state Kalman filter [altitude, velocity, accel bias], gating vs Mach lockout | Max altitude error 142 m -> 3 m with Mach lockout |
| 4a | `predictors.py` | Energy, analytic-drag and forward-simulation apogee predictors | Forward sim within 6 m at burnout |
| 4b | `control.py` | Closed-loop proportional controller with servo rate limit | 3,003 m against a 3,000 m target |
| 5 | `monte_carlo.py` | 1,000 flights with 9 random uncertainties, sensitivity analysis | 95% band -46/+36 m; pad temperature is the top error source |
| 6 | `adaptive.py` | In-flight learning of body and flap drag scales (Bayesian least squares) | 1-sigma 19.9 -> 18.5 m; drag error removed as a cause |

## Run it

```bash
pip install -r requirements.txt
python sim_v1.py        # each step script writes its figure to figures/
python monte_carlo.py   # about 2-3 minutes on 2 cores
```

## Interactive demo

`demo/index.html` is a self-contained JavaScript port of the same simulator. Open it in a browser to:

- enter your own rocket (diameter, Cd, masses, motor impulse and burn time, flap area, servo speed, controller gain) or pick a 3 km or 9 km preset
- change the real-world conditions the flight computer does not know about
- fly 100 to 1,000 Monte Carlo rockets with adjustable uncertainty spreads, and view the miss histogram, apogees with and without the airbrake, a sensitivity ranking, or miss vs drag error

## Main findings

- **Braking must happen early.** Drag scales with velocity squared, so a 3 s full-open pulse removes 577 m at burnout but only 35 m twelve seconds later.
- **Do not trust the barometer near Mach 1.** A plain Kalman filter blamed the transonic pressure dip on the accelerometer and was 87 m/s wrong in velocity at the moment braking matters most. A Mach-based lockout fixed it.
- **Predict, do not just react.** An energy-only apogee estimate is 1,947 m wrong at burnout; an onboard forward simulation is 6 m wrong.
- **The pad temperature sensor matters most.** It sets the pressure-to-altitude conversion; 1 K of error is about 10 m at 3 km, the largest single contributor to the miss.

## Limitations

- 1-D vertical flight: no wind, weathercocking or rail angle.
- Rocket, motor and flap values are placeholders, not a real vehicle.
- The transonic barometer error and the Cd-Mach curve are illustrative.

## Next steps

1. In-flight drag estimation (adaptive control) to remove the drag-mismatch error.
2. 2-D / 3-DOF simulation with wind and launch angle.
3. Real vehicle data from OpenRocket or CFD.
4. Port the estimator and controller to C on a microcontroller and run hardware-in-the-loop against this simulator.

## Author

A. E. M. Estiuk Ahmed, BSc Professional Pilot, Budapest University of Technology and Economics.
