# Switched-Reluctance Pulse-Motor Simulation

A parameterised Python simulation of a switched-reluctance pulse motor
(Bedini topology): stator coil/core electromagnetics, rotating magnet
dynamics, and the transistor drive + inductive flyback ("kickback") transients.


## Layout

```
pulsemotor/
  constants.py   physical constants, AWG table, soft-magnetic material DB
  coil.py        Coil  -- saturating inductance L(i), resistance, impedance, SRF
  rotor.py       Rotor -- magnets, L(theta), torque, back-EMF, mechanics
  circuit.py     Circuit -- drive pulse + flyback solver; PulseResult
  sim.py         Motor  -- couples Coil+Rotor+Circuit to a held operating point
  drive.py       DriveCycle -- spin-up to the steady-state limit cycle; SimLog
  sweep.py       parameter sweeps over steady-state metrics
verify_phase1.py         single held-speed pulse, prints the breakdown
phase2.py                drive cycle + sweeps + figures
requirements.txt
```

## Running

```bash
python verify_phase1.py   # Phase 1: one pulse, raw per-event energies
python phase2.py          # Phase 2: drive cycle + sweeps -> figs/*.png
python run_gui.py         # Phase 3: live GUI (needs a display)
```

## Class outline (Phase 1 deliverable)

**Coil** — stator winding + core.
- Geometry from AWG (standard `0.005*92^((36-n)/39)` inch formula) and material.
- `resistance_dc` from a multi-layer wire-length estimate; `resistance_ac(f)` with skin effect.
- `inductance(current)` = `mu0 * mu_eff * N^2 * A / l`, with two real corrections:
  - open-rod demagnetising factor `Nd` so a straight rod realises only
    `mu_eff = mur / (1 + Nd(mur-1))`, not the material `mur`;
  - smooth saturation roll-off toward `mu0` as the core hits `B_sat`.
- `saturation_current`, `impedance(f,i)`, Medhurst self-capacitance, self-resonant freq.

**Rotor** — rotating magnets + magnetic coupling.
- Moment of inertia from geometry (disk + magnets) or explicit override.
- Position-dependent inductance `L(theta) = base * (1 + swing * profile(theta))`,
  giving, exactly, torque `T = 1/2 * i^2 * dL/dtheta` and rotational back-EMF
  `e = i * dL/dtheta * omega` (the variable-reluctance machine equations).
- Flux coupling, trigger-winding EMF, trigger/on-time timing from speed.
- Symplectic-Euler mechanical step `J dw/dt = T - T_load - B*w`.

**Circuit** — drive + flyback transients (the critical calculator).
- ON-time: solves `V_s - Vce = R i + L_inc di/dt + i*omega*dL/dtheta`, where
  `L_inc = d(Li)/di` (differential inductance) is used so the current ramp stays
  correct as the core saturates.
- OFF-time: releases the stored field energy `W_field = integral L_inc(i)*i di`
  (= `1/2 L I^2` exactly when unsaturated) -- the **total inductive kickback
  energy**. Peak flyback voltage is reported as three uncapped estimates
  (`V=L di/dt`, LC-ring, clamp level). Recovery into the charge battery is
  integrated along the real path; diode and winding-R losses are separate,
  labelled terms.
- `PulseResult` carries `e_input`, `e_kickback_total`, `e_kickback_recovered`,
  the ratios, peak voltages, spike sharpness `dt`, and per-rate powers.

## Energy model (raw, per-event, no balancing)

Each drive pulse is reported event-by-event with its **raw spike TRIGGER (drive
input) energy** `E_in = integral V_s i dt` and its **raw spike OUTPUT (inductive
kickback) energy** `E_kickback = integral L_inc(i) i di = 1/2 L I_pk^2` (the
collapsing field energy, taken straight from the coil physics). The kickback is
computed directly from the coil, never derived as "input minus losses", so it is
not reverse-engineered to balance anything.

The only quantities subtracted from the *recovered* (battery-bound) energy are
the genuine measurable parasitics in the real current path -- transistor
`Vce(sat)`, diode `Vf`, winding resistance during the flyback decay -- reported
as explicit line items, never capped or hidden. Open-system gain accounting
(output-vs-input ratios etc.) is intentionally left to the analyst; the tool
emits the raw per-event energies and powers.

## Reference output (900t AWG23, mild-steel rod core, 8 magnets, 12 V)

Free-running steady-state limit cycle (`phase2.py`):

```
RPM                              378
pulse rate                       50 Hz
I_peak                           1.17 A
spike TRIGGER (input) / event    46 mJ
spike OUTPUT (kickback) / event  33 mJ
  -> delivered to charge battery 27 mJ
P_input                          2.32 W
P_kickback_total                 1.68 W
P_kickback_recovered             1.34 W
```

Source-voltage sweep (experimental, 1 V - 1 kV) settles to a finite, bounded
RPM at each point -- no runaway:

```
 1 V ->   41 rpm      24 V ->  577 rpm      100 V -> 1664 rpm
12 V ->  378 rpm      48 V ->  871 rpm     1000 V -> 30423 rpm
```

Held-speed single pulse (`verify_phase1.py`) emits the same raw per-event
energies for one event.

## Phase 3 -- live GUI (`pulsemotor/gui.py`, PySide6/Qt6)

A live dashboard driven by the same physics core:

- **Rotor canvas** -- QPainter wheel with the magnets at the correct angular
  spacing, the stator coil across the air gap (glows when energised), hub, and a
  live RPM readout; rotates at the simulated omega.
- **Dashboard gauges** -- RPM, pulse rate, I_pk, L, kickback peak V, and the raw
  per-event trigger/spike/recovered energies and powers.
- **Rolling graphs** -- rotor speed and per-event energies, updated live.
- **Controls** -- source voltage on a **log slider (1 V - 1 kV)**, dwell, turns,
  load (live reconfigure, no restart), a **sim-speed** slider (0.1x - 4x,
  slow-motion to fast-forward), and **Pause/Reset**. A live **sim-time counter**
  sits at the top of the dashboard.

The physics runs in a worker thread paced to wall-clock x time-scale, so Pause
freezes the sim and slow-motion works on the rotor, graphs and timer together.
Run on a real display with `run_gui.py`; verified headless via `gui_smoketest.py`
(screenshot in `figs/gui_screenshot.png`).

## Roadmap

- Phase 1: physics core + verification. **done**
- Phase 2: drive-cycle to steady state, parameter sweeps, dashboard figures. **done**
- Phase 3: live GUI (PySide6) -- rotor canvas, gauges, graphs, controls. **done**

## Material data sources

Typical soft-magnetic permeability / B_sat ranges set from standard reference
ranges (silicon steel, MnZn/NiZn ferrite, iron powder, nanocrystalline,
amorphous, permalloy). Wire resistivities are IACS-reference values. AWG from the
ASTM gauge definition. All values are representative and grade-dependent.
