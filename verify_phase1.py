"""
Phase-1 verification harness.

Assembles a reference Coil + Rotor + Circuit (a small switched-reluctance
pulse-motor topology), runs the machine at a held operating speed, and prints the
physics breakdown.  The headline pair is the TOTAL inductive kickback energy vs
the supply INPUT energy; peak flyback voltages and spike sharpness follow.

Run with the project venv:
    .venv/bin/python verify_phase1.py
"""
from __future__ import annotations

import math

from pulsemotor import Coil, Rotor, Circuit
from pulsemotor.sim import Motor


def fmt(name: str, value: float, unit: str, scale: float = 1.0) -> str:
    return f"  {name:<34} {value*scale:>14.4g} {unit}"


def main() -> None:
    # ---- reference parameterisation (a small bench-top build) --------------
    coil = Coil(
        turns=900,
        awg=23.0,
        wire_material="copper_enamelled",
        winding_length=0.060,
        core_material="mild_steel",
        core_diameter=0.010,
        core_length=0.100,
        core_closed=False,
    )
    rotor = Rotor(
        wheel_diameter=0.150,
        magnet_radius=0.070,
        rotor_mass=0.250,
        n_magnets=8,
        magnet_remanence=1.32,
        magnet_length=0.012,
        magnet_area=1.0e-4,
        magnet_mass=0.020,
        air_gap=0.002,
        inductance_swing=0.35,
        fire_phase=0.65,
        friction_coeff=5.0e-6,
    )
    circuit = Circuit(
        source_voltage=12.0,
        dwell_fraction=0.30,
        vce_sat=0.20,
        turn_off_time=50.0e-9,
        charge_battery_voltage=12.0,
        diode_vf=0.70,
        c_parasitic=25.0e-12,
    )
    motor = Motor(coil, rotor, circuit, load_torque=0.0)

    target_rpm = 600.0

    print("=" * 74)
    print(" PULSE-MOTOR SIMULATION  --  Phase 1 physics verification")
    print("=" * 74)

    print("\n[Coil & core electromagnetics]")
    s = coil.summary()
    for k, v in s.items():
        if isinstance(v, str):
            print(f"  {k:<28} {v:>14} ")
            continue
        u = ""
        if k.endswith("_mm2"):
            u = "mm^2"
        elif k.endswith("_mm"):
            u = "mm"
        elif k.endswith("_mH"):
            u = "mH"
        print(f"  {k:<28} {v:>14.4g} {u}")

    print(f"\n[Rotor]  J = {rotor.moment_of_inertia*1e3:.3f} g*m^2   "
          f"magnet step = {math.degrees(rotor.magnet_step):.1f} deg   "
          f"peak B at core ~ {rotor.peak_flux_density_at_core():.3f} T")

    print(f"\n[Operating point]  held at {target_rpm:.0f} RPM   "
          f"({target_rpm*2*math.pi/60:.1f} rad/s)   "
          f"pulse rate = {rotor.n_magnets*target_rpm/60.0:.1f} Hz")
    out = motor.settle(target_rpm=target_rpm)
    p: Circuit = out["pulse"]
    pr = out["pulse_rate_Hz"]

    print("\n" + "-" * 74)
    print(" RAW per-event spike energies  (no input/output balancing applied)")
    print("-" * 74)
    print(fmt("Spike TRIGGER (drive input) energy", p.e_input, "mJ", 1e3))
    print(fmt("Spike OUTPUT (inductive kickback) energy", p.e_kickback_total, "mJ", 1e3)
          + "   <-- 1/2 L I^2 from coil field")
    print(fmt("  -> delivered to charge battery", p.e_kickback_recovered, "mJ", 1e3))

    print("\n[Pulse electricals]")
    print(fmt("Peak winding current I_pk", p.i_peak, "A"))
    print(fmt("Chord inductance L at I_pk", p.L_at_peak * 1e3, "mH"))
    print(fmt("On-time (dwell)", p.t_on * 1e6, "us"))
    print(fmt("Flyback decay time", p.t_decay * 1e6, "us"))
    print(fmt("Spike sharpness dt (t_off)", p.dt_sharp * 1e9, "ns"))

    print("\n[Peak flyback voltage  -- three estimates, uncapped]")
    print(fmt("V = L di/dt  (collapse in t_off)", p.v_kickback_didt, "V"))
    print(fmt("LC ring peak (unclamped, C_par)", p.v_kickback_lc, "V"))
    print(fmt("Clamped level (V_batt + V_f)", p.v_kickback_clamped, "V"))

    print("\n[Average powers at %.1f Hz pulse rate]" % pr)
    pw = out["power"]
    for k, v in pw.items():
        print(f"  {k:<26} {v:>12.4g} W")

    print("\nDone. Phase-1 physics core is wired and producing figures.\n")


if __name__ == "__main__":
    main()
