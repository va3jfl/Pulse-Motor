"""
Phase-2 dashboard: drive-cycle spin-up, energy time series, and parameter sweeps.

Generates two PNG figures (Agg backend, no display needed):
    figs/phase2_drive.png   -- spin-up + per-pulse/cumulative energies + ratio(t)
    figs/phase2_sweep.png   -- steady RPM and kickback/input ratio vs swept param

Run:  .venv/bin/python phase2.py
"""
from __future__ import annotations

import os
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pulsemotor import Coil, Rotor, Circuit
from pulsemotor.sim import Motor
from pulsemotor.drive import DriveCycle
from pulsemotor.sweep import sweep, print_table

FIGS = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS, exist_ok=True)

# reference operating point (self-starting, motoring region)
REF = dict(source_voltage=12.0, dwell=0.30, turns=900, air_gap=0.002,
           fire_phase=0.65, load_torque=0.0, drag_coeff=3e-8, friction=8e-6)


def build_motor(source_voltage=REF["source_voltage"], dwell=REF["dwell"],
                turns=REF["turns"], air_gap=REF["air_gap"],
                fire_phase=REF["fire_phase"], load_torque=REF["load_torque"],
                drag_coeff=REF["drag_coeff"], friction=REF["friction"]) -> Motor:
    coil = Coil(turns=turns, awg=23, wire_material="copper_enamelled",
                winding_length=0.060, core_material="mild_steel",
                core_diameter=0.010, core_length=0.100, core_closed=False)
    rotor = Rotor(wheel_diameter=0.150, magnet_radius=0.070, rotor_mass=0.250,
                  n_magnets=8, magnet_remanence=1.32, magnet_length=0.012,
                  magnet_area=1.0e-4, magnet_mass=0.020, air_gap=air_gap,
                  inductance_swing=0.35, fire_phase=fire_phase,
                  friction_coeff=friction, drag_coeff=drag_coeff)
    circuit = Circuit(source_voltage=source_voltage, dwell_fraction=dwell,
                      vce_sat=0.20, turn_off_time=50e-9, charge_battery_voltage=12.0,
                      diode_vf=0.70, c_parasitic=25e-12)
    return Motor(coil, rotor, circuit, load_torque=load_torque)


def run_drive() -> dict:
    print("=" * 74)
    print(" PHASE 2 -- drive-cycle simulation (spin-up to steady-state limit cycle)")
    print("=" * 74)
    t0 = time.time()
    log = DriveCycle(motor=build_motor()).run()
    dt = time.time() - t0
    sm = log.steady_metrics()
    print(f"  simulated {len(log.rpm)} magnet periods in {dt:.1f}s; "
          f"settled={log.steady} (at period {log.settle_index})")
    print("\n  Steady-state limit cycle (mean of last 60 periods):")
    for k in ("rpm", "pulse_rate_Hz", "I_peak_A", "V_kickback_peak_V",
              "E_input_mJ", "E_kickback_total_mJ", "E_kickback_recovered_mJ",
              "P_input_W", "P_kickback_total_W", "P_kickback_recovered_W",
              "P_mech_W", "P_copper_W"):
        v = sm.get(k, float("nan"))
        print(f"    {k:<26} {v:>12.4g}")
    return {"log": log, "sm": sm}


def plot_drive(log, sm) -> str:
    A = log.arrays()
    t = A["t"]
    rate = log._n_magnets * A["omega"] / (2.0 * math.pi)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Phase 2  --  drive cycle: spin-up & energy  "
                 f"(steady ~{sm.get('rpm',0):.0f} RPM, "
                 f"kickback/input={sm.get('ratio_total_to_input',0):.2f})", fontsize=12)

    # (0,0) RPM spin-up
    ax[0, 0].plot(t, A["rpm"], color="tab:blue")
    if log.settle_index >= 0:
        ax[0, 0].axvline(t[log.settle_index], color="r", ls="--", alpha=0.6, label="settled")
    ax[0, 0].set_title("Rotor speed (spin-up)")
    ax[0, 0].set_ylabel("RPM")
    ax[0, 0].set_xlabel("time [s]")
    ax[0, 0].grid(alpha=0.3); ax[0, 0].legend()

    # (0,1) per-pulse energies
    ax[0, 1].plot(t, A["e_in"] * 1e3, label="input", color="tab:orange")
    ax[0, 1].plot(t, A["e_kick_total"] * 1e3, label="kickback total", color="tab:green")
    ax[0, 1].plot(t, A["e_kick_rec"] * 1e3, label="recovered", color="tab:purple", ls="--")
    ax[0, 1].set_title("Energy per pulse")
    ax[0, 1].set_ylabel("[mJ]")
    ax[0, 1].set_xlabel("time [s]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].grid(alpha=0.3, which="both"); ax[0, 1].legend()

    # (1,0) cumulative energy
    ax[1, 0].plot(t, A["cum_in"], label="cumulative input", color="tab:orange")
    ax[1, 0].plot(t, A["cum_kick_rec"], label="cumulative kickback recovered", color="tab:purple")
    ax[1, 0].plot(t, A["cum_mech"], label="cumulative mechanical", color="tab:brown")
    ax[1, 0].set_title("Cumulative energy")
    ax[1, 0].set_ylabel("[J]")
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].grid(alpha=0.3); ax[1, 0].legend()

    # (1,1) instantaneous power & ratio
    axr = ax[1, 1]
    axr.plot(t, A["e_in"] * rate, label="P input", color="tab:orange")
    axr.plot(t, A["e_kick_total"] * rate, label="P kickback total", color="tab:green")
    axr.plot(t, A["e_kick_rec"] * rate, label="P recovered", color="tab:purple", ls="--")
    axr.set_title("Instantaneous power")
    axr.set_ylabel("[W]")
    axr.set_xlabel("time [s]")
    axr.set_yscale("log")
    axr.grid(alpha=0.3, which="both"); axr.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGS, "phase2_drive.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_sweeps() -> list:
    print("\n" + "=" * 74)
    print(" PHASE 2 -- parameter sweeps (each run to steady state)")
    print("=" * 74)

    print("\n-- Sweep: source voltage [V] --")
    sv = sweep(lambda v: build_motor(source_voltage=v),
               [4, 6, 8, 10, 12, 16, 20, 24], label="Vsrc")
    print_table(sv, "Vsrc")

    print("\n-- Sweep: dwell fraction --")
    dw = sweep(lambda d: build_motor(dwell=d),
               [0.10, 0.18, 0.25, 0.30, 0.35, 0.42], label="dwell")
    print_table(dw, "dwell")

    print("\n-- Sweep: turns --")
    tn = sweep(lambda n: build_motor(turns=int(n)),
               [400, 600, 800, 900, 1200, 1600], label="turns")
    print_table(tn, "turns")
    return [sv, dw, tn]


def plot_sweeps(sweeps) -> str:
    sv, dw, tn = sweeps
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase 2  --  steady-state response to design / operating parameters",
                 fontsize=13)

    def _pair(a_rpm, a_e, data, label, title):
        xs = [d[label] for d in data]
        a_rpm.plot(xs, [d["rpm"] for d in data], "o-", color="tab:blue")
        a_rpm.set_title(f"{title}: RPM")
        a_rpm.set_xlabel(label); a_rpm.set_ylabel("RPM"); a_rpm.grid(alpha=0.3)
        # raw per-event energies (no balancing / no gain)
        a_e.plot(xs, [d["E_input_mJ"] for d in data], "s-", color="tab:orange", label="trigger in")
        a_e.plot(xs, [d["E_kickback_total_mJ"] for d in data], "D-", color="tab:green", label="spike out")
        a_e.set_title(f"{title}: per-event energy")
        a_e.set_xlabel(label); a_e.set_ylabel("[mJ]"); a_e.grid(alpha=0.3); a_e.legend()

    _pair(ax[0, 0], ax[1, 0], sv, "Vsrc", "source voltage")
    _pair(ax[0, 1], ax[1, 1], dw, "dwell", "dwell fraction")
    _pair(ax[0, 2], ax[1, 2], tn, "turns", "coil turns")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIGS, "phase2_sweep.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    res = run_drive()
    p1 = plot_drive(res["log"], res["sm"])
    print(f"\n  saved: {p1}")
    sweeps = run_sweeps()
    p2 = plot_sweeps(sweeps)
    print(f"\n  saved: {p2}")
    print("\nPhase 2 done.\n")


if __name__ == "__main__":
    main()
