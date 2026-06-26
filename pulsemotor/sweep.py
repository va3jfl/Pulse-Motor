"""
Parameter sweeps.

Run the drive cycle to steady state across a range of one design or operating
parameter and tabulate the steady-state metrics (RPM, powers, kickback/input
ratio).  The caller supplies a `build_motor(value) -> Motor` factory so any
parameter -- coil turns, air gap, source voltage, dwell, load, drag -- can be
swept uniformly.
"""
from __future__ import annotations

from typing import Callable, List

from .drive import DriveCycle
from .sim import Motor


def sweep(build_motor: Callable[[float], Motor], values, label: str = "value",
          drive_kwargs: dict | None = None, quiet: bool = True) -> List[dict]:
    """Run the drive cycle to steady state for each value; return metric dicts."""
    results: List[dict] = []
    for v in values:
        motor = build_motor(v)
        log = DriveCycle(motor=motor, **(drive_kwargs or {})).run()
        m = log.steady_metrics()
        m[label] = v
        m["settled"] = log.steady
        m["periods"] = len(log.rpm)
        if not quiet:
            print(f"  {label}={v}: rpm={m.get('rpm', 0):.0f} "
                  f"ratio_total/in={m.get('ratio_total_to_input', 0):.3f}")
        results.append(m)
    return results


def print_table(results: List[dict], label: str = "value") -> None:
    cols = [
        (label, 9, "g", label),
        ("rpm", 7, ".0f", "rpm"),
        ("rate_Hz", 7, ".0f", "pulse_rate_Hz"),
        ("Ipk_A", 7, ".3f", "I_peak_A"),
        ("Pin_W", 8, ".3f", "P_input_W"),
        ("Pspike_W", 9, ".3f", "P_kickback_total_W"),
        ("Precov_W", 9, ".3f", "P_kickback_recovered_W"),
    ]
    header = " ".join(f"{h:>{w}}" for h, w, _, _ in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        cells = []
        for h, w, fmt, key in cols:
            v = r.get(key, 0)
            cells.append(f"{float(v):>{w}{fmt}}")
        print(" ".join(cells))
