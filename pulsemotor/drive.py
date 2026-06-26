"""
Drive-cycle simulation.

Couples the Coil + Rotor + Circuit into a runnable machine and integrates it
period-by-period from a start flick up to the steady-state limit cycle, where the
average electromagnetic torque balances friction, windage and the load.

Method (quasi-static limit-cycle relaxation):
  For each magnet period, at the current speed omega:
    1. solve one drive pulse  -> mechanical work e_mech over the on-time
    2. apply the equivalent average torque T_em = e_mech / t_on over the dwell,
       then coast with T_em = 0 for the rest of the period
  Because the mechanical time constant (J/B) is far longer than one pulse, omega
  is effectively constant within a single pulse; the pulse is re-solved each
  period at the new omega, so the full spin-up transient and its equilibrium
  emerge honestly from the torque balance.

Outputs a SimLog time series (RPM, per-pulse energies, cumulative energies).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .sim import Motor
from .circuit import PulseResult


@dataclass
class SimLog:
    t: List[float] = field(default_factory=list)
    rpm: List[float] = field(default_factory=list)
    omega: List[float] = field(default_factory=list)
    e_in: List[float] = field(default_factory=list)
    e_kick_total: List[float] = field(default_factory=list)
    e_kick_rec: List[float] = field(default_factory=list)
    e_copper: List[float] = field(default_factory=list)
    e_mech: List[float] = field(default_factory=list)
    i_peak: List[float] = field(default_factory=list)
    v_kick_didt: List[float] = field(default_factory=list)
    cum_in: List[float] = field(default_factory=list)
    cum_kick_total: List[float] = field(default_factory=list)
    cum_kick_rec: List[float] = field(default_factory=list)
    cum_mech: List[float] = field(default_factory=list)
    steady: bool = False
    settle_index: int = -1

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in {
            "t": self.t, "rpm": self.rpm, "omega": self.omega,
            "e_in": self.e_in, "e_kick_total": self.e_kick_total,
            "e_kick_rec": self.e_kick_rec, "e_copper": self.e_copper,
            "e_mech": self.e_mech, "i_peak": self.i_peak,
            "v_kick_didt": self.v_kick_didt,
            "cum_in": self.cum_in, "cum_kick_total": self.cum_kick_total,
            "cum_kick_rec": self.cum_kick_rec, "cum_mech": self.cum_mech,
        }.items()}

    def steady_metrics(self, tail: int = 60) -> dict:
        """Average metrics over the last `tail` periods (the steady limit cycle)."""
        n = len(self.rpm)
        if n == 0:
            return {}
        a = max(0, n - tail)
        rpm = float(np.mean(self.rpm[a:]))
        rate = self._n_magnets * rpm / 60.0
        e_in = float(np.mean(self.e_in[a:]))
        e_kt = float(np.mean(self.e_kick_total[a:]))
        e_kr = float(np.mean(self.e_kick_rec[a:]))
        e_cu = float(np.mean(self.e_copper[a:]))
        e_m = float(np.mean(self.e_mech[a:]))
        ipk = float(np.mean(self.i_peak[a:]))
        vpk = float(np.mean(self.v_kick_didt[a:]))
        return {
            "rpm": rpm, "pulse_rate_Hz": rate,
            "I_peak_A": ipk, "V_kickback_peak_V": vpk,
            "P_input_W": e_in * rate,
            "P_kickback_total_W": e_kt * rate,
            "P_kickback_recovered_W": e_kr * rate,
            "P_mech_W": e_m * rate,
            "P_copper_W": e_cu * rate,
            "E_input_mJ": e_in * 1e3,
            "E_kickback_total_mJ": e_kt * 1e3,
            "E_kickback_recovered_mJ": e_kr * 1e3,
            "ratio_total_to_input": e_kt / e_in if e_in > 0 else float("nan"),
            "ratio_recovered_to_input": e_kr / e_in if e_in > 0 else float("nan"),
        }

    # filled by DriveCycle so steady_metrics can derive the pulse rate
    _n_magnets: int = 1


@dataclass
class DriveCycle:
    motor: Motor
    initial_omega: float = 8.0          # start flick [rad/s] (machine needs a start spin)
    max_periods: int = 2000
    dt_max: float = 2.0e-5              # max mechanical sub-step [s]
    pulse_on_steps: int = 700           # pulse solver resolution (drive-cycle mode)
    pulse_off_steps: int = 400
    settle_tol: float = 7.0e-4          # rel. omega change/period for "settled"
    settle_count: int = 50              # consecutive settled periods to confirm
    settle_margin: int = 80             # extra periods logged after settling
    min_omega: float = 0.5              # below this the machine is stalled

    def run(self) -> SimLog:
        coil, rotor, circuit = self.motor.coil, self.motor.rotor, self.motor.circuit
        load = self.motor.load_torque
        rotor.omega = self.initial_omega
        rotor.theta = 0.0
        fire_theta = rotor.fire_phase * rotor.magnet_step

        log = SimLog()
        log._n_magnets = rotor.n_magnets
        t = 0.0
        c_in = c_kt = c_kr = c_m = 0.0
        omega_prev = rotor.omega
        settle_run = 0
        settled_idx = -1

        for k in range(self.max_periods):
            omega = rotor.omega
            if omega <= self.min_omega:
                break
            period = rotor.magnet_step / omega
            dwell = min(circuit.dwell_fraction * period, 0.95 * period)
            pulse: PulseResult = circuit.pulse(
                coil, rotor, fire_theta, omega,
                on_steps=self.pulse_on_steps, off_steps=self.pulse_off_steps,
            )
            # Average electromagnetic torque over the FULL magnet period.
            # Work e_mech [J] is delivered once per period; the equivalent
            # constant torque is T_avg = e_mech / magnet_step [N*m] (work per
            # radian), NOT e_mech/dwell (which is power, wrong units).
            t_avg = pulse.e_mech / rotor.magnet_step
            n_m = max(1, int(math.ceil(period / self.dt_max)))
            dt_m = period / n_m
            for _ in range(n_m):
                rotor.mechanical_step(t_avg, load, dt_m)

            t += period
            c_in += pulse.e_input
            c_kt += pulse.e_kickback_total
            c_kr += pulse.e_kickback_recovered
            c_m += pulse.e_mech
            log.t.append(t)
            log.rpm.append(rotor.rpm)
            log.omega.append(rotor.omega)
            log.e_in.append(pulse.e_input)
            log.e_kick_total.append(pulse.e_kickback_total)
            log.e_kick_rec.append(pulse.e_kickback_recovered)
            log.e_copper.append(pulse.e_copper)
            log.e_mech.append(pulse.e_mech)
            log.i_peak.append(pulse.i_peak)
            log.v_kick_didt.append(pulse.v_kickback_didt)
            log.cum_in.append(c_in)
            log.cum_kick_total.append(c_kt)
            log.cum_kick_rec.append(c_kr)
            log.cum_mech.append(c_m)

            # steady-state detection
            if omega_prev > 0:
                rel = abs(rotor.omega - omega_prev) / omega_prev
                if rel < self.settle_tol:
                    settle_run += 1
                    if settled_idx < 0 and settle_run >= self.settle_count:
                        settled_idx = k
                        log.settle_index = k
                else:
                    settle_run = 0
            omega_prev = rotor.omega
            if settled_idx >= 0 and (k - settled_idx) >= self.settle_margin:
                log.steady = True
                break

        return log
