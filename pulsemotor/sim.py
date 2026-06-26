"""
Steady-state coupling of Coil + Rotor + Circuit.

Finds the equilibrium speed of the assembled motor at a given load torque by
iterating the mechanical dynamics, firing one circuit pulse per magnet crossing.
Used by the verification harness and (later) by the GUI dashboard.
"""
from __future__ import annotations

import math

from .coil import Coil
from .rotor import Rotor
from .circuit import Circuit, PulseResult


class Motor:
    """Couples the three Phase-1 objects into a runnable machine model."""

    def __init__(self, coil: Coil, rotor: Rotor, circuit: Circuit,
                 load_torque: float = 0.0):
        self.coil = coil
        self.rotor = rotor
        self.circuit = circuit
        self.load_torque = load_torque

    def settle(self, target_rpm: float | None = None, revolutions: int = 40,
               dt: float = 1e-5) -> dict:
        """
        Run the machine until it reaches a steady speed.

        If `target_rpm` is given the rotor is held at that speed (quasi-static
        electrical/mechanical coupling) and the per-pulse result is reported at
        that speed -- this is the clean way to read off the kickback/input
        figures at a known operating point.  Otherwise the rotor is free to find
        its own equilibrium under the load.
        """
        if target_rpm is not None:
            omega = target_rpm * 2.0 * math.pi / 60.0
            self.rotor.omega = omega
            self.rotor.theta = 0.0
            # sample one pulse at the firing angle implied by the rotor phase
            theta_fire = self.rotor.fire_phase * self.rotor.magnet_step
            res = self.circuit.pulse(self.coil, self.rotor, theta_fire, omega)
            pulse_rate = self.rotor.n_magnets * target_rpm / 60.0
            return {
                "rpm": target_rpm,
                "omega": omega,
                "pulse_rate_Hz": pulse_rate,
                "pulse": res,
                "power": res.power_at(pulse_rate),
                "held_speed": True,
            }

        # free run: integrate mechanical dynamics, fire a pulse at each crossing
        last = None
        steps = int(revolutions * self.rotor.n_magnets
                    * self.rotor.magnet_step / (self.rotor.omega * dt)) if self.rotor.omega > 0 else 0
        for _ in range(max(steps, 1)):
            theta_fire = self.rotor.fire_phase * self.rotor.magnet_step
            res = self.circuit.pulse(self.coil, self.rotor, theta_fire, self.rotor.omega)
            # average electromagnetic torque over one magnet period
            pulse_period = self.rotor.magnet_step / max(self.rotor.omega, 1e-9)
            t_avg = res.e_mech / pulse_period if pulse_period > 0 else 0.0
            self.rotor.mechanical_step(t_avg, self.load_torque, dt)
            last = res
        pulse_rate = self.rotor.n_magnets * self.rotor.rpm / 60.0
        return {
            "rpm": self.rotor.rpm,
            "omega": self.rotor.omega,
            "pulse_rate_Hz": pulse_rate,
            "pulse": last,
            "power": last.power_at(pulse_rate) if last else {},
            "held_speed": False,
        }
