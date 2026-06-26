"""
Rotor & magnetic dynamics.

Models the rotating magnet assembly and its magnetic coupling to the stator coil.

The cleanest faithful way to capture *both* the electromechanical torque and the
rotational back-EMF in one energy-consistent framework is a position-dependent
inductance L(theta): as a magnet aligns with the core, the magnetic reluctance of
the flux path drops and the coil inductance rises.  Standard variable-reluctance
energy conversion then gives, exactly,

        T(theta, i) = 1/2 * i^2 * dL/dtheta            (torque)
        e_rot(theta) = i * dL/dtheta * dtheta/dt       (rotational back-EMF)
        J * domega/dt = T - T_load - B*omega           (mechanical dynamics)

These relations are the textbook switched-reluctance / variable-reluctance machine
equations; they couple electrical and mechanical power without ad-hoc factors.

The inductance swing (fractional change of L as a magnet aligns) and the flux
coupling are geometry/material parameters, not free knobs -- they come from the
magnet strength, gap, and core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import constants as K


@dataclass
class Rotor:
    # --- geometry ---
    wheel_diameter: float = 0.150        # rotor outer diameter [m]
    magnet_radius: float = 0.070         # radius at which magnets sit [m]
    rotor_mass: float = 0.250            # rotor (hub) mass [kg]
    moment_of_inertia: float | None = None   # if None, computed from geometry

    # --- magnets ---
    n_magnets: int = 8
    magnet_remanence: float = 1.32       # Br [T]  (NdFeB ~1.2-1.4 T)
    magnet_length: float = 0.012         # magnetisation direction length [m]
    magnet_area: float = 1.0e-4          # pole face area [m^2]
    magnet_mass: float = 0.020           # per-magnet mass [kg]

    # --- magnetic coupling ---
    air_gap: float = 0.002               # core-to-magnet face gap [m]
    inductance_swing: float = 0.35       # fractional L rise at alignment
    trigger_turns: int = 200             # sense/trigger winding turns

    # --- trigger timing ---
    fire_phase: float = 0.65             # fraction of the magnet period at which
                                         # the switch turns ON (0 = aligned); the
                                         # motoring/approach region is ~0.55..0.75

    # --- mechanical loss ---
    friction_coeff: float = 1.0e-6       # viscous friction B [N*m*s/rad]
    drag_coeff: float = 0.0              # windage / air drag D [N*m*s^2/rad^2]
    coulomb_friction: float = 0.0        # constant bearing friction T_c [N*m]
    cogging_amplitude: float = 0.0       # detent/cogging torque amplitude [N*m]

    # state
    theta: float = 0.0                   # rotor mechanical angle [rad]
    omega: float = 0.0                   # angular speed [rad/s]

    # ---- derived -----------------------------------------------------------
    def __post_init__(self) -> None:
        if self.moment_of_inertia is None:
            self.moment_of_inertia = self._compute_inertia()

    def _compute_inertia(self) -> float:
        r_hub = self.wheel_diameter / 2.0
        j_hub = 0.5 * self.rotor_mass * r_hub * r_hub          # solid disk approx
        j_mags = self.n_magnets * self.magnet_mass * self.magnet_radius ** 2
        return j_hub + j_mags

    # ---- angular bookkeeping ----------------------------------------------
    @property
    def magnet_step(self) -> float:
        return 2.0 * math.pi / self.n_magnets

    @property
    def rpm(self) -> float:
        return self.omega * 60.0 / (2.0 * math.pi)

    def phase_fraction(self, theta: float) -> float:
        """Position within the current magnet period, psi in [0,1); 0 = aligned."""
        psi = (theta % self.magnet_step) / self.magnet_step
        return psi

    # ---- magnetic coupling profile ----------------------------------------
    def _profile(self, theta: float) -> float:
        """Smooth periodic bump, =1 at alignment (psi=0), ->0 at psi=0.5."""
        psi = self.phase_fraction(theta)
        return 0.5 * (1.0 + math.cos(2.0 * math.pi * psi))

    def inductance(self, base_inductance: float, theta: float, current: float = 0.0) -> float:
        """
        Coil inductance [H] at rotor angle theta, given the coil's current- and
        saturation-dependent base inductance.  The magnet alignment adds the
        `inductance_swing` fraction on top of the base.
        """
        p = self._profile(theta)
        return base_inductance * (1.0 + self.inductance_swing * p)

    def dL_dtheta(self, base_inductance: float, theta: float) -> float:
        """
        dL/dtheta [H/rad], analytic.  With the profile p(psi) = (1+cos 2pi psi)/2
        and dpsi/dtheta = 1/magnet_step,
            dL/dtheta = base * swing * (-pi sin(2 pi psi)) / magnet_step.
        Sign is motoring in the rising-inductance region (psi in (0.5, 1)).
        """
        psi = self.phase_fraction(theta)
        dprofile = -math.pi * math.sin(2.0 * math.pi * psi)
        return base_inductance * self.inductance_swing * dprofile / self.magnet_step

    def dL_dtheta_fd(self, base_inductance: float, theta: float) -> float:
        """Finite-difference dL/dtheta -- kept to cross-check the analytic form."""
        h = 1.0e-5
        lp = self.inductance(base_inductance, theta + h)
        lm = self.inductance(base_inductance, theta - h)
        return (lp - lm) / (2.0 * h)

    def torque(self, base_inductance: float, theta: float, current: float) -> float:
        """Electromagnetic torque [N*m] at (theta, current): T = 1/2 i^2 dL/dtheta."""
        dL = self.dL_dtheta(base_inductance, theta)
        return 0.5 * current * current * dL

    # ---- flux & trigger EMF ------------------------------------------------
    def peak_flux_density_at_core(self) -> float:
        """
        Approximate peak flux density [T] at the core face from a magnet of
        remanence Br across an air gap g.  Uses the short-magnet/gap magnetic-
        circuit form Br * Lm/(Lm+g) softened for fringing.  Order-of-magnitude
        coupling estimate; the dynamics depend mainly on dL/dtheta above.
        """
        lm, g = self.magnet_length, self.air_gap
        b0 = self.magnet_remanence * lm / (lm + g)
        fringing = 1.0 / (1.0 + (g / max(math.sqrt(self.magnet_area), 1e-6)) ** 1.5)
        return b0 * fringing

    def flux_linkage(self, turns: int, theta: float) -> float:
        """Magnetic flux linkage [Wb-turn] of `turns` with the rotor field."""
        b = self.peak_flux_density_at_core() * self._profile(theta)
        return turns * b * self.magnet_area

    def trigger_voltage(self, omega: float, theta: float, turns: int | None = None) -> float:
        """
        Open-circuit EMF [V] induced in the trigger winding by the passing
        magnet: e = -N dPhi/dt = -N (dPhi/dtheta) omega.  This is the signal that
        fires the transistor; its sign/level sets the trigger threshold timing.
        """
        n = self.trigger_turns if turns is None else turns
        h = 1.0e-5
        dphi = (self.flux_linkage(n, theta + h) - self.flux_linkage(n, theta - h)) / (2.0 * h)
        return -n * dphi * omega

    # ---- trigger timing ----------------------------------------------------
    def pulse_period(self) -> float:
        """Seconds between consecutive magnet crossings at the current speed."""
        if self.omega <= 0:
            return float("inf")
        return self.magnet_step / self.omega

    def on_time(self, dwell_fraction: float = 0.30) -> float:
        """Transistor conduction (on) time [s] for a given dwell angle fraction."""
        return dwell_fraction * self.pulse_period()

    # ---- mechanical dynamics ----------------------------------------------
    def mechanical_step(self, torque: float, load_torque: float, dt: float) -> None:
        """
        Semi-implicit (symplectic) Euler step of the mechanical dynamics:
            J domega/dt = T_em - T_load - B*w - D*w*|w| - T_c*sign(w) - T_cog
        Viscous friction (B), quadratic windage (D) and constant Coulomb
        bearing friction (T_c) give a finite, quickly-reached no-load speed.
        """
        j = self.moment_of_inertia
        w = self.omega
        cogging = self.cogging_amplitude * math.sin(self.n_magnets * self.theta)
        drag = self.drag_coeff * w * abs(w)
        coulomb = self.coulomb_friction if w >= 0 else -self.coulomb_friction
        net = torque - load_torque - self.friction_coeff * w - drag - coulomb - cogging
        self.omega += (net / j) * dt
        self.theta = (self.theta + self.omega * dt) % (2.0 * math.pi)
