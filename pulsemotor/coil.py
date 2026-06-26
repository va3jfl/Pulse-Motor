"""
Coil & core physics engine.

Computes the electromagnetic properties of the stator winding/core assembly from
its geometry and materials:

    * DC resistance from a multi-layer winding-length estimate
    * AC resistance (skin effect) at a given frequency
    * Saturating inductance via a closed-form arctan (Frohlich-type) B-H model,
      with an open-rod demagnetising correction
    * Saturation current I_sat
    * Impedance |Z|(f, i)
    * Estimated self-capacitance (Medhurst) and self-resonant frequency

Two real corrections to the naive "L = mu0 mur N^2 A / l" are modelled, and both
are measurable:

  1. Open-circuit (rod) cores do not realise the material mu_r.  The rod has a
     demagnetising factor N_d that reduces the effective permeability to
     mu_eff = mur / (1 + N d (mur - 1)).
  2. As the core approaches B_sat its incremental permeability collapses toward
     mu0 (vacuum).  The DIFFERENTIAL inductance L_inc = d(lambda)/di therefore
     floors at the air-core value -- it never goes negative.  This is enforced by
     construction via the saturating model below.

Saturation model (n=2 Frohlich / arctan, chosen so every quantity is closed-form
and L_inc >= L_air by construction):
    f(i) = 1 / (1 + (i/I_sat)^2)
    L_inc(i) = L0 * (1 + (mu_eff - 1) f(i))                >= L0 (air core)
    lambda(i) = L0 * (i + (mu_eff - 1) I_sat atan(i/I_sat))   flux linkage
    W (i)     = L0 * (i^2/2 + (mu_eff - 1)(I_sat^2/2) ln(1 + (i/I_sat)^2))   field energy
    W'(i)     = integral lambda ds  (co-energy; drives torque)
where L0 = mu0 N^2 A / l is the air-core inductance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import constants as K


@dataclass
class Coil:
    # --- winding ---
    turns: int                      # N
    awg: float = 20.0               # wire gauge (may be fractional)
    wire_material: str = "copper_enamelled"
    winding_length: float = 0.050   # axial length of the winding [m]
    packing_factor: float = 0.93    # enamelled-wire packing (area fill)

    # --- core ---
    core_material: str = "ferrite_mnzn"
    core_diameter: float = 0.012    # core cross-section diameter [m]
    core_length: float = 0.080      # core magnetic-path length (rod) [m]
    core_closed: bool = False       # True for a closed (toroidal/loop) core

    # --- operating / thermal ---
    temperature_c: float = 25.0

    # ---- cached derived geometry -------------------------------------------
    def __post_init__(self) -> None:
        self._mat = K.core_material(self.core_material)
        self._wire_rho = K.conductor_resistivity(self.wire_material, self.temperature_c)
        self.wire_diameter = K.awg_diameter(self.awg)
        self.wire_area = K.awg_area(self.awg) * self.packing_factor
        self.core_area = math.pi * (self.core_diameter / 2.0) ** 2
        self.material_mu_r = self._mat["mu_r"]
        self.b_sat = self._mat["b_sat"]

    # ------------------------------------------------------------------ core
    def _base_factor(self) -> float:
        """L0 = mu0 N^2 A / l  -- the air-core inductance [H]."""
        return K.MU_0 * (self.turns ** 2) * self.core_area / self.core_length

    def demagnetising_factor(self) -> float:
        """
        N_d for a solid cylinder magnetised along its axis (open core).
        Approximation valid for aspect ratio m = length/diameter >= ~1.
        For a closed core N_d -> 0 by definition.
        """
        if self.core_closed:
            return 0.0
        m = self.core_length / self.core_diameter
        if m <= 1.0:
            return 0.333          # sphere-ish limit
        return (1.0 / (m * m)) * (math.log(2.0 * m) - 1.0)

    def effective_mu_r(self) -> float:
        """Material mu_r as realised by the actual core geometry (open or closed)."""
        nd = self.demagnetising_factor()
        mur = self.material_mu_r
        return mur / (1.0 + nd * (mur - 1.0))

    # ------------------------------------------------------------------ wire
    def mean_length_per_turn_by_layer(self) -> list[float]:
        """Circumference of each winding layer [m] (close-wound, wire_diameter pitch)."""
        diam = self.wire_diameter
        turns_per_layer = max(1, int(self.winding_length / diam))
        n_layers = max(1, math.ceil(self.turns / turns_per_layer))
        return [math.pi * (self.core_diameter + (2 * k + 1) * diam) for k in range(n_layers)]

    def total_wire_length(self) -> float:
        diam = self.wire_diameter
        turns_per_layer = max(1, int(self.winding_length / diam))
        mlts = self.mean_length_per_turn_by_layer()
        total = 0.0
        remaining = self.turns
        for mlt in mlts:
            in_layer = min(turns_per_layer, remaining)
            total += in_layer * mlt
            remaining -= in_layer
            if remaining <= 0:
                break
        return total

    @property
    def resistance_dc(self) -> float:
        """DC winding resistance [Ohm]."""
        return self._wire_rho * self.total_wire_length() / max(self.wire_area, 1e-30)

    def resistance_ac(self, frequency: float) -> float:
        """
        AC resistance [Ohm] at `frequency`, including skin effect in the round
        conductor.  Proximity effect between turns is not modelled; the reported
        value is a lower bound on the true Rac at high frequency.
        """
        rho = self._wire_rho
        omega = 2.0 * math.pi * frequency
        if frequency <= 0:
            return self.resistance_dc
        mu_for_skin = min(self.material_mu_r, 1.0e4)
        delta = math.sqrt(2.0 * rho / (K.MU_0 * mu_for_skin * omega))
        a = self.wire_diameter / 2.0
        if a <= delta:
            return self.resistance_dc
        return self.resistance_dc * (a / (2.0 * delta))

    # ---------------------------------------------------- saturation / inductance
    def saturation_current(self) -> float:
        """
        Current [A] at which the core material reaches B_sat.  Returns +inf for a
        non-saturating (air / non-magnetic) core.
        """
        if not math.isfinite(self.b_sat):
            return float("inf")
        nd = self.demagnetising_factor()
        # ampere-turns to drive the material to B_sat, with the demagnetising field
        mmf_total = (self.b_sat / (K.MU_0 * self.material_mu_r)) * self.core_length \
                    * (1.0 + nd * (self.material_mu_r - 1.0))
        return mmf_total / self.turns

    def differential_inductance(self, current: float = 0.0) -> float:
        """L_inc(i) = d(lambda)/di [H].  Floors at the air-core value, never < 0."""
        i = abs(current)
        i_sat = self.saturation_current()
        mur = self.effective_mu_r()
        if not math.isfinite(i_sat) or i_sat <= 0:
            return self._base_factor() * mur
        f = 1.0 / (1.0 + (i / i_sat) ** 2)
        return self._base_factor() * (1.0 + (mur - 1.0) * f)

    def flux_linkage(self, current: float = 0.0) -> float:
        """lambda(i) = integral_0^i L_inc ds  [Wb-turn]."""
        i = abs(current)
        mur = self.effective_mu_r()
        i_sat = self.saturation_current()
        if i == 0:
            return 0.0
        if not math.isfinite(i_sat) or i_sat <= 0:
            return self._base_factor() * mur * i
        return self._base_factor() * (i + (mur - 1.0) * i_sat * math.atan(i / i_sat))

    def field_energy_base(self, current: float = 0.0) -> float:
        """W(i) = integral_0^i s L_inc(s) ds  [J]  -- magnetic field energy."""
        i = abs(current)
        mur = self.effective_mu_r()
        i_sat = self.saturation_current()
        if not math.isfinite(i_sat) or i_sat <= 0:
            return self._base_factor() * mur * i * i / 2.0
        r2 = (i / i_sat) ** 2
        return self._base_factor() * (i * i / 2.0
                                      + (mur - 1.0) * (i_sat * i_sat / 2.0) * math.log1p(r2))

    def coenergy_base(self, current: float = 0.0) -> float:
        """W'(i) = integral_0^i lambda(s) ds  [J]  -- co-energy (drives torque)."""
        i = abs(current)
        mur = self.effective_mu_r()
        i_sat = self.saturation_current()
        if i == 0:
            return 0.0
        if not math.isfinite(i_sat) or i_sat <= 0:
            return self._base_factor() * mur * i * i / 2.0
        term = i * math.atan(i / i_sat) - (i_sat / 2.0) * math.log1p((i / i_sat) ** 2)
        return self._base_factor() * (i * i / 2.0 + (mur - 1.0) * i_sat * term)

    def inductance(self, current: float = 0.0) -> float:
        """Chord (apparent) inductance lambda(i)/i [H].  Used for display/impedance."""
        i = abs(current)
        if i == 0:
            return self.differential_inductance(0.0)
        return self.flux_linkage(current) / i

    def inductance_unsaturated(self) -> float:
        """Inductance [H] in the linear (i -> 0) regime."""
        return self._base_factor() * self.effective_mu_r()

    # ------------------------------------------------------------- impedance
    def impedance(self, frequency: float, current: float = 0.0) -> complex:
        """Complex impedance [Ohm] = Rac + j w L_chord(i)."""
        r = self.resistance_ac(frequency)
        xl = 2.0 * math.pi * frequency * self.inductance(current)
        return complex(r, xl)

    # -------------------------------------------------- self capacitance / SRF
    def self_capacitance(self) -> float:
        """Estimated winding self-capacitance [F] via the Medhurst approximation."""
        d_cm = self.core_diameter * 100.0
        l_cm = self.winding_length * 100.0
        ratio = l_cm / d_cm if d_cm > 0 else 1.0
        table = [(0.3, 0.47), (0.5, 0.32), (1.0, 0.56), (2.0, 0.75),
                 (3.0, 0.86), (5.0, 0.95), (10.0, 1.0), (20.0, 1.05)]
        h = _interp_log(table, ratio)
        return (h * d_cm) * 1e-12

    def self_resonant_frequency(self, current: float = 0.0) -> float:
        """Self-resonant frequency [Hz] from L_chord(i) and the estimated self-C."""
        l = self.inductance(current)
        c = self.self_capacitance()
        if l <= 0 or c <= 0:
            return float("inf")
        return 1.0 / (2.0 * math.pi * math.sqrt(l * c))

    def resonant_frequency(self, capacitance: float, current: float = 0.0) -> float:
        """Resonant frequency [Hz] with an explicit external capacitance [F]."""
        l = self.inductance(current)
        if l <= 0 or capacitance <= 0:
            return float("inf")
        return 1.0 / (2.0 * math.pi * math.sqrt(l * capacitance))

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        return {
            "turns": self.turns,
            "awg": self.awg,
            "wire_diameter_mm": self.wire_diameter * 1e3,
            "total_wire_length_m": self.total_wire_length(),
            "resistance_dc_ohm": self.resistance_dc,
            "core_material": self.core_material,
            "material_mu_r": self.material_mu_r,
            "effective_mu_r": self.effective_mu_r(),
            "demag_factor_Nd": self.demagnetising_factor(),
            "inductance_unsat_mH": self.inductance_unsaturated() * 1e3,
            "inductance_at_I_sat_mH": self.inductance(self.saturation_current()) * 1e3,
            "saturation_current_A": self.saturation_current(),
            "self_capacitance_pF": self.self_capacitance() * 1e12,
            "self_resonant_freq_kHz": self.self_resonant_frequency() / 1e3,
            "core_area_mm2": self.core_area * 1e6,
            "core_path_length_mm": self.core_length * 1e3,
        }


def _interp_log(table: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation in (x, y) pairs; clamps outside the range."""
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return table[-1][1]
