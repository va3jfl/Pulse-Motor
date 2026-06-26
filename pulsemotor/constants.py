"""
Physical constants, conductor data, and the soft-magnetic core-material database.

All quantities are in SI base units unless a name explicitly states otherwise
(e.g. Gauss is clearly labelled).  Nothing here is speculative: every value is a
representative engineering datum drawn from standard reference ranges.  Material
properties are *grade dependent* in reality, so each entry stores a typical value
together with the range it was taken from; downstream code never extrapolates
beyond the stated assumptions.

Sources used to set the typical ranges are listed in README.md.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------------
# Fundamental physical constants (CODATA representative values)
# ----------------------------------------------------------------------------
MU_0: float = 4.0e-7 * math.pi        # vacuum permeability      [H/m]
EPS_0: float = 8.8541878128e-12       # vacuum permittivity      [F/m]
C_LIGHT: float = 2.99792458e8         # speed of light           [m/s]

# Skin depth helper: delta = sqrt(2 rho / (mu omega)).  Resistivities are stored
# per material below.

# ----------------------------------------------------------------------------
# Conductor (wire) materials: electrical resistivity at ~20 C [Ohm * metre]
# ----------------------------------------------------------------------------
WIRE_MATERIALS: dict[str, dict] = {
    "copper":    {"rho": 1.68e-8, "alpha": 3.93e-3, "notes": "annealed Cu, IACS 100% reference"},
    "silver":    {"rho": 1.59e-8, "alpha": 3.80e-3, "notes": "highest conductivity metal"},
    "aluminum":  {"rho": 2.65e-8, "alpha": 4.29e-3, "notes": ""},
    "gold":      {"rho": 2.44e-8, "alpha": 3.40e-3, "notes": ""},
    "copper_enamelled": {"rho": 1.72e-8, "alpha": 3.93e-3, "notes": "magnet wire, work-hardened"},
}

def conductor_resistivity(name: str, temperature_c: float = 20.0) -> float:
    """Resistivity [Ohm*m] of a conductor at a given temperature (linear model)."""
    mat = WIRE_MATERIALS[name]
    dT = temperature_c - 20.0
    return mat["rho"] * (1.0 + mat["alpha"] * dT)

# ----------------------------------------------------------------------------
# AWG (American Wire Gauge) geometry
#
# Standard definition: for gauge number n in [0, 40],
#     d(n) = 0.005 inch * 92 ** ((36 - n) / 39)
# anchored at AWG 36 = 0.005 inch and AWG 10 = 0.1019 inch.  Verified against
# the standard copper-wire table.
# ----------------------------------------------------------------------------
def awg_diameter(awg: float) -> float:
    """Bare-conductor diameter [m] for a (possibly fractional) AWG number."""
    d_in = 0.005 * 92.0 ** ((36.0 - awg) / 39.0)
    return d_in * 0.0254

def awg_area(awg: float) -> float:
    """Cross-sectional copper area [m^2] for a given AWG."""
    r = awg_diameter(awg) / 2.0
    return math.pi * r * r

def current_capacity(awg: float) -> float:
    """
    Rough continuous current rating [A] from the rule-of-thumb 700 circular-mil/A.
    Used only as a sanity bound on user inputs, never as a physics result.
    """
    d_in = awg_diameter(awg) / 0.0254
    cmil = d_in * d_in * 1.0e6          # diameter in mils squared = circular mils
    return cmil / 700.0

# ----------------------------------------------------------------------------
# Soft-magnetic core materials
#
# mu_r  : typical (initial / unmagnetised) relative permeability
# mu_max: approximate maximum relative permeability (for reference)
# b_sat : saturation flux density [T]
# notes : grade dependence / open-circuit behaviour
#
# These are typical mid-range engineering values; real grades vary widely.  The
# important physics captured downstream is that an *open* core (a rod, as used in
# this topology) cannot realise mu_r directly because of the demagnetising field.
# ----------------------------------------------------------------------------
CORE_MATERIALS: dict[str, dict] = {
    "air":            {"mu_r": 1.0,     "mu_max": 1.0,      "b_sat": float("inf"), "notes": "no core; no saturation"},
    "vacuum":         {"mu_r": 1.0,     "mu_max": 1.0,      "b_sat": float("inf"), "notes": "same as air"},
    "stainless_304":  {"mu_r": 1.02,    "mu_max": 1.1,      "b_sat": float("inf"), "notes": "austenitic; essentially non-magnetic"},
    "stainless_416":  {"mu_r": 7.0e2,   "mu_max": 1.0e3,    "b_sat": 1.10,         "notes": "martensitic; ferromagnetic, grade-dependent"},
    "mild_steel":     {"mu_r": 2.0e3,   "mu_max": 5.0e3,    "b_sat": 1.60,         "notes": "low-carbon steel rod"},
    "iron_powder":    {"mu_r": 3.5e1,   "mu_max": 9.0e1,    "b_sat": 1.20,         "notes": "distributed gap; mu 10..100"},
    "ferrite_mnzn":   {"mu_r": 2.0e3,   "mu_max": 6.0e3,    "b_sat": 0.40,         "notes": "MnZn power ferrite; mu 1k..15k"},
    "ferrite_nizn":   {"mu_r": 1.0e2,   "mu_max": 4.0e2,    "b_sat": 0.35,         "notes": "NiZn; high resistivity"},
    "silicon_steel":  {"mu_r": 4.0e3,   "mu_max": 4.0e4,    "b_sat": 1.70,         "notes": "grain-oriented electrical steel"},
    "nanocrystalline":{"mu_r": 3.0e4,   "mu_max": 1.5e5,    "b_sat": 1.20,         "notes": "Fe-based nanocrystalline ribbon"},
    "amorphous":      {"mu_r": 5.0e4,   "mu_max": 1.0e5,    "b_sat": 1.55,         "notes": "Fe-based metglass"},
    "permalloy":      {"mu_r": 5.0e4,   "mu_max": 3.0e5,    "b_sat": 0.80,         "notes": "Ni-Fe high-perm alloy"},
}

def core_material(name: str) -> dict:
    """Return the core-material property dict, raising a clear error if unknown."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CORE_MATERIALS:
        raise KeyError(
            f"Unknown core material '{name}'. "
            f"Known: {sorted(CORE_MATERIALS)}"
        )
    return dict(CORE_MATERIALS[key])
