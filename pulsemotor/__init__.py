"""
pulsemotor -- Switched-reluctance pulse-motor simulation (Phase 1: physics core).

Public objects:
    Coil      -- stator winding + core electromagnetics
    Rotor     -- rotating magnet assembly + variable-reluctance coupling
    Circuit   -- drive / flyback (kickback) transient solver
"""
from .coil import Coil
from .rotor import Rotor
from .circuit import Circuit, PulseResult
from . import constants

__all__ = ["Coil", "Rotor", "Circuit", "PulseResult", "constants"]
__version__ = "0.1.0"
