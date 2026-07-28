# core/_forcefield/models.py
#
# This file is part of the CEMD distribution
# Copyright (c) 2022-2026 Jérôme Claverie.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


"""
Data classes for force field parameters.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AtomType:
    """Atom type parameters."""
    element: str
    charge: float
    environment: str = ""
    ref: str = ""
    mass: Optional[float] = None
    model: Optional[str] = None


@dataclass
class LJParams:
    """Lennard-Jones 12-6 parameters."""
    epsilon: float  # kcal/mol
    sigma: float    # Å
    ref: str = ""
    model: Optional[str] = None


@dataclass
class BuckinghamParams:
    """Buckingham potential parameters."""
    A: float        # kcal/mol
    rho: float      # Å
    C: float        # kcal/mol·Å⁶
    ref: str = ""
    model: Optional[str] = None


@dataclass
class BondParams:
    """Harmonic bond parameters."""
    k: float        # kcal/(mol·Å²)
    r0: float       # Å
    ref: str = ""
    model: Optional[str] = None


@dataclass
class AngleParams:
    """Harmonic angle parameters."""
    k: float        # kcal/(mol·rad²)
    theta0: float   # degrees
    ref: str = ""
    model: Optional[str] = None


@dataclass
class ForceFieldModel:
    """Force field model metadata."""
    name: str
    description: str = ""
    ref: str = ""
    tags: list[str] = field(default_factory=list)