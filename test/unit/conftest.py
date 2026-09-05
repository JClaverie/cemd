"""Shared fixtures for the AtomicSystem unit test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from cemd import AtomicSystem

DATA_DIR = Path(__file__).parent / "data"

HAS_PACKMOL = shutil.which("packmol") is not None

requires_packmol = pytest.mark.skipif(
    not HAS_PACKMOL, reason="packmol binary not found on PATH"
)


def _water_atoms_df() -> pd.DataFrame:
    """Two rigid, non-overlapping water molecules."""
    return pd.DataFrame(
        {
            "type": ["Ow", "Hw", "Hw", "Ow", "Hw", "Hw"],
            "charge": [-0.8, 0.4, 0.4, -0.8, 0.4, 0.4],
            "x": [0.0, 0.96, -0.24, 10.0, 10.96, 9.76],
            "y": [0.0, 0.0, 0.93, 0.0, 0.0, 0.93],
            "z": [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
        },
        index=range(1, 7),
    )


def make_water_system() -> AtomicSystem:
    """Build a fresh two-molecule water system with bonds and angles."""
    system = AtomicSystem(
        {
            "atoms": _water_atoms_df(),
            "box": [30.0, 30.0, 30.0, 90.0, 90.0, 90.0],
            "masses": {"Ow": 15.9994, "Hw": 1.007947},
            "charges": {"Ow": -0.8, "Hw": 0.4},
        }
    )
    system.add_bond([1, 2])
    system.add_bond([1, 3])
    system.add_bond([4, 5])
    system.add_bond([4, 6])
    system.add_angle([2, 1, 3])
    system.add_angle([5, 4, 6])
    return system


@pytest.fixture
def water_system() -> AtomicSystem:
    """Two water molecules, bonded, with angles; string atom types."""
    return make_water_system()


def make_compact_molecule(box: float = 40.0) -> AtomicSystem:
    """A single small water molecule in a box much larger than its extent.

    Kept deliberately compact relative to the box (unlike the two spread-out
    molecules in :func:`make_water_system`) so that MDAnalysis's periodic
    (Bai & Breen) center-of-mass reduces to the plain weighted mean; that
    algorithm is only expected to match a naive COM when the object's span
    is small relative to the box, not for a cluster spanning a large
    fraction of it.
    """
    atoms = pd.DataFrame(
        {
            "type": ["Ow", "Hw", "Hw"],
            "charge": [-0.8, 0.4, 0.4],
            "x": [2.0, 2.96, 1.76],
            "y": [2.0, 2.0, 2.93],
            "z": [2.0, 2.0, 2.0],
        },
        index=[1, 2, 3],
    )
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": [box, box, box, 90.0, 90.0, 90.0],
            "masses": {"Ow": 15.9994, "Hw": 1.007947},
            "charges": {"Ow": -0.8, "Hw": 0.4},
        }
    )


@pytest.fixture
def compact_molecule() -> AtomicSystem:
    return make_compact_molecule()


@pytest.fixture
def numeric_type_system() -> AtomicSystem:
    """A tiny system using numeric (LAMMPS-style) atom types, no topology."""
    atoms = pd.DataFrame(
        {
            "type": [1, 1, 2],
            "charge": [0.0, 0.0, 0.0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        },
        index=[1, 2, 3],
    )
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {1: 12.011, 2: 15.999},
            "charges": {1: 0.0, 2: 0.0},
        }
    )


@pytest.fixture
def empty_atoms_system() -> AtomicSystem:
    """A system with zero atoms, for edge-case testing."""
    atoms = pd.DataFrame(columns=["type", "charge", "x", "y", "z"])
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {},
            "charges": {},
        }
    )
