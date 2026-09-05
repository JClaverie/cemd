"""Tests for cemd.build.SurfaceBuilder, using the real calcite.cif fixture."""

from __future__ import annotations

import pytest

from cemd import AtomicSystem
from cemd.build import SurfaceBuilder

from conftest import DATA_DIR


@pytest.fixture(scope="module")
def calcite() -> AtomicSystem:
    return AtomicSystem.from_file(DATA_DIR / "calcite.cif")


def test_surface_builder_repr(calcite):
    builder = SurfaceBuilder(calcite)
    text = repr(builder)
    assert "SurfaceBuilder" in text
    assert "sites   : 30" in text


def test_build_generates_slabs_with_vacuum(calcite):
    builder = SurfaceBuilder(calcite)
    slabs, shifts, dipoles, broken = builder.build(
        miller_indices=[1, 0, 4], min_slab_size=10.0, min_vacuum_size=10.0
    )

    assert len(slabs) >= 1
    assert len(slabs) == len(shifts) == len(dipoles)
    assert all(isinstance(s, AtomicSystem) for s in slabs)
    assert slabs[0].num_atoms > calcite.num_atoms
    assert broken >= 0

    # `build()` also stashes its own return value on `.result`.
    assert builder.result == (slabs, shifts, dipoles, broken)


def test_build_rejects_structure_with_too_few_atoms():
    atoms_df = calcite_single_atom_df()
    tiny = AtomicSystem(
        {
            "atoms": atoms_df,
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"Ca": 40.078},
            "charges": {},
        }
    )
    builder = SurfaceBuilder(tiny)
    with pytest.raises(ValueError, match="at least 2 atoms"):
        builder.build(miller_indices=[1, 0, 0])


def calcite_single_atom_df():
    import pandas as pd

    return pd.DataFrame(
        {"type": ["Ca"], "charge": [0.0], "x": [0.0], "y": [0.0], "z": [0.0]},
        index=[1],
    )
