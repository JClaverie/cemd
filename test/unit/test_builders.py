"""Tests for AtomicSystem's builder-backed methods: add_structure,
add_liquid_layer and add_droplet.

`add_liquid_layer`/`add_droplet` shell out to Packmol, so those two are
skipped when the binary isn't on PATH (see `requires_packmol` in conftest).
`add_structure` needs no external solver and always runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cemd import AtomicSystem
from cemd.build import SolutionBuilder

from conftest import requires_packmol


def _make_solid(box=(12.0, 12.0, 10.0, 90.0, 90.0, 90.0)):
    atoms = pd.DataFrame(
        {
            "type": ["Na", "Cl"],
            "charge": [1.0, -1.0],
            "x": [0.0, box[0] / 2.0],
            "y": [0.0, box[1] / 2.0],
            "z": [1.0, 1.0],
        },
        index=[1, 2],
    )
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": list(box),
            "masses": {"Na": 22.989769282, "Cl": 35.4532},
            "charges": {},
        }
    )


# ---------------------------------------------------------------------------
# add_structure (no external solver required)
# ---------------------------------------------------------------------------


def test_add_structure_merges_atoms_and_returns_self():
    solid = _make_solid()
    molecule = AtomicSystem(
        {
            "atoms": pd.DataFrame(
                {
                    "type": ["Ow", "Hw", "Hw"],
                    "charge": [-0.8, 0.4, 0.4],
                    "x": [0.0, 0.96, -0.24],
                    "y": [0.0, 0.0, 0.93],
                    "z": [0.0, 0.0, 0.0],
                },
                index=[1, 2, 3],
            ),
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"Ow": 15.9994, "Hw": 1.007947},
            "charges": {},
        }
    )

    result = solid.add_structure(molecule, distance=2.0, vacuum=5.0)

    assert result is solid
    assert solid.num_atoms == 5
    assert set(solid.atom_types) == {"Na", "Cl", "Ow", "Hw"}


def test_add_structure_from_file_path(tmp_path):
    solid = _make_solid()
    molecule = AtomicSystem(
        {
            "atoms": pd.DataFrame(
                {
                    "type": ["Ow"],
                    "charge": [0.0],
                    "x": [0.0],
                    "y": [0.0],
                    "z": [0.0],
                },
                index=[1],
            ),
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"Ow": 15.9994},
            "charges": {},
        }
    )
    path = tmp_path / "molecule.data"
    molecule.write(path)

    solid.add_structure(str(path), distance=2.0, vacuum=5.0)
    assert solid.num_atoms == 3


# ---------------------------------------------------------------------------
# add_liquid_layer / add_droplet (require the packmol binary)
# ---------------------------------------------------------------------------


@requires_packmol
def test_add_liquid_layer_adds_water_and_returns_self():
    solid = _make_solid()
    n0 = solid.num_atoms
    blueprint = SolutionBuilder.from_water(density=1.0)

    result = solid.add_liquid_layer(blueprint, thickness=6.0, distance=2.0, vacuum=5.0)

    assert result is solid
    assert solid.num_atoms > n0
    assert "Na" in solid.atom_types and "Cl" in solid.atom_types


@requires_packmol
def test_add_droplet_adds_water_and_returns_self():
    solid = _make_solid(box=(20.0, 20.0, 10.0, 90.0, 90.0, 90.0))
    n0 = solid.num_atoms
    blueprint = SolutionBuilder.from_water(density=1.0)

    result = solid.add_droplet(blueprint, radius=5.0, distance=2.0, vacuum=5.0)

    assert result is solid
    assert solid.num_atoms > n0
    # Droplet water goes through SolutionBuilder.build_hemisphere(); make
    # sure its Hw/Ow naming fix (see test_build_solution.py) actually
    # reaches this end-to-end path too.
    assert "Ow" in solid.atom_types and "Hw" in solid.atom_types


# ---------------------------------------------------------------------------
# _add_vacuum (private helper, no external solver needed)
# ---------------------------------------------------------------------------


def test_add_vacuum_extends_the_chosen_axis():
    from cemd.build.interface import _add_vacuum

    solid = _make_solid()
    box_before = solid.box.copy()

    result = _add_vacuum(solid, thickness=10.0, axis="z")

    assert result is solid
    assert solid.box[2] == pytest.approx(box_before[2] + 10.0)
    # Other axes untouched.
    assert solid.box[0] == pytest.approx(box_before[0])
    assert solid.box[1] == pytest.approx(box_before[1])


def test_add_vacuum_invalid_axis_raises():
    from cemd.build.interface import _add_vacuum

    with pytest.raises(KeyError):
        _add_vacuum(_make_solid(), thickness=10.0, axis="q")
