"""Tests for cemd.build.Splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cemd import AtomicSystem
from cemd.build import Splitter, SolutionBuilder

from conftest import requires_packmol


def _make_chain_system() -> AtomicSystem:
    """Four atoms in a chain along z: bonds 1-2 and 3-4 stay on one side
    of coordinate=5, bond 2-3 crosses it."""
    atoms = pd.DataFrame(
        {
            "type": ["X", "X", "X", "X"],
            "charge": [0.0] * 4,
            "x": [0.0] * 4,
            "y": [0.0] * 4,
            "z": [1.0, 4.0, 6.0, 9.0],
        },
        index=range(1, 5),
    )
    system = AtomicSystem(
        {
            "atoms": atoms,
            "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"X": 1.0},
            "charges": {},
        }
    )
    system.add_bond([1, 2])
    system.add_bond([2, 3])
    system.add_bond([3, 4])
    return system


def test_split_opens_gap_and_grows_box():
    system = _make_chain_system()
    result = Splitter(system, coordinate=5.0, axis="z", gap_size=10.0).split()

    assert result is system
    assert result.box[2] == pytest.approx(20.0)

    # Atoms below the cut are untouched; atoms above are shifted by gap_size.
    assert result.atoms.loc[1, "z"] == pytest.approx(1.0)
    assert result.atoms.loc[2, "z"] == pytest.approx(4.0)
    assert result.atoms.loc[3, "z"] == pytest.approx(16.0)
    assert result.atoms.loc[4, "z"] == pytest.approx(19.0)


def test_split_removes_bond_crossing_the_gap():
    system = _make_chain_system()
    result = Splitter(system, coordinate=5.0, axis="z", gap_size=10.0).split()

    bond_pairs = {
        frozenset((int(row.atom_1), int(row.atom_2)))
        for row in result.bonds.itertuples()
    }
    assert frozenset((2, 3)) not in bond_pairs
    assert frozenset((1, 2)) in bond_pairs
    assert frozenset((3, 4)) in bond_pairs


def test_axis_accepts_string_or_int():
    a = Splitter(_make_chain_system(), coordinate=5.0, axis="z", gap_size=10.0)
    b = Splitter(_make_chain_system(), coordinate=5.0, axis=2, gap_size=10.0)
    assert a.axis == b.axis == 2


def test_with_coordinate_sets_cut_position():
    splitter = Splitter(_make_chain_system(), coordinate=0.0, axis="z", gap_size=10.0)
    splitter.with_coordinate(5.0)
    assert splitter.coordinate == 5.0


def test_build_is_an_alias_for_split():
    system = _make_chain_system()
    result = Splitter(system, coordinate=5.0, axis="z", gap_size=10.0).build()
    assert result.box[2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# add_solution / split (requires packmol)
# ---------------------------------------------------------------------------


@requires_packmol
def test_split_with_solution_fills_the_gap():
    atoms = pd.DataFrame(
        {
            "type": ["Na", "Cl"],
            "charge": [1.0, -1.0],
            "x": [0.0, 6.0],
            "y": [0.0, 6.0],
            "z": [1.0, 1.0],
        },
        index=[1, 2],
    )
    system = AtomicSystem(
        {
            "atoms": atoms,
            "box": [12.0, 12.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"Na": 22.989769282, "Cl": 35.4532},
            "charges": {},
        }
    )
    blueprint = SolutionBuilder.from_water(density=1.0)

    result = (
        Splitter(system, coordinate=5.0, axis="z", gap_size=20.0)
        .add_solution(blueprint, padding=2.0, vacuum=0.0)
        .split()
    )

    assert result.num_atoms > 2
    assert "Ow" in result.atom_types
    assert "Na" in result.atom_types and "Cl" in result.atom_types


@requires_packmol
def test_solution_is_placed_inside_the_pore_not_stacked_on_top():
    # Regression test: `_insert_solution` centred the liquid in the gap and
    # then handed it to `_merge_structure`, the surface-builder helper that
    # lays a film *on top* of a slab -- which translated it above the
    # topmost atom and stretched the cell a second time. The pore came out
    # empty, the fluid sat outside the solid, and the box was far taller
    # than `original + gap_size`.
    atoms = pd.DataFrame(
        {
            "type": ["Na", "Cl"],
            "charge": [1.0, -1.0],
            "x": [0.0, 6.0],
            "y": [0.0, 6.0],
            "z": [1.0, 1.0],
        },
        index=[1, 2],
    )
    system = AtomicSystem(
        {
            "atoms": atoms,
            "box": [12.0, 12.0, 10.0, 90.0, 90.0, 90.0],
            "masses": {"Na": 22.989769282, "Cl": 35.4532},
            "charges": {},
        }
    )
    blueprint = SolutionBuilder.from_water(density=1.0)

    result = (
        Splitter(system, coordinate=5.0, axis="z", gap_size=20.0)
        .add_solution(blueprint, padding=2.0, vacuum=0.0)
        .split()
    )

    # The box grows by exactly one gap, not by the gap plus a liquid slab.
    assert result.box[2] == pytest.approx(30.0)

    # And the water sits within the gap, [5, 25] in the new coordinates.
    water_z = result.atoms.loc[result.atoms["type"] == "Ow", "z"]
    assert water_z.min() >= 5.0
    assert water_z.max() <= 25.0


@requires_packmol
def test_split_with_solution_keeps_the_angles_of_the_solution():
    # The merge used to copy only atoms/bonds back onto the system, so the
    # solution's angles (and dihedrals, ff keys and charges) were dropped.
    system = _make_chain_system()
    blueprint = SolutionBuilder.from_water(density=1.0)

    result = (
        Splitter(system, coordinate=5.0, axis="z", gap_size=20.0)
        .add_solution(blueprint, padding=2.0, vacuum=0.0)
        .split()
    )

    n_water = (result.atoms["type"] == "Ow").sum()
    assert n_water > 0
    assert result.angles is not None
    assert len(result.angles) == n_water


@requires_packmol
def test_split_with_solution_gap_too_small_raises():
    system = _make_chain_system()
    blueprint = SolutionBuilder.from_water(density=1.0)

    splitter = Splitter(
        system, coordinate=5.0, axis="z", gap_size=3.0
    ).add_solution(blueprint, padding=2.0, vacuum=0.0)

    with pytest.raises(ValueError, match="too small"):
        splitter.split()


# ---------------------------------------------------------------------------
# Severed-contact detection (distance-based, no explicit bonds needed)
# ---------------------------------------------------------------------------


def _make_layered_system() -> AtomicSystem:
    """Two silicate 'layers' (vertical Si-O pairs at z=0 and z=12) with a
    water interlayer at z=7. No explicit bonds anywhere -- the Si-O
    framework carries none under ClayFF/CSHFF conventions.
    """
    types: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for layer_z in (0.0, 12.0):
        for k in range(3):
            types += ["Si", "O"]
            xs += [k * 4.0, k * 4.0]
            ys += [0.0, 0.0]
            zs += [layer_z, layer_z + 1.6]

    for k in range(2):
        types += ["Ow", "Hw", "Hw"]
        xs += [k * 4.0, k * 4.0 + 0.96, k * 4.0 - 0.24]
        ys += [0.0, 0.0, 0.93]
        zs += [7.0, 7.0, 7.0]

    atoms = pd.DataFrame(
        {"type": types, "charge": [0.0] * len(types), "x": xs, "y": ys, "z": zs},
        index=range(1, len(types) + 1),
    )
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": [20.0, 20.0, 24.0, 90.0, 90.0, 90.0],
            "masses": {"Si": 28.085, "O": 15.999, "Ow": 15.999, "Hw": 1.008},
            "charges": {},
        }
    )


def test_detects_contacts_that_carry_no_explicit_bond():
    system = _make_layered_system()
    assert system.bonds is None  # nothing explicit to rely on

    splitter = Splitter(system, coordinate=1.0, axis="z", gap_size=15.0)
    assert splitter.count_broken_bonds() == 3


def test_find_broken_bonds_returns_the_severed_pairs():
    system = _make_layered_system()
    splitter = Splitter(system, coordinate=1.0, axis="z", gap_size=15.0)

    broken = splitter.find_broken_bonds()
    assert len(broken) == 3
    for id_1, id_2 in broken:
        elements = {
            system.elements[str(system.atoms.loc[i, "type"])] for i in (id_1, id_2)
        }
        assert elements == {"Si", "O"}


def test_cut_through_the_interlayer_breaks_nothing():
    system = _make_layered_system()
    splitter = Splitter(system, coordinate=5.0, axis="z", gap_size=15.0)
    assert splitter.count_broken_bonds() == 0


def test_scan_broken_bonds_separates_layers_from_interlayer():
    system = _make_layered_system()
    splitter = Splitter(system, coordinate=0.0, axis="z", gap_size=15.0)

    scan = splitter.scan_broken_bonds(step=1.0)

    assert list(scan.columns) == ["coordinate", "n_broken"]
    assert scan["n_broken"].max() == 3
    # The best cut sits in the interlayer, away from both silicate layers.
    best = scan.loc[scan["n_broken"] == 0, "coordinate"]
    assert ((best > 2.0) & (best < 12.0)).any()


def test_scan_broken_bonds_rejects_non_positive_step():
    splitter = Splitter(_make_layered_system(), coordinate=1.0, axis="z")
    with pytest.raises(ValueError, match="step must be positive"):
        splitter.scan_broken_bonds(step=0.0)


def test_type_keys_take_priority_over_element_keys():
    """A type-keyed dict can exclude contacts that element keys would catch."""
    atoms = pd.DataFrame(
        {
            "type": ["Si", "O", "Ow", "Hw"],
            "charge": [0.0] * 4,
            "x": [0.0, 0.0, 8.0, 8.0],
            "y": [0.0] * 4,
            "z": [0.4, 2.0, 0.6, 1.56],
        },
        index=[1, 2, 3, 4],
    )
    payload = {
        "atoms": atoms,
        "box": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
        "masses": {"Si": 28.085, "O": 15.999, "Ow": 15.999, "Hw": 1.008},
        "charges": {},
    }

    # Default table keys on elements, so the water O-H is caught too.
    default = Splitter(AtomicSystem(payload), coordinate=1.0, axis="z")
    assert sorted(default.find_broken_bonds()) == [(1, 2), (3, 4)]

    # Keying on types lets the water float free, as it should.
    framework_only = Splitter(
        AtomicSystem(payload),
        coordinate=1.0,
        axis="z",
        bonds_dict={("Si", "O"): 1.8},
    )
    assert framework_only.find_broken_bonds() == [(1, 2)]


def test_contacts_through_the_periodic_boundary_are_not_broken():
    # Si and O are 1.6 A apart *through* the z boundary; the split shifts
    # the box and the moving fragment together, so the contact survives.
    atoms = pd.DataFrame(
        {
            "type": ["O", "Si"],
            "charge": [0.0, 0.0],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.8, 19.2],
        },
        index=[1, 2],
    )
    system = AtomicSystem(
        {
            "atoms": atoms,
            "box": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            "masses": {"Si": 28.085, "O": 15.999},
            "charges": {},
        }
    )
    splitter = Splitter(system, coordinate=10.0, axis="z", gap_size=10.0)
    assert splitter.count_broken_bonds() == 0


# ---------------------------------------------------------------------------
# Automatic repair
# ---------------------------------------------------------------------------


def _make_broken_pair_system() -> AtomicSystem:
    """A single vertical Si-O contact, severed by a cut at z=1.0."""
    atoms = pd.DataFrame(
        {
            "type": ["Si", "O"],
            "charge": [0.0, 0.0],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.4, 2.0],
        },
        index=[1, 2],
    )
    return AtomicSystem(
        {
            "atoms": atoms,
            "box": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            "masses": {"Si": 28.085, "O": 15.999},
            "charges": {},
        }
    )


def test_split_without_repair_leaves_dangling_atoms():
    system = _make_broken_pair_system()
    Splitter(system, coordinate=1.0, axis="z", gap_size=10.0).split()
    assert system.num_atoms == 2


def test_repair_caps_both_sides_of_a_severed_contact():
    system = _make_broken_pair_system()
    result = Splitter(system, coordinate=1.0, axis="z", gap_size=10.0).split(
        repair=True
    )

    # Si keeps a hydroxyl (new O + H); the displaced O is capped with an H.
    assert result.num_atoms == 5
    assert result.get_count("H") == 2
    assert result.get_count("O") == 2
    assert result.get_count("Si") == 1


def test_repair_places_caps_along_the_broken_direction():
    system = _make_broken_pair_system()
    result = Splitter(system, coordinate=1.0, axis="z", gap_size=10.0).split(
        repair=True, oh_length=1.0
    )

    zs = sorted(result.atoms["z"].tolist())
    # Si stays at 0.4 with its restored O at 2.0 and H at 3.0; the original
    # O moved to 12.0 and is capped by an H at 11.0.
    assert zs == pytest.approx([0.4, 2.0, 3.0, 11.0, 12.0])


def test_repair_respects_a_restricted_bonds_dict():
    system = _make_broken_pair_system()
    # No cutoff covers Si-O here, so nothing is detected and nothing capped.
    result = Splitter(system, coordinate=1.0, axis="z", gap_size=10.0).split(
        repair=True, bonds_dict={("Ow", "Hw"): 1.1}
    )
    assert result.num_atoms == 2


def test_build_forwards_repair_to_split():
    system = _make_broken_pair_system()
    result = Splitter(system, coordinate=1.0, axis="z", gap_size=10.0).build(
        repair=True
    )
    assert result.num_atoms == 5


def test_repair_reports_what_it_did():
    system = _make_broken_pair_system()
    splitter = Splitter(system, coordinate=1.0, axis="z", gap_size=10.0)
    splitter.split(repair=True)

    assert splitter.repair_report == {"broken": 1, "capped": 3, "skipped": 0}


def test_repair_does_not_stack_atoms_on_a_shared_bridging_oxygen():
    """A bridging O that loses both its cations must not have each of them
    restore a copy of it at the very same site."""
    atoms = pd.DataFrame(
        {
            "type": ["Si", "Si", "O"],
            "charge": [0.0] * 3,
            "x": [-1.0, 1.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.6, 0.6, 1.8],
        },
        index=[1, 2, 3],
    )
    system = AtomicSystem(
        {
            "atoms": atoms,
            "box": [20.0, 20.0, 20.0, 90.0, 90.0, 90.0],
            "masses": {"Si": 28.085, "O": 15.999},
            "charges": {},
        }
    )

    splitter = Splitter(system, coordinate=1.2, axis="z", gap_size=10.0)
    assert splitter.count_broken_bonds() == 2

    result = splitter.split(repair=True)

    assert splitter.repair_report["broken"] == 2
    assert splitter.repair_report["skipped"] >= 1

    # No two atoms may end up sitting on top of each other.
    positions = result.atoms[["x", "y", "z"]].to_numpy(dtype=float)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert np.linalg.norm(positions[i] - positions[j]) > 0.5
