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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..core._format import lattice2vectors, vectors2lattice
from .base import BaseBuilder

if TYPE_CHECKING:
    from ..core.atomic_system import AtomicSystem
    from .solution import SolutionBuilder


# Interatomic distances (Å) below which a contact is considered a real,
# structure-bearing bond that a cut would sever. Deliberately restricted to
# short, essentially covalent contacts: ionic coordination shells (Ca...Ow,
# Na...Cl) are *not* listed, since a cut passing through them is physically
# fine and must not be counted as damage. Keys can be atom types (checked
# first) or elements (fallback) -- see `Splitter.find_broken_bonds`.
DEFAULT_BOND_CUTOFFS: dict[tuple[str, str], float] = {
    ("Si", "O"): 1.8,
    ("Al", "O"): 1.8,
    ("C", "O"): 1.6,
    ("S", "O"): 1.7,
    ("H", "O"): 1.1,
}


@dataclass
class Splitter(BaseBuilder):
    """
    Split a system along an axis and optionally add a solution.

    A cut can sever real interatomic bonds. Bonds that exist explicitly in
    the system's topology are dropped automatically. Contacts that are
    structure-bearing but carry no explicit bond -- the Si-O framework
    under ClayFF/CSHFF, for instance -- are found geometrically instead,
    from ``bonds_dict``; use :meth:`count_broken_bonds` or
    :meth:`scan_broken_bonds` to choose where to cut, and ``repair=True``
    to cap the dangling atoms left behind.

    Parameters
    ----------
    system : AtomicSystem
        The system to split.
    coordinate : float
        Position along ``axis`` at which to cut, as a projection on the
        axis unit vector.
    axis : int or str, default=2
        Axis along which to split ('x'/'y'/'z' or 0/1/2).
    gap_size : float, default=20.0
        Width of the gap to open, in Å.
    bonds_dict : dict, optional
        ``{(type_or_element, type_or_element): cutoff}`` used to detect
        severed contacts. Defaults to :data:`DEFAULT_BOND_CUTOFFS`.

    Examples
    --------
    >>> # Simple split: cut at z=15, opening a 20 A gap there
    >>> result = Splitter(system, coordinate=15.0, axis='z', gap_size=20.0).split()

    >>> # Split with solution (fluent API)
    >>> result = (Splitter(system, coordinate=15.0, axis='z', gap_size=30.0)
    ...           .add_solution(blueprint, padding=2.0)
    ...           .split())

    >>> # Look for the least damaging cut, then split and cap what breaks
    >>> splitter = Splitter(system, coordinate=0.0, axis='z', gap_size=20.0)
    >>> scan = splitter.scan_broken_bonds(step=0.25)
    >>> best = scan.loc[scan["n_broken"].idxmin(), "coordinate"]
    >>> result = splitter.with_coordinate(best).split(repair=True)
    """

    system: AtomicSystem
    coordinate: float
    axis: int | str = 2
    gap_size: float = 20.0
    bonds_dict: dict[tuple[str, str], float] | None = None

    # Solution options (default: no solution)
    _solution_blueprint: SolutionBuilder | None = field(default=None, repr=False)
    _padding: float = field(default=2.0, repr=False)
    _vacuum: float = field(default=0.0, repr=False)
    _has_solution: bool = field(default=False, repr=False)

    # Internal state (non-serializable)
    _unit_norm: np.ndarray | None = field(default=None, repr=False, init=False)
    _vec_list: list | None = field(default=None, repr=False, init=False)
    _axis_name: str | None = field(default=None, repr=False, init=False)

    # Filled in by `split(repair=True)`
    repair_report: dict[str, int] | None = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Initialize derived attributes."""
        # Convert axis if string
        if isinstance(self.axis, str):
            axis_map = {"x": 0, "y": 1, "z": 2}
            self.axis = axis_map[self.axis.lower()]

        self._axis_name = ["x", "y", "z"][self.axis]

        # Compute axis information
        vec_list = list(lattice2vectors(self.system.box))
        target_vec = vec_list[self.axis]
        self._unit_norm = target_vec / np.linalg.norm(target_vec)
        self._vec_list = vec_list

    def add_solution(
        self, blueprint: SolutionBuilder, padding: float = 2.0, vacuum: float = 0.0
    ) -> Splitter:
        """
        Add a liquid solution in the gap.

        Parameters
        ----------
        blueprint : SolutionBuilder
            The solution blueprint defining composition.
        padding : float, default=2.0
            Empty space between solution and surfaces (Å).
        vacuum : float, default=0.0
            Additional vacuum above the solution (Å).

        Returns
        -------
        Splitter
            Self for method chaining.
        """
        self._solution_blueprint = blueprint
        self._padding = padding
        self._vacuum = vacuum
        self._has_solution = True
        return self

    def with_coordinate(self, coordinate: float) -> Splitter:
        """
        Set the cutting position.

        Parameters
        ----------
        coordinate : float
            Position along the axis where to cut.

        Returns
        -------
        Splitter
            Self for method chaining.
        """
        self.coordinate = coordinate
        return self

    # ------------------------------------------------------------------
    # Severed-contact detection
    # ------------------------------------------------------------------

    def _resolve_cutoffs(self, bonds_dict: dict | None) -> dict:
        """Return the cutoff table to use, falling back to the defaults."""
        if bonds_dict is not None:
            return bonds_dict
        if self.bonds_dict is not None:
            return self.bonds_dict
        return DEFAULT_BOND_CUTOFFS

    @staticmethod
    def _lookup_cutoff(
        cutoffs: dict, type_1: str, type_2: str, element_1: str, element_2: str
    ) -> float | None:
        """Find the cutoff for a pair: atom types first, elements as fallback.

        Types take priority so that a system already carrying force-field
        types can distinguish structure-bearing oxygens (``Ob``, ``Osih``)
        from ones that merely sit nearby (``Ow``), which elements alone
        cannot express.
        """
        for key in ((type_1, type_2), (type_2, type_1)):
            if key in cutoffs:
                return cutoffs[key]

        for key in ((element_1, element_2), (element_2, element_1)):
            if key in cutoffs:
                return cutoffs[key]

        return None

    def find_broken_bonds(
        self,
        coordinate: float | None = None,
        bonds_dict: dict | None = None,
    ) -> list[tuple[int, int]]:
        """Find the contacts a cut would sever, from the current geometry.

        Unlike :meth:`_remove_crossing_bonds`, this looks at interatomic
        *distances* rather than the explicit bond table, so it also finds
        structure-bearing contacts that carry no explicit bond (the Si-O
        framework under ClayFF, for instance).

        Parameters
        ----------
        coordinate : float, optional
            Cut position to test. Defaults to this splitter's own
            ``coordinate``.
        bonds_dict : dict, optional
            Cutoff table, see :data:`DEFAULT_BOND_CUTOFFS`.

        Returns
        -------
        list of tuple of int
            Pairs of atom ids whose contact the cut would break.

        Notes
        -----
        Call this *before* splitting: it reports on the geometry as it
        currently stands. Contacts that are neighbours only through the
        periodic boundary are excluded -- the split shifts the box and the
        moving fragment by the same amount, so those survive intact.
        """
        from MDAnalysis.lib.distances import self_capped_distance

        coordinate = self.coordinate if coordinate is None else coordinate
        cutoffs = self._resolve_cutoffs(bonds_dict)

        atoms = self.system.atoms
        if not cutoffs or len(atoms) < 2:
            return []

        positions = atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
        ids = atoms.index.to_numpy()
        types = atoms["type"].astype(str).to_numpy()

        elements = self.system.elements
        atom_elements = [elements.get(t, t) for t in types]

        pairs, distances = self_capped_distance(
            positions.astype(np.float32),
            max_cutoff=max(cutoffs.values()),
            box=np.asarray(self.system.box, dtype=np.float32),
        )

        projections = positions @ self._unit_norm
        moves = projections >= coordinate

        broken = []
        for (i, j), distance in zip(pairs, distances):
            # Both atoms on the same side travel together: nothing breaks.
            if moves[i] == moves[j]:
                continue

            cutoff = self._lookup_cutoff(
                cutoffs, types[i], types[j], atom_elements[i], atom_elements[j]
            )
            if cutoff is None or distance > cutoff:
                continue

            # Neighbours only through the periodic boundary keep their
            # separation once the box grows with the gap.
            if np.linalg.norm(positions[j] - positions[i]) > distance + 1e-3:
                continue

            broken.append((int(ids[i]), int(ids[j])))

        return broken

    def count_broken_bonds(
        self,
        coordinate: float | None = None,
        bonds_dict: dict | None = None,
    ) -> int:
        """Count the contacts a cut at ``coordinate`` would sever.

        See :meth:`find_broken_bonds` for the detection rules.
        """
        return len(self.find_broken_bonds(coordinate, bonds_dict))

    def scan_broken_bonds(
        self,
        start: float | None = None,
        stop: float | None = None,
        step: float = 0.5,
        bonds_dict: dict | None = None,
    ) -> pd.DataFrame:
        """Count severed contacts across a range of cut positions.

        Useful to pick a cut plane that damages the structure as little as
        possible before committing to it.

        Parameters
        ----------
        start, stop : float, optional
            Range of cut coordinates to scan. Defaults to the span of the
            system along the split axis.
        step : float, default=0.5
            Scanning increment in Å.
        bonds_dict : dict, optional
            Cutoff table, see :data:`DEFAULT_BOND_CUTOFFS`.

        Returns
        -------
        pandas.DataFrame
            Columns ``coordinate`` and ``n_broken``, one row per position.
        """
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")

        positions = self.system.atoms[["x", "y", "z"]].to_numpy(dtype=np.float64)
        projections = positions @ self._unit_norm

        start = float(projections.min()) if start is None else start
        stop = float(projections.max()) if stop is None else stop

        coordinates = np.arange(start, stop + step / 2, step)

        return pd.DataFrame(
            {
                "coordinate": coordinates,
                "n_broken": [
                    self.count_broken_bonds(c, bonds_dict) for c in coordinates
                ],
            }
        )

    def _move_fragments(self, universe) -> None:
        """Move the second fragment to create the gap."""
        pos = universe.atoms.positions
        mask = np.dot(pos, self._unit_norm) >= self.coordinate
        shift_vector = self._unit_norm * self.gap_size
        pos[mask] += shift_vector
        universe.atoms.positions = pos

    def _remove_crossing_bonds(self, original_positions) -> None:
        """Remove bonds stretched across the newly created gap.

        A bond whose two atoms ended up on opposite sides of the cut gets
        physically severed by the split: its length grows by roughly
        ``gap_size``. Compare each bond's length before/after the move to
        detect this, then drop the bond and any angle/dihedral/improper
        built on it (they'd otherwise reference a pair of atoms that are
        no longer meaningfully connected).
        """
        bonds = self.system.bonds
        if bonds is None or bonds.empty:
            return

        current_positions = self.system.atoms[["x", "y", "z"]]
        broken_pairs = set()

        for _, row in bonds.iterrows():
            a1, a2 = int(row["atom_1"]), int(row["atom_2"])
            d_before = np.linalg.norm(
                original_positions.loc[a1].to_numpy()
                - original_positions.loc[a2].to_numpy()
            )
            d_after = np.linalg.norm(
                current_positions.loc[a1].to_numpy()
                - current_positions.loc[a2].to_numpy()
            )
            if d_after > d_before + self.gap_size / 2:
                broken_pairs.add(frozenset((a1, a2)))

        if not broken_pairs:
            return

        def _references_broken_pair(row, n_atoms) -> bool:
            atom_cols = [f"atom_{i}" for i in range(1, n_atoms + 1)]
            indices = [int(row[c]) for c in atom_cols]
            return any(
                frozenset((indices[i], indices[i + 1])) in broken_pairs
                for i in range(len(indices) - 1)
            )

        for name, n_atoms in [
            ("bonds", 2),
            ("angles", 3),
            ("dihedrals", 4),
            ("impropers", 4),
        ]:
            df = getattr(self.system, name)
            if df is None or df.empty:
                continue
            cleaned = df[~df.apply(_references_broken_pair, axis=1, args=(n_atoms,))]
            setattr(self.system, name, cleaned if not cleaned.empty else None)

    def _update_box(self) -> None:
        """Update the simulation box after splitting."""
        self._vec_list[self.axis] = (
            self._vec_list[self.axis] + self._unit_norm * self.gap_size
        )
        new_box = vectors2lattice(tuple(self._vec_list))
        self.system.set_box(new_box)

    def _calculate_solution_thickness(self) -> float:
        """Calculate the optimal thickness for the liquid."""
        thickness = self.gap_size - 2 * self._padding - self._vacuum
        if thickness <= 0:
            raise ValueError(
                f"Gap size ({self.gap_size:.1f} Å) too small for "
                f"padding ({self._padding:.1f} Å) and vacuum ({self._vacuum:.1f} Å). "
                f"Available space: {thickness:.1f} Å"
            )
        return thickness

    def _insert_solution(self) -> None:
        """Fill the gap opened by the split with the solution.

        The box is already the final one here (``_update_box`` has grown it
        by ``gap_size``), so the liquid is merged *into* it rather than
        stacked onto the system: ``_merge_structure`` -- which is what the
        surface builder uses to lay a film on top of a slab -- would have
        translated the liquid above the topmost atom and stretched the cell
        a second time, leaving the pore empty and the fluid outside the
        solid.
        """
        from .interface import _calculate_liquid_box, _merge_data

        liquid_thickness = self._calculate_solution_thickness()

        # Calculate the box for the liquid
        liquid_box = _calculate_liquid_box(
            self.system, liquid_thickness, self._axis_name
        )

        # Build the liquid
        liquid = self._solution_blueprint.build(liquid_box)

        # Packmol packs the liquid in its own cell starting at the origin;
        # the transverse dimensions match the host cell, so the liquid only
        # has to be moved onto the host's own origin.
        bounds = self.system._box_lmp
        for index, name in enumerate("xyz"):
            if index == self.axis:
                continue
            liquid.atoms[name] += float(bounds[index][0])

        # Center the liquid in the gap
        gap_center = self.coordinate + self.gap_size / 2.0
        liq_pos = liquid.atoms[self._axis_name].values
        liq_center = (liq_pos.max() + liq_pos.min()) / 2.0
        liquid.atoms[self._axis_name] += gap_center - liq_center

        # Merge everything, keeping the box set by `_update_box`
        result = _merge_data(self.system, liquid, self.system._box_lmp)

        # Replace the system with the merged result
        self.system._replace_internals(result)

    def _collect_dangling(self, bonds_dict: dict | None) -> list[tuple[int, np.ndarray, float]]:
        """Record which atoms lose a partner, and in which direction.

        Must run *before* the fragments move: the direction toward the
        lost partner is taken from the intact geometry (a translation
        leaves it unchanged, so it stays valid afterwards).

        Returns
        -------
        list of (atom id, unit vector toward the lost partner, bond length)
        """
        positions = self.system.atoms[["x", "y", "z"]]
        dangling = []

        for id_1, id_2 in self.find_broken_bonds(bonds_dict=bonds_dict):
            r_1 = positions.loc[id_1].to_numpy(dtype=float)
            r_2 = positions.loc[id_2].to_numpy(dtype=float)

            delta = r_2 - r_1
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue

            direction = delta / length
            dangling.append((id_1, direction, length))
            dangling.append((id_2, -direction, length))

        return dangling

    def _repair_dangling(
        self,
        dangling: list[tuple[int, np.ndarray, float]],
        oh_length: float,
        min_distance: float = 0.5,
    ) -> dict[str, int]:
        """Cap atoms left under-coordinated by the cut.

        An exposed oxygen is protonated into a hydroxyl; any other exposed
        atom (Si, Al, C, S, ...) receives a hydroxyl group in the direction
        its partner used to occupy. The added atoms are placed collinearly
        with the broken contact and carry a zero charge -- this is a
        starting geometry meant to be relaxed, and to be re-typed with
        `set_topology()` / `set_ff_from_database()` afterwards.

        A cap that would land on top of an existing atom is skipped rather
        than created: a bridging oxygen that loses *both* its cations would
        otherwise have each of them restore a copy of it at the very same
        site. Skipped caps are reported back so the caller knows some
        atoms were left under-coordinated.
        """
        elements = self.system.elements
        occupied = list(self.system.atoms[["x", "y", "z"]].to_numpy(dtype=float))
        report = {"capped": 0, "skipped": 0}

        def place(atom_type: str, position: np.ndarray) -> bool:
            if any(np.linalg.norm(position - other) < min_distance for other in occupied):
                report["skipped"] += 1
                return False

            self.system.add_atom(atom_type, position)
            occupied.append(position)
            report["capped"] += 1
            return True

        for atom_id, direction, length in dangling:
            atom = self.system.atoms.loc[atom_id]
            element = elements.get(str(atom["type"]), str(atom["type"]))

            # An exposed hydrogen means its oxygen went the other way; the
            # oxygen side gets capped on its own, so leave this one alone.
            if element == "H":
                continue

            position = atom[["x", "y", "z"]].to_numpy(dtype=float)

            if element == "O":
                place("H", position + direction * oh_length)
                continue

            # Cation side: restore the missing oxygen, then cap it.
            oxygen_position = position + direction * length
            if place("O", oxygen_position):
                place("H", oxygen_position + direction * oh_length)

        return report

    def split(
        self,
        repair: bool = False,
        bonds_dict: dict | None = None,
        oh_length: float = 1.0,
    ) -> AtomicSystem:
        """
        Execute the split and return the resulting AtomicSystem.

        Parameters
        ----------
        repair : bool, default=False
            Cap the atoms left under-coordinated by the cut (see
            :meth:`find_broken_bonds` for how they are detected).
        bonds_dict : dict, optional
            Cutoff table used for that detection, see
            :data:`DEFAULT_BOND_CUTOFFS`.
        oh_length : float, default=1.0
            O-H distance in Å used when capping.

        Returns
        -------
        AtomicSystem
            The split system, optionally repaired and filled with solution.

        Notes
        -----
        Repair adds neutral H (and O) atoms: re-run ``set_topology()`` and
        ``set_ff_from_database()`` afterwards so the new atoms get proper
        types and charges. After a repair, ``repair_report`` summarises how
        many caps were added and how many were skipped as overlapping.
        """
        original_positions = self.system.atoms[["x", "y", "z"]].copy()

        # Detection has to happen on the intact geometry.
        dangling = self._collect_dangling(bonds_dict) if repair else []

        universe = self.system.to_mda()
        self._move_fragments(universe)
        self.system.atoms[["x", "y", "z"]] = universe.atoms.positions

        self._remove_crossing_bonds(original_positions)

        self._update_box()

        if repair:
            self.repair_report = {
                "broken": len(dangling) // 2,
                **self._repair_dangling(dangling, oh_length),
            }

        if self._has_solution:
            self._insert_solution()

        if hasattr(self.system, "metadata"):
            self.system.metadata["split_info"] = {
                "axis": self.axis,
                "axis_name": self._axis_name,
                "coordinate": self.coordinate,
                "gap_size": self.gap_size,
                "has_solution": self._has_solution,
                "solution_padding": self._padding if self._has_solution else None,
                "solution_thickness": self._calculate_solution_thickness()
                if self._has_solution
                else None,
                "solution_vacuum": self._vacuum if self._has_solution else None,
            }

        return self.system

    def build(self, **kwargs) -> AtomicSystem:
        """Alias for :meth:`split`, to satisfy the ``BaseBuilder`` interface."""
        return self.split(**kwargs)
