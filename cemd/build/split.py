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

from ..core._format import lattice2vectors, vectors2lattice
from .base import BaseBuilder

if TYPE_CHECKING:
    from ..core.atomic_system import AtomicSystem
    from .solution import SolutionBuilder


@dataclass
class Splitter(BaseBuilder):
    """
    Split a system along an axis and optionally add a solution.

    Examples
    --------
    >>> # Simple split
    >>> result = Splitter(system, axis='z', gap_size=20.0).split()

    >>> # Split with solution (fluent API)
    >>> result = (Splitter(system, axis='z', gap_size=30.0)
    ...           .add_solution(blueprint, padding=2.0)
    ...           .split())

    >>> # Or with the convenience function
    >>> result = split(system, axis='z', gap_size=30.0,
    ...                solution=blueprint, padding=2.0)
    """

    system: AtomicSystem
    coordinate: float
    axis: int | str = 2
    gap_size: float = 20.0

    # Solution options (default: no solution)
    _solution_blueprint: SolutionBuilder | None = field(default=None, repr=False)
    _padding: float = field(default=2.0, repr=False)
    _vacuum: float = field(default=0.0, repr=False)
    _has_solution: bool = field(default=False, repr=False)

    # Internal state (non-serializable)
    _unit_norm: np.ndarray | None = field(default=None, repr=False, init=False)
    _vec_list: list | None = field(default=None, repr=False, init=False)
    _axis_name: str | None = field(default=None, repr=False, init=False)

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
        """Insert the solution into the gap."""
        from .interface import _calculate_liquid_box, _merge_structure

        liquid_thickness = self._calculate_solution_thickness()

        # Calculate the box for the liquid
        liquid_box = _calculate_liquid_box(
            self.system, liquid_thickness, self._axis_name
        )

        # Build the liquid
        liquid = self._solution_blueprint.build(liquid_box)

        # Center the liquid in the gap
        gap_center = self.coordinate + self.gap_size / 2.0
        liq_pos = liquid.atoms[self._axis_name].values
        liq_center = (liq_pos.max() + liq_pos.min()) / 2.0
        liquid.atoms[self._axis_name] += gap_center - liq_center

        # Merge everything
        result = _merge_structure(
            self.system,
            liquid,
            distance=0.0,
            axis=self._axis_name,
            vacuum=self._vacuum,
        )

        # Replace the system with the merged result
        self.system.atoms = result.atoms
        self.system.set_box(result.box)
        if hasattr(result, "bonds"):
            self.system.bonds = result.bonds

    def split(self) -> AtomicSystem:
        """
        Execute the split and return the resulting AtomicSystem.

        Returns
        -------
        AtomicSystem
            The split system, optionally with solution.
        """
        original_positions = self.system.atoms[["x", "y", "z"]].copy()

        universe = self.system.to_mda()
        self._move_fragments(universe)
        self.system.atoms[["x", "y", "z"]] = universe.atoms.positions

        self._remove_crossing_bonds(original_positions)

        self._update_box()

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
        return self.split()
