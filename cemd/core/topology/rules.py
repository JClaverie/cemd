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


@dataclass
class NeighborCriterion:
    """
    Defines a neighbor requirement for a topology rule.

    Parameters
    ----------
    selection : str
        MDAnalysis selection string for neighbor atoms.
    cutoff : float
        Maximum distance from center atom in Å.
    count : int
        Exact number of neighbors required.
    new_type : str, optional
        New type to assign to matching neighbors.

    Examples
    --------
    >>> NeighborCriterion("type Si", 1.85, 2)
    >>> NeighborCriterion("type H", 1.2, 2, "Hw")
    """

    selection: str
    cutoff: float
    count: int
    new_type: str | None = None

    def __post_init__(self):
        """Validate the criterion."""
        if not self.selection:
            raise ValueError("selection cannot be empty")
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {self.cutoff}")
        if self.count < 0:
            raise ValueError(f"count must be non-negative, got {self.count}")

    def to_dict(self) -> dict:
        """Convert to internal dict format."""
        return {
            "sel": self.selection,
            "cutoff": self.cutoff,
            "n": self.count,
            "new_type": self.new_type,
        }

    def __repr__(self):
        return f"NeighborCriterion({self.selection!r}, cutoff={self.cutoff}, n={self.count})"


@dataclass
class TopologyRule:
    """
    Defines a geometry-based connectivity rule.

    Parameters
    ----------
    center : str
        MDAnalysis selection for the central atom.
    neighbors : list of NeighborCriterion
        Neighbor requirements.
    new_type : str, optional
        New type for the central atom if the rule matches.
    bonds : bool, default=False
        Create bonds between center and neighbors.
    angles : bool, default=False
        Create angles between neighbor pairs.
    impropers : bool, default=False
        Create impropers (center as central atom).

    Examples
    --------
    >>> # Bridging oxygen (Si-O-Si)
    >>> rule = TopologyRule(
    ...     center="type O",
    ...     neighbors=[NeighborCriterion("type Si", 1.85, 2)],
    ...     new_type="Ob"
    ... )
    >>>
    >>> # Water molecule
    >>> water = TopologyRule(
    ...     center="type O",
    ...     neighbors=[NeighborCriterion("type H", 1.2, 2, "Hw")],
    ...     new_type="Ow",
    ...     bonds=True,
    ...     angles=True,
    ... )
    """

    center: str
    neighbors: list[NeighborCriterion] | NeighborCriterion = field(default_factory=list)
    new_type: str | None = None
    bonds: bool = False
    angles: bool = False
    impropers: bool = False

    def __post_init__(self):
        """Validate the rule."""
        if not self.center:
            raise ValueError("center cannot be empty")
        if isinstance(self.neighbors, NeighborCriterion):
            self.neighbors = [self.neighbors]
        elif not isinstance(self.neighbors, list):
            raise TypeError(
                f"neighbors must be a NeighborCriterion or list of NeighborCriterion, "
                f"got {type(self.neighbors)}"
            )

        for n in self.neighbors:
            if not isinstance(n, NeighborCriterion):
                raise TypeError(f"Expected NeighborCriterion, got {type(n)}")

    def add_neighbor(
        self, selection: str, cutoff: float, count: int, new_type: str | None = None
    ) -> TopologyRule:
        """
        Add a neighbor criterion — chainable.

        Returns
        -------
        TopologyRule
            Self for method chaining.
        """
        self.neighbors.append(NeighborCriterion(selection, cutoff, count, new_type))
        return self

    def to_dict(self) -> dict:
        """Convert to internal dict format."""
        return {
            "center_sel": self.center,
            "new_type": self.new_type,
            "neighbors": [n.to_dict() for n in self.neighbors],
            "create_bond": self.bonds,
            "create_angle": self.angles,
            "create_improper": self.impropers,
        }

    def __repr__(self):
        return (
            f"TopologyRule({self.center!r} → {self.new_type!r}, "
            f"{len(self.neighbors)} neighbor(s))"
        )


@dataclass
class DihedralRule:
    """Defines an explicit proper dihedral pattern."""

    i: str  # selection atome i
    j: str  # selection atome j
    k: str  # selection atome k
    l: str  # selection atome l
    cutoffs: list[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
