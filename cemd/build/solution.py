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

import os
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .._constants import AVOGADRO, MASSES_DICT
from .._utils import require_program
from ._packmol import add_packmol_structure, get_structure_path, run_packmol

if TYPE_CHECKING:
    from ..core.atomic_system import AtomicSystem

WATER_MOLAR_MASS: float = 2 * MASSES_DICT["H"] + MASSES_DICT["O"]


def concentration2count(molarity: float | int, volume: float) -> tuple[int, float]:
    """Calculates the integer particle count and relative error for a single molarity.

    Parameters
    ----------
    molarity : float | int
        Target concentration in mol/L (M).
    volume : float
        Volume of the simulation box in cubic Angstroms (Å³).

    Returns
    -------
    tuple[int, float]
        -Particle count (integer).
        -Relative discretization error percentage (float).
    """
    if molarity == 0:
        return 0, 0.0

    theoretical_n = molarity * AVOGADRO * volume * 1e-27
    final_count = int(round(theoretical_n))

    if theoretical_n > 0:
        error_pct = abs(final_count - theoretical_n) / theoretical_n
    else:
        error_pct = 0.0

    return final_count, error_pct


@dataclass
class SolutionBuilder:
    """
    Blueprint for creating a solution.

    This blueprint defines WHAT the solution contains (composition),
    not HOW it is built (geometry). The geometry is determined by
    the builder or method that uses this blueprint.

    Parameters
    ----------
    density : float, default=1.0
        Target density in g/cm³
    molarities : dict, optional
        Species molarities in mol/L
    counts : dict, optional
        Species explicit counts (number of molecules/ions)
    structures : dict, optional
        Custom AtomicSystem structures for complex molecules

    Examples
    --------
    >>> # Define a salt solution
    >>> blueprint = SolutionBuilder(
    ...     density=1.0,
    ...     molarities={'NaCl': 0.1, 'KCl': 0.05}
    ... )

    >>> # Use in different contexts
    >>> solution = blueprint.build(box=[30, 30, 30])  # Standalone
    >>> system = surface.add_layer(blueprint, thickness=30.0)  # Layer
    >>> system = surface.add_droplet(blueprint, radius=15.0)  # Droplet

    >>> # Pure water
    >>> water = SolutionBuilder.from_water()

    >>> # With explicit counts
    >>> blueprint = SolutionBuilder.from_counts(
    ...     density=1.0,
    ...     counts={'Na': 50, 'Cl': 50}
    ... )
    """

    density: float = 1.0
    molarities: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    structures: dict[str, AtomicSystem] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the blueprint."""
        self._validate()

    def __repr__(self) -> str:
        """
        Return a styled string representation of the SolutionBuilder.

        Returns
        -------
        str
            Summary of the solution composition and density.
        """
        lines = ["<SolutionBuilder>"]
        lines.append("")

        # ====== Composition ======
        lines.append("┌─ Composition")
        lines.append(f"│   density:  {self.density:.2f} g/cm³")

        # Solutes
        if self.molarities:
            lines.append("│   type:     molarities")
            for species, molarity in self.molarities.items():
                lines.append(f"│             {species}: {molarity:.3f} M")
        elif self.counts:
            lines.append("│   type:     counts")
            for species, count in self.counts.items():
                lines.append(f"│             {species}: {count}")
        else:
            lines.append("│   type:     pure water")

        # ====== Custom structures ======
        if self.structures:
            lines.append("│")
            lines.append("├─ Custom structures")
            for species, struct in self.structures.items():
                lines.append(f"│   {species}: {struct.num_atoms} atoms")

        return "\n".join(lines)

    def _validate(self):
        """Check that the blueprint is valid."""
        if self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")

        # Check that you don't have both at the same time
        if self.molarities and self.counts:
            raise ValueError(
                "Provide either 'molarities' OR 'counts', not both.\n"
                f"molarities: {list(self.molarities.keys())}\n"
                f"counts: {list(self.counts.keys())}"
            )

        # Validate the molarities
        for species, value in self.molarities.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Molarity for '{species}' must be a number, "
                    f"got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"Molarity must be >= 0 for '{species}'")

        # Validate accounts
        for species, value in self.counts.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"Count for '{species}' must be an integer, "
                    f"got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"Count must be >= 0 for '{species}'")

        # Validate custom structures
        for species, struct in self.structures.items():
            from ..core.atomic_system import AtomicSystem

            if not isinstance(struct, AtomicSystem):
                raise TypeError(
                    f"Structure for '{species}' must be an AtomicSystem, "
                    f"got {type(self.structures[species]).__name__}"
                )

    def to_counts(self, volume: float) -> dict[str, int]:
        """
        Convert molarities to explicit counts for a given volume.

        Parameters
        ----------
        volume : float
            Volume in Å³

        Returns
        -------
        dict[str, int]
            Dictionary mapping species to integer counts
        """
        # If we already have counts, return them
        if self.counts:
            return self.counts.copy()

        # If we have molarities, convert them
        if self.molarities:
            counts = {}
            errors = {}

            for species, molarity in self.molarities.items():
                count, error = concentration2count(molarity, volume)
                counts[species] = count
                errors[species] = error

                if error > 0.1:  # 10% error threshold
                    warnings.warn(
                        f"Species '{species}' has {error:.1%} discretization error. "
                        f"Using {counts[species]} molecules (target: {molarity} M)."
                    )

            return counts

        # Pure water
        return {}

    def get_solute_mass(self, volume: float) -> float:
        """
        Calculate total mass of solutes in atomic mass units for a given volume.

        Parameters
        ----------
        volume : float
            Volume in Å³

        Returns
        -------
        float
            Total solute mass in amu

        Raises
        ------
        ValueError
            If a species is not found in the mass database.
        """
        counts = self.to_counts(volume)
        total_mass = 0.0

        for species, count in counts.items():
            if species in self.structures:
                total_mass += count * self.structures[species].total_mass
            elif species in MASSES_DICT:
                total_mass += count * MASSES_DICT[species]
            else:
                raise ValueError(
                    f"Species '{species}' not found in mass database "
                    f"and no custom structure provided.\n"
                    f"Available: {list(MASSES_DICT.keys())}"
                )

        return total_mass

    def get_water_count(self, volume: float) -> int:
        """
        Calculate number of water molecules for a given volume.

        Parameters
        ----------
        volume : float
            Volume in Å³

        Returns
        -------
        int
            Number of water molecules

        Raises
        ------
        ValueError
            If the density or volume is too low for the solutes.
        """
        mass_total_target_g = self.density * volume * 1e-24
        mass_solutes_g = self.get_solute_mass(volume) / AVOGADRO
        mass_water_g = mass_total_target_g - mass_solutes_g

        if mass_water_g <= 0:
            raise ValueError(
                f"Target density ({self.density} g/cm³) or volume ({volume:.1f} Å³) "
                f"is too low to hold the solutes.\n"
                f"Solute mass: {mass_solutes_g:.4f} g\n"
                f"Target total mass: {mass_total_target_g:.4f} g"
            )

        return int(round((mass_water_g * AVOGADRO) / WATER_MOLAR_MASS))

    def build(self, box: Sequence[float], margin: float = 0.95) -> AtomicSystem:
        """
        Build a standalone solution system from this blueprint.

        Parameters
        ----------
        box : Sequence[float]
            Box dimensions [a, b, c] in Å
        margin : float, default=0.95
        Safety scaling factor to prevent atoms from sitting directly on box edges.

        Returns
        -------
        AtomicSystem
            The built solution system

        Examples
        --------
        >>> blueprint = SolutionBuilder(
        ...     density=1.0,
        ...     molarities={'NaCl': 0.1}
        ... )
        >>> solution = blueprint.build(box=[30, 30, 30])
        """
        require_program("packmol")

        boxa, boxb, boxc = box[:3]
        volume = boxa * boxb * boxc

        # Obtain solute counts
        solute_counts = self.to_counts(volume)

        # Calculate the number of water molecules
        num_water = self.get_water_count(volume)

        box_constraint = f"inside box 0 0 0 {boxa * margin:.4f} {boxb * margin:.4f} {boxc * margin:.4f}"

        # Build with Packmol
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            structures = []

            # Add the water
            h2o_path = get_structure_path("H2O", tmp)
            structures.append(
                add_packmol_structure(
                    h2o_path,
                    num_water,
                    f"inside box 0 0 0 {boxa * 0.95:.4f} {boxb * 0.95:.4f} {boxc * 0.95:.4f}",
                )
            )

            # Add the solutes
            for species, count in solute_counts.items():
                if count <= 0:
                    continue

                # Check if a custom structure is provided
                if species in self.structures:
                    struct_path = os.path.join(tmp, f"custom_{species}.pdb")
                    self.structures[species].write(struct_path)
                else:
                    struct_path = get_structure_path(species, tmp)

                structures.append(
                    add_packmol_structure(
                        struct_path,
                        count,
                        "center",
                        f"inside box 0 0 0 {boxa * 0.95:.4f} {boxb * 0.95:.4f} {boxc * 0.95:.4f}",
                    )
                )

            data = run_packmol(structures)

        # Set box
        box_angles = [90.0, 90.0, 90.0]
        final_box = list(box[:3]) + box_angles if len(box) == 3 else list(box)
        data.set_box(final_box)

        # Store metadata
        data._solution_metadata = {
            "density": self.density,
            "solutes": solute_counts,
            "num_water": num_water,
            "volume": volume,
            "blueprint": self,
        }

        return data
