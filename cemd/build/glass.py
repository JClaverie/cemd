# cemd/builders/glass.py

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._constants import AVOGADRO, MASSES_DICT
from .._utils import require_program
from ..core.atomic_system import AtomicSystem
from ._packmol import add_packmol_structure, get_structure_path, run_packmol

if TYPE_CHECKING:
    pass


@dataclass
class GlassBuilder:
    """
    Blueprint for creating a glass structure.

    This blueprint defines WHAT the glass contains (composition),
    not HOW it is built (geometry). The geometry is determined by
    the builder or method that uses this blueprint.

    Parameters
    ----------
    density : float
        Target density in g/cm³.
    composition : dict[str, float]
        Dictionary mapping formula strings to stoichiometric coefficients.
        Can be elements (e.g., 'Si', 'Al'), oxides (e.g., 'SiO2', 'Al2O3'),
        or molecules (e.g., 'H2O').
    structures : dict, optional
        Custom AtomicSystem structures for complex molecules.

    Examples
    --------
    >>> # Define a glass by elements
    >>> blueprint = GlassBuilder(
    ...     density=2.3,
    ...     composition={'Si': 3, 'Al': 4, 'Na': 4, 'O': 14}
    ... )
    >>> glass = blueprint.build(box=[20, 20, 20])
    >>>
    >>> # Define a glass by oxides
    >>> blueprint = GlassBuilder(
    ...     density=2.3,
    ...     composition={'SiO2': 3, 'Al2O3': 2, 'Na2O': 2}
    ... )
    >>> glass = blueprint.build(box=[20, 20, 20])
    >>>
    >>> # With water
    >>> blueprint = GlassBuilder(
    ...     density=2.3,
    ...     composition={'SiO2': 3, 'Al2O3': 2, 'Na2O': 2, 'H2O': 6}
    ... )
    """

    density: float = field(default=None)
    composition: dict[str, float] = field(default_factory=dict)
    structures: dict[str, AtomicSystem] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the blueprint."""
        self._validate()

    def __repr__(self) -> str:
        """
        Return a styled string representation of the GlassBuilder.

        Returns
        -------
        str
            Summary of the glass composition and density.
        """
        lines = ["<GlassBuilder>"]
        lines.append("")

        # ====== Composition ======
        lines.append("┌─ Composition")
        lines.append(f"│   density:  {self.density:.2f} g/cm³")

        for formula, coeff in self.composition.items():
            lines.append(f"│             {formula}: {coeff:.3f}")

        # ====== Structures personnalisées ======
        if self.structures:
            lines.append("│")
            lines.append("├─ Custom structures")
            for species, struct in self.structures.items():
                lines.append(f"│   {species}: {struct.num_atoms} atoms")

        return "\n".join(lines)

    def _validate(self):
        """Check that the blueprint is valid."""
        if self.density is None or self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")

        if not self.composition:
            raise ValueError("composition must not be empty")

        # Valider les structures personnalisées
        for species in self.structures:
            from ..core.atomic_system import AtomicSystem

            if not isinstance(self.structures[species], AtomicSystem):
                raise TypeError(
                    f"Structure for '{species}' must be an AtomicSystem, "
                    f"got {type(self.structures[species]).__name__}"
                )

    # =========================================================================
    # COMPOSITION ANALYSIS
    # =========================================================================

    def get_mass_per_formula_unit(self) -> float:
        """
        Calculate mass per formula unit.

        A formula unit is defined by the stoichiometric coefficients
        in the composition dictionary.

        Returns
        -------
        float
            Mass per formula unit in amu.

        Raises
        ------
        ValueError
            If a species is not found in the mass database and no custom
            structure is provided.
        """
        total_mass = 0.0

        for formula, coeff in self.composition.items():
            # Vérifier si une structure personnalisée est fournie
            if formula in self.structures:
                mass = self.structures[formula].total_mass
            else:
                # Essayer de décomposer la formule avec pymatgen
                try:
                    from pymatgen.core import Composition

                    comp = Composition(formula)
                    mass = sum(
                        MASSES_DICT.get(el, 0) * amt
                        for el, amt in comp.get_el_amt_dict().items()
                    )
                except Exception:
                    # Si pymatgen échoue, essayer comme élément simple
                    if formula in MASSES_DICT:
                        mass = MASSES_DICT[formula]
                    else:
                        raise ValueError(
                            f"Formula '{formula}' not found in mass database "
                            f"and no custom structure provided.\n"
                            f"Available: {list(MASSES_DICT.keys())}"
                        )

            total_mass += mass * coeff

        return total_mass

    def get_elemental_composition(self) -> dict[str, float]:
        """
        Get elemental composition from the blueprint.

        Returns
        -------
        dict[str, float]
            Dictionary mapping element symbols to amounts.
        """
        from pymatgen.core import Composition

        elemental = {}

        for formula, coeff in self.composition.items():
            if formula in self.structures:
                # Custom structure: compter les éléments
                struct = self.structures[formula]
                for elem in struct.elements:
                    elemental[elem] = elemental.get(elem, 0) + coeff
            else:
                try:
                    comp = Composition(formula)
                    for el, amt in comp.get_el_amt_dict().items():
                        elemental[el] = elemental.get(el, 0) + amt * coeff
                except Exception:
                    # Si c'est un élément simple
                    if formula in MASSES_DICT:
                        elemental[formula] = elemental.get(formula, 0) + coeff

        return elemental

    def get_num_formula_units(self, volume: float) -> int:
        """
        Calculate number of formula units for a given volume.

        Parameters
        ----------
        volume : float
            Volume in Å³

        Returns
        -------
        int
            Number of formula units.
        """
        mass_per_unit = self.get_mass_per_formula_unit()
        num_units = int(
            np.round((self.density * volume * 1e-24 * AVOGADRO) / mass_per_unit)
        )
        return max(1, num_units)

    def get_actual_density(self, volume: float) -> float:
        """
        Calculate actual density for a given volume.

        Parameters
        ----------
        volume : float
            Volume in Å³

        Returns
        -------
        float
            Actual density in g/cm³.
        """
        num_units = self.get_num_formula_units(volume)
        mass_per_unit = self.get_mass_per_formula_unit()
        return (num_units * mass_per_unit) / (AVOGADRO * volume * 1e-24)

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================

    def build(self, box: Sequence[float]) -> AtomicSystem:
        """
        Build a standalone glass system from this blueprint.

        Parameters
        ----------
        box : Sequence[float]
            Box dimensions [a, b, c] in Å

        Returns
        -------
        AtomicSystem
            The built glass system.

        Examples
        --------
        >>> blueprint = GlassBuilder(
        ...     density=2.3,
        ...     composition={'SiO2': 3, 'Al2O3': 2, 'Na2O': 2}
        ... )
        >>> glass = blueprint.build(box=[20, 20, 20])
        """
        require_program("packmol")

        boxa, boxb, boxc = box[:3]
        volume = boxa * boxb * boxc

        # Calculer le nombre d'unités
        num_units = self.get_num_formula_units(volume)

        # Calculer la densité réelle
        actual_density = self.get_actual_density(volume)
        density_error = (actual_density - self.density) / self.density

        if abs(density_error) > 0.05:
            warnings.warn(
                f"Density error: {density_error:.1%}. "
                f"Target: {self.density:.3f} g/cm³, Actual: {actual_density:.3f} g/cm³"
            )

        # Obtenir la composition élémentaire
        elemental_comp = self.get_elemental_composition()

        with tempfile.TemporaryDirectory(dir=".") as tmp:
            structures = []

            # Ajouter les éléments
            for elem, amount in elemental_comp.items():
                count = int(np.round(amount * num_units))
                if count <= 0:
                    continue

                path2structure = get_structure_path(elem, tmp)
                structures.append(
                    add_packmol_structure(
                        path2structure,
                        count,
                        "center",
                        f"inside box 1 1 1 {boxa - 1:.4f} {boxb - 1:.4f} {boxc - 1:.4f}",
                    )
                )

            data = run_packmol(structures)

        # Définir la boîte
        if not isinstance(box, list):
            box = list(box)
        data.set_box(box + [90, 90, 90])

        # Stocker les métadonnées
        data._glass_metadata = {
            "density": self.density,
            "actual_density": actual_density,
            "composition": self.composition,
            "num_units": num_units,
            "volume": volume,
            "blueprint": self,
        }

        return data

    # =========================================================================
    # MÉTHODES DE CRÉATION
    # =========================================================================

    @classmethod
    def from_stoichiometry(
        cls,
        density: float,
        composition: dict[str, float],
        structures: dict[str, AtomicSystem] | None = None,
    ) -> GlassBuilder:
        """
        Create a blueprint from stoichiometric coefficients.

        Parameters
        ----------
        density : float
            Target density in g/cm³.
        composition : dict
            Dictionary mapping formula strings to stoichiometric coefficients.
        structures : dict, optional
            Custom structures for species.

        Returns
        -------
        GlassBuilder
            The blueprint.

        Examples
        --------
        >>> # By elements
        >>> blueprint = GlassBuilder.from_stoichiometry(
        ...     density=2.3,
        ...     composition={'Si': 3, 'Al': 4, 'Na': 4, 'O': 14}
        ... )
        >>>
        >>> # By oxides
        >>> blueprint = GlassBuilder.from_stoichiometry(
        ...     density=2.3,
        ...     composition={'SiO2': 3, 'Al2O3': 2, 'Na2O': 2}
        ... )
        """
        return cls(
            density=density, composition=composition, structures=structures or {}
        )

    @classmethod
    def from_oxides(
        cls,
        density: float,
        oxides: dict[str, float],
        structures: dict[str, AtomicSystem] | None = None,
    ) -> GlassBuilder:
        """
        Create a blueprint from oxide composition (alias for from_stoichiometry).

        Parameters
        ----------
        density : float
            Target density in g/cm³.
        oxides : dict
            Dictionary mapping oxide formulas to stoichiometric coefficients.
        structures : dict, optional
            Custom structures for species.

        Returns
        -------
        GlassBuilder
            The blueprint.
        """
        return cls.from_stoichiometry(density, oxides, structures)

    @classmethod
    def from_elements(
        cls,
        density: float,
        elements: dict[str, float],
        structures: dict[str, AtomicSystem] | None = None,
    ) -> GlassBuilder:
        """
        Create a blueprint from elemental composition.

        Parameters
        ----------
        density : float
            Target density in g/cm³.
        elements : dict
            Dictionary mapping element symbols to stoichiometric coefficients.
        structures : dict, optional
            Custom structures for species.

        Returns
        -------
        GlassBuilder
            The blueprint.

        Examples
        --------
        >>> blueprint = GlassBuilder.from_elements(
        ...     density=2.3,
        ...     elements={'Si': 3, 'Al': 4, 'Na': 4, 'O': 14}
        ... )
        """
        return cls.from_stoichiometry(density, elements, structures)

    # =========================================================================
    # MÉTHODES UTILITAIRES
    # =========================================================================

    def is_pure(self) -> bool:
        """Check if the blueprint contains only one component."""
        return len(self.composition) == 1

    def get_components(self) -> list[str]:
        """Get list of components in the composition."""
        return list(self.composition.keys())

    def get_coefficient(self, formula: str) -> float:
        """Get the stoichiometric coefficient for a specific formula."""
        return self.composition.get(formula, 0.0)

    def normalize(self) -> GlassBuilder:
        """
        Normalize the composition so that the sum of coefficients equals 1.

        Returns
        -------
        GlassBuilder
            A new builder with normalized composition.
        """
        total = sum(self.composition.values())
        if total == 0:
            raise ValueError("Cannot normalize empty composition")

        normalized = {k: v / total for k, v in self.composition.items()}
        return GlassBuilder(
            density=self.density, composition=normalized, structures=self.structures
        )
