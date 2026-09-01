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
from typing import Any


@dataclass
class ForceFieldParams:
    """Force-field parameters assigned to a system."""

    pair: dict[Any, Any] = field(default_factory=dict)
    bond: dict[Any, Any] = field(default_factory=dict)
    angle: dict[Any, Any] = field(default_factory=dict)
    dihedral: dict[Any, Any] = field(default_factory=dict)
    improper: dict[Any, Any] = field(default_factory=dict)

    bondbond: dict[Any, Any] = field(default_factory=dict)
    bondangle: dict[Any, Any] = field(default_factory=dict)
    middlebondtorsion: dict[Any, Any] = field(default_factory=dict)
    endbondtorsion: dict[Any, Any] = field(default_factory=dict)
    angletorsion: dict[Any, Any] = field(default_factory=dict)
    angleangletorsion: dict[Any, Any] = field(default_factory=dict)
    angleangle: dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def from_system_dict(cls, system_dict: dict[str, Any]) -> ForceFieldParams:
        """Create force-field parameters from a system dictionary."""
        return cls(
            pair=system_dict.get("pair_params", {}),
            bond=system_dict.get("bond_params", {}),
            angle=system_dict.get("angle_params", {}),
            dihedral=system_dict.get("dihedral_params", {}),
            improper=system_dict.get("improper_params", {}),
            bondbond=system_dict.get("bondbond_params", {}),
            bondangle=system_dict.get("bondangle_params", {}),
            middlebondtorsion=system_dict.get("middlebondtorsion_params", {}),
            endbondtorsion=system_dict.get("endbondtorsion_params", {}),
            angletorsion=system_dict.get("angletorsion_params", {}),
            angleangletorsion=system_dict.get("angleangletorsion_params", {}),
            angleangle=system_dict.get("angleangle_params", {}),
        )

    def to_system_dict(self) -> dict[str, dict[Any, Any]]:
        """Convert force-field parameters to system dictionary."""
        return {
            "pair_params": self.pair,
            "bond_params": self.bond,
            "angle_params": self.angle,
            "dihedral_params": self.dihedral,
            "improper_params": self.improper,
            "bondbond_params": self.bondbond,
            "bondangle_params": self.bondangle,
            "middlebondtorsion_params": self.middlebondtorsion,
            "endbondtorsion_params": self.endbondtorsion,
            "angletorsion_params": self.angletorsion,
            "angleangletorsion_params": self.angleangletorsion,
            "angleangle_params": self.angleangle,
        }

    def active(
        self,
        interaction: str,
        types,
    ) -> dict[Any, Any]:
        """Return force-field parameters for active types."""
        params = getattr(self, interaction)

        return {t: params[t] for t in types if t in params}

    def active_for(self, system) -> ForceFieldParams:
        """Return force-field parameters for active types."""
        return ForceFieldParams(
            pair=self.pair,
            bond=self.active(
                "bond",
                system.bond_types,
            ),
            angle=self.active(
                "angle",
                system.angle_types,
            ),
            dihedral=self.active(
                "dihedral",
                system.dihedral_types,
            ),
            improper=self.active(
                "improper",
                system.improper_types,
            ),
            bondbond=self.active(
                "bondbond",
                system.angle_types,
            ),
            bondangle=self.active(
                "bondangle",
                system.angle_types,
            ),
            middlebondtorsion=self.active(
                "middlebondtorsion",
                system.dihedral_types,
            ),
            endbondtorsion=self.active(
                "endbondtorsion",
                system.dihedral_types,
            ),
            angletorsion=self.active(
                "angletorsion",
                system.dihedral_types,
            ),
            angleangletorsion=self.active(
                "angleangletorsion",
                system.dihedral_types,
            ),
            angleangle=self.active(
                "angleangle",
                system.angle_types,
            ),
        )


@dataclass
class ForceFieldKeys:
    """Force-field key mappings."""

    atom: dict[Any, str] = field(default_factory=dict)
    bond: dict[Any, str] = field(default_factory=dict)
    angle: dict[Any, str] = field(default_factory=dict)
    dihedral: dict[Any, str] = field(default_factory=dict)
    improper: dict[Any, str] = field(default_factory=dict)

    @classmethod
    def from_system_dict(cls, system_dict: dict[str, Any]) -> ForceFieldKeys:
        """Create force-field key mappings from a system dictionary."""
        return cls(
            atom=system_dict.get("atom_ff_keys", {}),
            bond=system_dict.get("bond_ff_keys", {}),
            angle=system_dict.get("angle_ff_keys", {}),
            dihedral=system_dict.get("dihedral_ff_keys", {}),
            improper=system_dict.get("improper_ff_keys", {}),
        )

    def to_system_dict(self) -> dict[str, dict[Any, str]]:
        """Convert force-field key mappings to system dictionary."""
        return {
            "atom_ff_keys": self.atom,
            "bond_ff_keys": self.bond,
            "angle_ff_keys": self.angle,
            "dihedral_ff_keys": self.dihedral,
            "improper_ff_keys": self.improper,
        }

    def active(self, interaction: str, types) -> dict[Any, str]:
        """Return force-field keys for the active types."""
        mapping = getattr(self, interaction)

        return {t: mapping[t] for t in types if t in mapping}

    def active_for(self, system) -> ForceFieldKeys:
        """Return force-field keys for active types."""
        return ForceFieldKeys(
            atom=self.active("atom", system.atom_types),
            bond=self.active("bond", system.bond_types),
            angle=self.active("angle", system.angle_types),
            dihedral=self.active("dihedral", system.dihedral_types),
            improper=self.active("improper", system.improper_types),
        )
