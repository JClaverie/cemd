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
    def from_topology(cls, topology: dict[str, Any]) -> ForceFieldParams:
        """Create force-field parameters from a topology dictionary."""
        return cls(
            pair=topology.get("pair_params", {}),
            bond=topology.get("bond_params", {}),
            angle=topology.get("angle_params", {}),
            dihedral=topology.get("dihedral_params", {}),
            improper=topology.get("improper_params", {}),
            bondbond=topology.get("bondbond_params", {}),
            bondangle=topology.get("bondangle_params", {}),
            middlebondtorsion=topology.get("middlebondtorsion_params", {}),
            endbondtorsion=topology.get("endbondtorsion_params", {}),
            angletorsion=topology.get("angletorsion_params", {}),
            angleangletorsion=topology.get("angleangletorsion_params", {}),
            angleangle=topology.get("angleangle_params", {}),
        )

    def to_topology(self) -> dict[str, dict[Any, Any]]:
        """Convert force-field parameters to topology dictionary."""
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
        """Return force-field parameters for active topology types."""
        params = getattr(self, interaction)

        return {t: params[t] for t in types if t in params}

    def active_for(self, system) -> ForceFieldParams:
        """Return force-field parameters for active topology types."""
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
                system.bond_types,
            ),
            bondangle=self.active(
                "bondangle",
                system.bond_types,
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
    def from_topology(cls, topology: dict[str, Any]) -> ForceFieldKeys:
        """Create force-field key mappings from a topology dictionary."""
        return cls(
            atom=topology.get("atom_ff_keys", {}),
            bond=topology.get("bond_ff_keys", {}),
            angle=topology.get("angle_ff_keys", {}),
            dihedral=topology.get("dihedral_ff_keys", {}),
            improper=topology.get("improper_ff_keys", {}),
        )

    def to_topology(self) -> dict[str, dict[Any, str]]:
        """Convert force-field key mappings to topology dictionary."""
        return {
            "atom_ff_keys": self.atom,
            "bond_ff_keys": self.bond,
            "angle_ff_keys": self.angle,
            "dihedral_ff_keys": self.dihedral,
            "improper_ff_keys": self.improper,
        }

    def active(self, interaction: str, types) -> dict[Any, str]:
        """Return force-field keys for the active topology types."""
        mapping = getattr(self, interaction)

        return {t: mapping[t] for t in types if t in mapping}

    def active_for(self, system) -> ForceFieldKeys:
        """Return force-field keys for active topology types."""
        return ForceFieldKeys(
            atom=self.active("atom", system.atom_types),
            bond=self.active("bond", system.bond_types),
            angle=self.active("angle", system.angle_types),
            dihedral=self.active("dihedral", system.dihedral_types),
            improper=self.active("improper", system.improper_types),
        )
