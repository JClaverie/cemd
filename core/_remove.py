#
# This file is part of the CEMD distribution
# Copyright (c) 2024-2026 Jérôme Claverie.
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

import numpy as np
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from atomic_system import AtomicSystem

def remove_atoms(lmp_data: AtomicSystem, indices: list[int]) -> None:
    """Remove the given atoms

    Parameters
    ----------
        indices: list
            Indices of the atoms to remove

    """

    # create a copy of the atoms DataFrame
    df_atoms = lmp_data.atoms.copy()
    old_types = lmp_data.atom_types

    # remove atoms that are in the 'indices' list
    df_atoms = df_atoms.drop(indices)

    # create list to remap the atom indices in the DataFrame
    old_ids = df_atoms.index
    new_ids = np.arange(1, len(df_atoms) + 1)
    df_atoms.set_index(new_ids, inplace=True)
    lmp_data.atoms = df_atoms

    new_types = lmp_data.atom_types

    for i, t in enumerate(old_types):
        if t not in new_types:
            lmp_data.masses.pop(i)

    if lmp_data.velocities is not None:
        df_vel = lmp_data.velocities.copy()
        df_vel = df_vel.drop(indices)
        df_vel.set_index(new_ids, inplace=True)
        lmp_data.velocities = df_vel

    id_map = dict(zip(old_ids, new_ids))

    for name, n_cols in [('bonds', 2), ('angles', 3), ('dihedrals', 4), ('impropers', 4)]:
        df = getattr(lmp_data, name)
        if df is None:
            continue
        atom_cols = [f'atom_{i}' for i in range(1, n_cols + 1)]
        df = df.loc[~df[atom_cols].isin(indices).any(axis=1)].copy()
        df.index = np.arange(1, len(df) + 1)
        for col in atom_cols:
            df[col] = df[col].map(id_map)
        setattr(lmp_data, name, df if len(df) > 0 else None)


def remove_connection_types(lmp_data: AtomicSystem, 
                            bond_types: Sequence[str | int], 
                            angle_types: Sequence[str | int], 
                            dihedral_types: Sequence[str | int], improper_types: Sequence[str | int]
                            ) -> None:
    """Remove the given bond, angle, dihedral or improper types

    Parameters
    ----------
        bond_types: list
            Bond types to remove in the format [1,2] or ["O-Si", "H-O"] for example
        angle_types: list
            Angle types to remove
        dihedral_types: list
            Dihedral types to remove
        improper_types: list
            Improper types to remove

    """

    if bond_types is None: bond_types = []
    if angle_types is None: angle_types = []
    if dihedral_types is None: dihedral_types = []
    if improper_types is None: improper_types = []

    if len(bond_types) != 0:
        # list of remaining bond types
        remaining_bonds_list = list(set(lmp_data.bond_types) - set(bond_types))

        # remove bonds
        lmp_data.bonds = lmp_data.bonds[~lmp_data.bonds.type.isin(bond_types)]

        # update bonds indices
        lmp_data.bonds.index = list(range(1, len(lmp_data.bonds) + 1))

        # update bonds types range
        lmp_data.bonds.type.replace(lmp_data.bonds.type.unique(), remaining_bonds_list, inplace=True)

        # update system info
        if len(lmp_data.bonds) == 0:
            lmp_data.bonds = None

    if len(angle_types) != 0:
        # list of remaining angle types
        remaining_angles_list = list(set(lmp_data.angle_types) - set(angle_types))

        # remove angles
        lmp_data.angles = lmp_data.angles[~lmp_data.angles.type.isin(angle_types)]

        # update angles indices
        lmp_data.angles.index = list(range(1, len(lmp_data.angles) + 1))

        # update angles types range
        lmp_data.angles.type.replace(lmp_data.angles.type.unique(), remaining_angles_list, inplace=True)

        # update system info
        if len(lmp_data.angles) == 0:
            lmp_data.angles = None

    # remove dihedrals
    if len(dihedral_types) != 0:
        # list of remaining dihedrals types
        remaining_dihedrals_list = list(set(lmp_data.dihedral_types) - set(dihedral_types))

        # remove dihedrals
        lmp_data.dihedrals = lmp_data.dihedrals[lmp_data.dihedrals.type.isin(dihedral_types)]

        # update dihedrals indices
        lmp_data.dihedrals.index = list(range(1, len(lmp_data.dihedrals) + 1))

        # update dihedrals types range
        lmp_data.dihedrals.type.replace(
            lmp_data.dihedrals.type.unique(),
            remaining_dihedrals_list, inplace=True)

        # update system info
        if len(lmp_data.dihedrals) == 0:
            lmp_data.dihedrals = None

    # remove impropers
    if len(improper_types) != 0:
        # list of remaining improper types
        remaining_impropers_list = list(set(lmp_data.improper_types) - set(improper_types))

        # remove impropers
        lmp_data.impropers = lmp_data.impropers[lmp_data.impropers.type.isin(improper_types)]

        # update impropers indices
        lmp_data.impropers.index = list(range(1, len(lmp_data.impropers) + 1))

        # update impropers types range
        lmp_data.impropers.type.replace(
            lmp_data.impropers.type.unique(),
            remaining_impropers_list, inplace=True)

        # update system info
        if len(lmp_data.impropers) == 0:
            lmp_data.impropers = None

def keep_connection_types(lmp_data: AtomicSystem, 
                          bond_types: Sequence[str | int], 
                          angle_types: Sequence[str | int], 
                          dihedral_types: Sequence[str | int], 
                          improper_types: Sequence[str | int]
                          ) -> None:
    """Keep the given bond, angle, dihedral or improper types. Remove the other.

    Parameters
    ----------
        bond_types
            Bond types to remove
        angle_types
            Angle types to remove
        dihedral_types
            Dihedral types to remove
        improper_types
            Improper types to remove

    """

    if bond_types is None: bond_types = []
    if angle_types is None: angle_types = []
    if dihedral_types is None: dihedral_types = []
    if improper_types is None: improper_types = []

    bondtypes2remove = []
    angletypes2remove = []
    dihedraltypes2remove = []
    impropertypes2remove = []

    if lmp_data.num_bond_types != 0:
        bondtypes2remove = list(set(lmp_data.bond_types) - set(bond_types))

    if lmp_data.num_angle_types != 0:
        angletypes2remove = list(set(lmp_data.angle_types) - set(angle_types))

    if lmp_data.num_dihedral_types != 0:
        dihedraltypes2remove = list(set(lmp_data.dihedral_types) - set(dihedral_types))

    if lmp_data.num_improper_types != 0:
        impropertypes2remove = list(set(lmp_data.improper_types) - set(improper_types))

    lmp_data.remove_connection_types(
        bond_types=bondtypes2remove,
        angle_types=angletypes2remove,
        dihedral_types=dihedraltypes2remove,
        improper_types=impropertypes2remove)