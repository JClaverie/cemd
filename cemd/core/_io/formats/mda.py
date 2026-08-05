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

import numpy as np
import pandas as pd

from .base import BaseReader
from ...._utils import lattice2lammps


class MdaReader(BaseReader):
    """Read from MDAnalysis Universe or AtomGroup."""

    @staticmethod
    def _remap2numerical(types):
        """Remap bonds, angles, ... to numbers if atom types are numericals."""
        unique_types = np.unique(types)
        remap_dic = {k: v for k, v in zip(unique_types, list(range(1, len(unique_types) + 1)))}
        mapped_func = np.vectorize(remap_dic.get)
        return mapped_func(types)

    @staticmethod
    def _get_connection_types(conn, name: str, numerical_types: bool = False) -> np.ndarray:
        """Get connection types from MDAnalysis connection."""
        
        # Nombre d'atomes par type de connexion
        n_atoms_map = {
            'bonds': 2,
            'angles': 3,
            'dihedrals': 4,
            'impropers': 4,
        }
        n_atoms = n_atoms_map.get(name, 0)
        
        types = []
        
        for connection_idx in range(len(conn)):
            atom_indices = conn.indices[connection_idx]
            
            # Ignorer les connexions avec un nombre d'atomes incorrect
            if len(atom_indices) != n_atoms:
                continue
            
            atom_types = []
            for idx in atom_indices:
                atom = conn.universe.atoms[idx]
                atom_types.append(str(atom.type))
            
            # Trier les types pour normaliser (ex: "H-O" au lieu de "O-H")
            type_str = '-'.join(sorted(atom_types))
            types.append(type_str)
        
        types_arr = np.array(types)
        
        # Remapper en numérique si nécessaire
        if numerical_types:
            types_arr = MdaReader._remap2numerical(types_arr)
        
        return types_arr

    @classmethod
    def read(cls, obj) -> dict:
        """Read from MDAnalysis Universe or AtomGroup."""
        
        if hasattr(obj, 'universe'):
            universe = obj.universe
        else:
            universe = obj

        if universe.dimensions is None:
            box = np.array([10, 10, 10, 90, 90, 90])
        else:
            box = universe.dimensions

        indices = universe.atoms.indices

        if universe.atoms.ids is None:
            new_indices = np.arange(len(universe.atoms))
            indices_remapping = {k: v for k, v in zip(indices, new_indices)}
            ids = new_indices + 1
            remap_indices = np.vectorize(lambda x: indices_remapping.get(x, x))
        else:
            ids = universe.atoms.ids

        types = universe.atoms.types
        masses = universe.atoms.masses
        positions = universe.atoms.positions

        try:
            charges = universe.atoms.charges
        except Exception:
            charges = np.zeros(len(ids))

        univ_masses_dic = {t: m for t, m in zip(types, masses)}
        univ_masses_dic = dict(sorted(univ_masses_dic.items()))

        univ_charges_dic = {t: c for t, c in zip(types, charges)}
        univ_charges_dic = dict(sorted(univ_charges_dic.items()))

        if isinstance(types[0], str):
            numerical_types = types[0].isdigit()
        else:
            numerical_types = True

        # Atoms
        stacked_arrays = np.column_stack((ids, types, charges, positions))
        columns = ['id', 'type', 'charge', 'x', 'y', 'z']
        df_atoms = pd.DataFrame(stacked_arrays, columns=columns)
        df_atoms['id'] = df_atoms['id'].astype(int)
        df_atoms[['charge', 'x', 'y', 'z']] = df_atoms[['charge', 'x', 'y', 'z']].astype(float)
        df_atoms.set_index('id', inplace=True)

        # Velocities
        velocities = None
        if hasattr(universe.atoms, 'velocities'):
            stacked_arrays = np.column_stack((ids, universe.atoms.velocities))
            columns = ['id', 'vx', 'vy', 'vz']
            df_velocities = pd.DataFrame(stacked_arrays, columns=columns)
            df_velocities['id'] = df_velocities['id'].astype(int)
            df_velocities.set_index('id', inplace=True)
            velocities = df_velocities

        # Bonds
        if hasattr(universe, 'bonds') and len(universe.bonds) != 0:
            if universe.atoms.ids is None:
                bonds = remap_indices(universe.bonds.indices) + 1
            else:
                bonds = universe.bonds.indices + 1

            bond_types = cls._get_connection_types(universe.bonds, 'bonds', numerical_types)

            columns = ['id', 'type', 'atom_1', 'atom_2']
            ids_conn = np.arange(1, len(bonds) + 1)
            stacked_arrays = np.column_stack((ids_conn, bond_types, bonds))
            df_bonds = pd.DataFrame(stacked_arrays, columns=columns)
            df_bonds['id'] = df_bonds['id'].astype(int)
            df_bonds.set_index('id', inplace=True)
        else:
            df_bonds = None

        # Angles
        if hasattr(universe, 'angles') and len(universe.angles) != 0:
            if universe.atoms.ids is None:
                angles = remap_indices(universe.angles.indices) + 1
            else:
                angles = universe.angles.indices + 1

            angle_types = cls._get_connection_types(universe.angles, 'angles', numerical_types)

            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3']
            ids_conn = np.arange(1, len(angles) + 1)
            stacked_arrays = np.column_stack((ids_conn, angle_types, angles))
            df_angles = pd.DataFrame(stacked_arrays, columns=columns)
            df_angles['id'] = df_angles['id'].astype(int)
            df_angles.set_index('id', inplace=True)
        else:
            df_angles = None

        # Dihedrals
        if hasattr(universe, 'dihedrals') and len(universe.dihedrals) != 0:
            if universe.atoms.ids is None:
                dihedrals = remap_indices(universe.dihedrals.indices) + 1
            else:
                dihedrals = universe.dihedrals.indices + 1

            dihedral_types = cls._get_connection_types(universe.dihedrals, 'dihedrals', numerical_types)

            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            ids_conn = np.arange(1, len(dihedrals) + 1)
            stacked_arrays = np.column_stack((ids_conn, dihedral_types, dihedrals))
            df_dihedrals = pd.DataFrame(stacked_arrays, columns=columns)
            df_dihedrals['id'] = df_dihedrals['id'].astype(int)
            df_dihedrals.set_index('id', inplace=True)
        else:
            df_dihedrals = None

        # Impropers
        if hasattr(universe, 'impropers') and len(universe.impropers) != 0:
            if universe.atoms.ids is None:
                impropers = remap_indices(universe.impropers.indices) + 1
            else:
                impropers = universe.impropers.indices + 1

            improper_types = cls._get_connection_types(universe.impropers, 'impropers', numerical_types)

            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            ids_conn = np.arange(1, len(impropers) + 1)
            stacked_arrays = np.column_stack((ids_conn, improper_types, impropers))
            df_impropers = pd.DataFrame(stacked_arrays, columns=columns)
            df_impropers['id'] = df_impropers['id'].astype(int)
            df_impropers.set_index('id', inplace=True)
        else:
            df_impropers = None

        topology = {
            'lmp_box': lattice2lammps(box),
            'masses': univ_masses_dic,
            'charges': univ_charges_dic,
            'atom_types': sorted(np.unique(types).tolist()),
            'atoms': df_atoms,
            'bonds': df_bonds,
            'angles': df_angles,
            'dihedrals': df_dihedrals,
            'impropers': df_impropers,
            'velocities': velocities,
            'atom_style': 'full',
        }

        return topology