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

import copy
from typing import Sequence, Any, Self

import numpy as np
import pandas as pd
import scipy.constants as cst

from ._edit import EditMixin
from ._topology import TopologyMixin
from ._io import IOMixin
from ._forcefield import ForceFieldMixin

from .visualization import view

from .._constants import(
    MASSES_DICT, 
    INV_MASSES, 
    MASS_KEYS
)
from .._utils import (
    lammps2lattice, 
    lattice2vectors, 
)

class AtomicSystem(EditMixin, IOMixin, TopologyMixin, ForceFieldMixin):
    """Object that contains all the information inside a LAMMPS datafile."""

    atoms: pd.DataFrame
    bonds: pd.DataFrame | None
    angles: pd.DataFrame | None
    dihedrals: pd.DataFrame | None
    impropers: pd.DataFrame | None
    velocities: pd.DataFrame | None

    _types: Sequence[str | int]
    _masses_storage: dict[str | int, float]
    _charges_storage: dict[str | int, float]
    _atom_style: str

    _lmp_box: tuple[
        tuple[float, float], # [xlo, xhi]
        tuple[float, float], # [ylo, yhi]
        tuple[float, float], # [zlo, zhi]
        tuple[float, float, float]  # [xy, xz, yz]
    ]
    _box: np.ndarray
    _box_vectors: Sequence[np.ndarray]

    pair_params: dict[str, Any]
    bond_params: dict[str, Any]
    angle_params: dict[str, Any]
    dihedral_params: dict[str, Any]
    improper_params: dict[str, Any]

    def __init__(self, topology: dict[str, Any]) -> None:
        """Initializes the AtomicSystem from a topology dictionary."""
        self._assign_topology(topology)
        self._finalize_data()

    def __setattr__(self, name, value) -> None:
        if name in ('atoms', 'bonds', 'angles', 'dihedrals', 'impropers'):
            self.__dict__['_cache'] = {}
        super().__setattr__(name, value)

    def _assign_topology(self, topology: dict[str, Any]) -> None:
        """Helper to map the topology dictionary to class attributes."""
        self._cache = {}
        self.atoms = topology['atoms']
        self.bonds = topology.get('bonds')
        self.angles = topology.get('angles')
        self.dihedrals = topology.get('dihedrals')
        self.impropers = topology.get('impropers')
        self.velocities = topology.get('velocities')
        
        self._lmp_box = topology['lmp_box']
        self._types = list(topology['atom_types'])
        self._masses_storage = dict(topology['masses'])
        self._charges_storage = dict(topology['charges'])
        self._atom_style = topology.get('atom_style', 'full')
        
        self._box = lammps2lattice(self._lmp_box)
        self._box_vectors = lattice2vectors(self._box)

        self._pmg_struct = topology.get('_pmg_struct', None)

        self.pair_params = topology.get('pair_params', None)
        self.bond_params = topology.get('bond_params', None)
        self.angle_params = topology.get('angle_params', None)
        self.dihedral_params = topology.get('dihedral_params', None)
        self.improper_params =topology.get('improper_params', None)

    def _finalize_data(self) -> None:
        """Sorts indices and ensures integer types for atom references."""
        for df in [self.atoms, self.velocities, self.bonds, self.angles, self.dihedrals, self.impropers]:
            if df is not None:
                df.sort_index(inplace=True)

        topology_dfs = {
            'bonds': 2, 'angles': 3, 'dihedrals': 4, 'impropers': 4
        }
        for name, n_atoms in topology_dfs.items():
            df = getattr(self, name)
            if df is not None:
                cols = [f'atom_{i}' for i in range(1, n_atoms + 1)]
                df[cols] = df[cols].astype(int)

    def _replace_internals(self, other: AtomicSystem) -> None:
        """Remplace le contenu interne par celui d'un autre système, sans casser les références."""
        self._assign_topology({
            'atoms':      other.atoms.copy(),
            'bonds':      other.bonds,
            'angles':     other.angles,
            'dihedrals':  other.dihedrals,
            'impropers':  other.impropers,
            'velocities': other.velocities,
            'lmp_box':    other._lmp_box,
            'atom_types': list(other._types),
            'masses':     dict(other._masses_storage),
            'charges':    dict(other._charges_storage),
            'atom_style': other._atom_style,
        })
        self.pair_params     = dict(other.pair_params)
        self.bond_params     = dict(other.bond_params)
        self.angle_params    = dict(other.angle_params)
        self.dihedral_params = dict(other.dihedral_params)
        self.improper_params = dict(other.improper_params)
        if other._pmg_struct is not None:
            import copy
            self._pmg_struct = copy.deepcopy(other._pmg_struct)

    def copy(self) -> Self:
        new = self.__class__.__new__(self.__class__)
        
        topology = {
            'atoms':     self.atoms.copy(),
            'bonds':     self.bonds.copy()     if self.bonds     is not None else None,
            'angles':    self.angles.copy()    if self.angles    is not None else None,
            'dihedrals': self.dihedrals.copy() if self.dihedrals is not None else None,
            'impropers': self.impropers.copy() if self.impropers is not None else None,
            'velocities':self.velocities.copy()if self.velocities is not None else None,
            'lmp_box':   self.lmp_box,
            'atom_types': list(self._types),
            'masses':    dict(self._masses_storage),
            'charges':   dict(self._charges_storage),
            'atom_style': self._atom_style,
        }
        
        new._assign_topology(topology)
        new._finalize_data()
        
        # Copie des paramètres de champ de force si définis
        new.pair_params     = dict(self.pair_params)
        new.bond_params     = dict(self.bond_params)
        new.angle_params    = dict(self.angle_params)
        new.dihedral_params = dict(self.dihedral_params)
        new.improper_params = dict(self.improper_params)
        
        if self._pmg_struct is not None:
            import copy
            new._pmg_struct = copy.deepcopy(self._pmg_struct)

        return new

    @property
    def box(self) -> np.ndarray:
        '''Return the lattice parameters'''
        return copy.copy(self._box)
        
    @property
    def volume(self) -> float:
        '''Return the volume of the box.'''
        v1, v2, v3 = self._box_vectors
        return np.dot(v1, np.cross(v2, v3))

    @property
    def masses(self) -> list[float]:
        '''Return the list of masses corresponding to self.atom_types.'''
        if 'masses' not in self._cache:
            mass_list = [
                float(self._masses_storage.get(t, MASSES_DICT.get(t, 1.0)))
                for t in self.atom_types
            ]
            self._cache['masses'] = mass_list
        return self._cache['masses']

    @property
    def charges(self) -> list[float]:
        '''Return the list of charges corresponding to self.atom_types.'''
        if 'charges' not in self._cache:
            self._cache['charges'] = [
                float(self._charges_storage.get(atype, 0))
                for atype in self.atom_types
            ]
        return self._cache['charges']

    @property
    def elements(self) -> dict[str | int , str]:
        """Returns a perfectly aligned dictionary {type: symbol}."""
        
        if 'elements' not in self._cache:
            element_list = []
            current_types = self.atom_types 
            current_masses = self.masses     
            
            for i, t_id in enumerate(current_types):
                m_val = float(current_masses[i])
                best_match = MASS_KEYS[(np.abs(MASS_KEYS - m_val)).argmin()]
                symbol = str(INV_MASSES[best_match])
                
                # Cleaning the type to remove the np.str_
                clean_id = str(t_id)
                if clean_id.isdigit(): clean_id = int(clean_id)
                    
                element_list.append(symbol)

            self._cache['elements'] = element_list
            
        return self._cache['elements']

    @property
    def num_atoms(self) -> int:
        '''Return the number of atoms.'''
        return len(self.atoms)

    @property
    def atom_types(self) -> list[str | int]:
        '''Return the list of atom types.'''
        if 'atom_types' not in self._cache:
            self._cache['atom_types'] = sorted([str(t) for t in self.atoms.type.unique()])
        return self._cache['atom_types']

    @property
    def bond_types(self) -> list[str | int]:
        '''Return the list of bond types.'''
        if self.bonds is not None:
            return sorted(self.bonds.type.unique().tolist())
        else:
            return list()

    @property
    def angle_types(self) -> list[str | int]:
        '''Return the list of angles types.'''
        if self.angles is not None:
            return sorted(self.angles.type.unique().tolist())
        else:
            return list()

    @property
    def dihedral_types(self) -> list[str | int]:
        '''Return the list of dihedral types.'''
        if self.dihedrals is not None:
            return sorted(self.dihedrals.type.unique().tolist())
        else:
            return list()

    @property
    def improper_types(self) -> list[str | int]:
        '''Return the list of improper types.'''
        if self.impropers is not None:
            return sorted(self.impropers.type.unique().tolist())
        else:
            return list()

    @property
    def num_bonds(self) -> int:
        '''Return the number of bonds.'''
        if self.bonds is None:
            return 0
        else:
            return len(self.bonds)
    @property
    def num_angles(self) -> int:
        '''Return the number of angles.'''
        if self.angles is None:
            return 0
        else:
            return len(self.angles)
    @property
    def num_dihedrals(self) -> int:
        '''Return the number of dihedrals.'''
        if self.dihedrals is None:
            return 0
        else:
            return len(self.dihedrals)
    @property
    def num_impropers(self) -> int:
        '''Return the number of impropers.'''
        if self.impropers is None:
            return 0
        else:
            return len(self.impropers)

    @property
    def num_atom_types(self) -> int:
        '''Return the number of atom types.'''
        return len(self.atom_types)
    @property
    def num_bond_types(self) -> int:
        '''Return the number of bond types.'''
        return len(self.bond_types)
    @property
    def num_angle_types(self) -> int:
        '''Return the number of angle types.'''
        return len(self.angle_types)
    @property
    def num_dihedral_types(self) -> int:
        '''Return the number of dihedrals types.'''
        return len(self.dihedral_types)
    @property
    def num_improper_types(self) -> int:
        '''Return the number of improper types.'''
        return len(self.improper_types)

    @property
    def total_charge(self) -> float:
        '''Return the total charge.'''
        return self.atoms.charge.sum()

    @property
    def total_mass(self) -> float:
        '''Return the total mass.'''
        counts = self.atoms['type'].value_counts()
        total_mass = sum(counts.get(atype, 0) * mass for atype, mass in zip(self.atom_types, self.masses))
        return total_mass

    @property
    def density(self) -> float:
        '''Return the density.'''
        return self.total_mass / cst.Avogadro / self.volume / 1e-24

    @property
    def type_summary(self) -> pd.DataFrame:
        """Returns a summarized DataFrame of atom types, numbers, masses and charges."""
        if 'type_summary' not in self._cache:
            df_atoms = self.atoms.copy()
            
            df_atoms['number'] = df_atoms.groupby('type')['type'].transform('size')
            
            red_df = df_atoms.drop_duplicates(subset='type')[['type', 'number', 'charge']]
            
            red_df['sort_key'] = red_df['type'].apply(lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else x))
            red_df = red_df.sort_values('sort_key').drop(columns=['sort_key'])
            
            red_df['mass'] = red_df['type'].apply(
            lambda t: float(self._masses_storage.get(t, MASSES_DICT.get(t, 1.0)))
        )

            self._cache['type_summary'] = red_df
        
        return self._cache['type_summary']

    def __repr__(self) -> str:
        """Return information about the box size, number of atoms, charge and density."""
        
        # Header with interaction count
        output_string = f"<AtomicSystem with {self.num_atoms} atoms"
        sections = {
            "bonds": self.num_bonds,
            "angles": self.num_angles,
            "dihedrals": self.num_dihedrals,
            "impropers": self.num_impropers
        }
        active_sections = [f"{v} {k}" for k, v in sections.items() if v > 0]
        if active_sections:
            output_string += ", " + ", ".join(active_sections)
        output_string += ">\n\n"

        # Section Box
        output_string += "Box\n"
        df_box = pd.DataFrame(np.reshape(self.box.T, (1, 6)), columns=[
            'a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)'
        ])
        output_string += df_box.to_string(index=False, float_format='%.2f') + "\n\n"

        # Atoms Section (Using type_summary)
        output_string += "Atoms\n"
        red_df = self.type_summary.copy()
        
        # Added percentage for console display
        red_df['%'] = (red_df['number'] / red_df['number'].sum()) * 100
        red_df['%'] = red_df['%'].map(lambda x: f'{x:,.2f}')
        
        # Reorganization of columns for visual rendering
        column_order = ['type', 'number', '%', 'mass', 'charge']
        output_string += red_df[column_order].to_string(index=False) + "\n\n"

        # Sections Interactions (Bonds, Angles, etc.)
        def append_interaction_info(output_string, df, name):
            if df is not None and len(df) > 0:
                output_string += f"{name}\n"
                df_copy = df.copy()
                df_copy['number'] = df_copy.groupby('type')['type'].transform('size')
                summary = df_copy.drop_duplicates(subset='type')[['type', 'number']]
                # Sorting interaction types
                summary['sk'] = summary['type'].apply(lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else x))
                summary = summary.sort_values('sk').drop(columns=['sk'])
                output_string += summary.to_string(index=False) + "\n\n"
            return output_string

        output_string = append_interaction_info(output_string, self.bonds, "Bonds")
        output_string = append_interaction_info(output_string, self.angles, "Angles")
        output_string = append_interaction_info(output_string, self.dihedrals, "Dihedrals")
        output_string = append_interaction_info(output_string, self.impropers, "Impropers")

        # Footer (Physical Summary)
        output_string += f"Total charge: {self.total_charge:.3f}e\n"
        output_string += f"Volume: {self.volume/1e3:.2f} nm3\n"
        output_string += f"Density: {self.density:.2f} g/cm3"

        return output_string

    def __str__(self) -> str:
        return self.__repr__()
    
    def get_count(self, symbol: str | int) -> int:
        """
        Calculates the exact number of atoms for a specific type.
        Raises a ValueError if the symbol is not found.
        """
        summary = self.type_summary
        
        mask = summary['type'].astype(str) == symbol
        
        if not mask.any():
            res = 0

        else: 
            res = int(summary.loc[mask, 'number'].sum())
            
        return res
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculates the center of mass (COM) of the system.
        
        Returns:
            np.ndarray: [x, y, z] coordinates of the center of mass.
        """
        atom_masses = self.atoms['type'].map(
        lambda t: self._masses_storage.get(t, MASSES_DICT.get(t, 1.0))
        )
        
        total_mass = atom_masses.sum()
        
        if total_mass <= 0:
            return np.zeros(3)

        weighted_pos = self.atoms[['x', 'y', 'z']].multiply(atom_masses, axis=0)

        return weighted_pos.sum().values / total_mass

    def view(self, trajectory=None) -> None:
        """Visualizes the current system in VMD, with an optional trajectory.

        Args:
            trajectory (str, optional): Path to a trajectory file (.dcd) 
                to overlay onto this system's topology. Defaults to None.
                
        Example:
            >>> system = AtomicSystem("input.data")
            >>> system.view()
            >>> system.view(trajectory="production.dcd")
        """

        view(self, trajectory=trajectory)




 