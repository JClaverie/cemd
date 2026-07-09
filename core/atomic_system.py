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

import os
import copy
from typing import Sequence, Any, Self

import MDAnalysis as mda
import numpy as np
import pandas as pd
import scipy.constants as cst
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from . import _set, _remove
from ._io import (
    read_lmp, 
    read_mda, 
    read_pmg, 
    read_smiles,
    to_mda, 
    to_pmg, 
    write_lmp,
    write_pdb
)
from .visualization import view

from .._config import FF_DATABASE_FILE
from .._constants import(
    MASSES_DICT, 
    INV_MASSES, 
    MASS_KEYS
)
from .._utils import (
    lammps2lattice, 
    lattice2vectors, 
    lattice2lammps, 
    vectors2lattice
)

class AtomicSystem:
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

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Creates a system from a file (.pdb, .cif, .data)."""

        if not os.path.exists(path):
            raise FileNotFoundError(f"The file '{path}' does not exist.")
    
        ext = os.path.splitext(path)[1]

        if ext == '.data':
            topology = read_lmp(path)

        elif ext == '.cif':
            raw_struct = Structure.from_file(path)
            analyzer = SpacegroupAnalyzer(raw_struct)
            refined_struct = analyzer.get_refined_structure()
            topology = read_pmg(refined_struct, raw_struct.lattice.abc)
            topology['_pmg_struct'] = raw_struct

        elif ext == '.pdb':
            univ = mda.Universe(path)
            univ.atoms.types = [string.capitalize() for string in univ.atoms.names]

            masses = []
            for atype in univ.atoms.types:
                masses.append(MASSES_DICT[atype])
            univ.atoms.masses = np.array(masses)

            topology = read_mda(univ)
                
        else:
            raise ValueError(f"Extension {ext} not supported.")
        
        return cls(topology)
            
    @classmethod
    def from_dict(cls, topo_dict: dict) -> Self:
        """Creates a system from a topology dictionnary."""
        return cls(topo_dict)
    
    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        """Creates a system from a SMILES string"""
        topo_dict = read_smiles(smiles)
        return cls(topo_dict)
    
    @classmethod
    def from_mda(cls, obj: mda.Universe | mda.AtomGroup) -> Self:
        """Creates a system from a MDAnalysis object."""
        topology = read_mda(obj)
        return cls(topology)

    @classmethod
    def from_pymatgen(cls, struct: Structure) -> Self:
        """Creates a system from a Pymatgen Structure."""
        analyzer = SpacegroupAnalyzer(struct)
        refined_struct  = analyzer.get_refined_structure()
        topology = read_pmg(refined_struct, struct.lattice.abc)
        topology['_pmg_struct'] = struct
        return cls(topology)

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

        # (type_i, type_j): [epsilon, sigma]
        self.pair_params = {}
        self.bond_params = {}
        self.angle_params = {}
        self.dihedral_params = {}
        self.improper_params = {}

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
    
    def center_on_com(self) -> None:
        """
        Translates the system so its center of mass is at the center of the box,
        correctly handling Periodic Boundary Conditions (PBC).
        """
        import numpy as np

        # Prepare dimensions and weights
        box_dims = self.box[:3]  # [Lx, Ly, Lz]
        atom_masses = self.atoms['type'].map(lambda t: self._masses_storage.get(t, MASSES_DICT.get(t, 1.0)))
        total_mass = atom_masses.sum()
        
        if total_mass <= 0:
            return

        # Calculate the Periodic Center of Mass (Bai and Breen method)
        # transform coordinates to periodic angles: theta = 2 * pi * x / L
        com_coords = []
        for i, axis in enumerate(['x', 'y', 'z']):
            L = box_dims[i]
            theta = (self.atoms[axis] / L) * 2 * np.pi
            
            # Average of sine and cosine weighted by mass
            avg_sin = (np.sin(theta) * atom_masses).sum() / total_mass
            avg_cos = (np.cos(theta) * atom_masses).sum() / total_mass
            
            # Back to average angle, then back to coordinate
            avg_theta = np.arctan2(-avg_sin, -avg_cos) + np.pi
            com_coords.append((avg_theta / (2 * np.pi)) * L)
        
        com_pbc = np.array(com_coords)

        # Translate the system
        # Target: put the COM at the center of the box (box_dims / 2)
        target = box_dims / 2
        shift = target - com_pbc
        
        self.atoms[['x', 'y', 'z']] += shift
        
        # Wrap everything back into the box [0, L]
        self.wrap()

    def set_box(self, new_box: Sequence[float] | np.ndarray) -> None:
        """Assign a new box to the system.

        Parameters
        ----------
            new_box: list of float
                list of the new box parameters to assign
        """

        if isinstance(new_box, list):
            new_box = np.array(new_box)
            
        self._box = new_box
        self._lmp_box = lattice2lammps(new_box)
        self._box_vectors = lattice2vectors(self._box)

    def set_types(self, new_types: Sequence[str | int] | dict[str | int, str | int]) -> None:
        """Assign types to atoms.

        Parameters
        ----------
            new_types: list of str
                list of the new types to assign (sorted by atom type)
        """
        self._cache = {}
        _set.set_types(self, new_types)

    def set_type2atoms(self, indices: list[int], atom_type: str | int) -> None:
        """Assign a type to a list of atoms.

        Parameters
        ----------
            indices: list of int
                list of indices of atoms
            atom_type: str
                The type to assign to the atoms in indices
        """
        self._cache = {}
        _set.set_type2atoms(self, indices, atom_type)

    def set_coordinates(
    self, 
    indice: int, 
    position: Sequence[float] | np.ndarray
) -> None:
        """
        Assign new coordinates to one atom.
        
        Args:
            indices: Single index or list of atom indices to modify.
            position: list or tuple of 3 values [x, y, z]. 
                    Use None for coordinates that should remain unchanged.
        """
        if len(position) != 3:
            raise ValueError("position must contain 3 elements: [x, y, z]")

        _set.set_coordinates(self, indice, position)


    def set_masses(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """
        Sets the masses for the atomic system.
        Accepts a list (matched to current atom_types order) or a dictionary {type: mass}.
        """
        if isinstance(value, dict):
            # Update internal storage by type name
            self._masses_storage.update(value)
            
        elif isinstance(value, (list, np.ndarray, tuple)):
            # Security check: ensure the length matches the number of types
            current_types = self.atom_types
            if len(value) != len(current_types):
                import traceback
                traceback.print_stack()  # ← pour voir d'où vient l'appel
                print(f"ERROR: ...")
                return
                print(f"ERROR: Mass list length ({len(value)}) does not match "
                    f"the number of types ({len(current_types)}).")
                return
                
            # Create a temporary map to update the internal storage
            new_map = dict(zip(current_types, value))
            self._masses_storage.update(new_map)

        self._cache = {}

    def set_charges(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """
        Sets or updates the charges for the atomic system and the atoms DataFrame.
        
        Supports partial dictionary updates or complete sequence mapping.
        """
        if isinstance(value, dict):
            # Optional check: warn the user if a target atom type does not exist
            current_types = set(self.atom_types)
            for atype in value.keys():
                if atype not in current_types:
                    print(f"WARNING: Type '{atype}' targeted in set_charges is not currently in the system.")
            
            # Update the internal storage (overwrites existing or adds new keys)
            self._charges_storage.update(value)
            
        elif isinstance(value, (list, np.ndarray, tuple)):
            current_types = self.atom_types
            if len(value) != len(current_types):
                print(f"ERROR: Charge list length ({len(value)}) does not match "
                    f"the number of types ({len(current_types)}).")
                return
            new_map = dict(zip(current_types, value))
            self._charges_storage.update(new_map)

        # Update the atoms DataFrame while safely handling partial dictionary updates
        if hasattr(self, 'atoms') and self.atoms is not None:
            # 1. Map current atom types to the stored charges (creates NaN for omitted types)
            mapped_charges = self.atoms['type'].map(self._charges_storage)
            
            # 2. If mapped_charges contains a NaN, fall back to the existing charge in the DataFrame.
            #    If the 'charge' column does not exist yet, initialize it directly with mapped_charges.
            if 'charge' in self.atoms.columns:
                self.atoms['charge'] = mapped_charges.fillna(self.atoms['charge'])
            else:
                self.atoms['charge'] = mapped_charges

            # 3. Final safety check: Warn if any atom row ends up with an unassigned/NaN charge
            if self.atoms['charge'].isna().any():
                missing = self.atoms[self.atoms['charge'].isna()]['type'].unique()
                print(f"WARNING: Some atom types still have no charge assigned: {list(missing)}")
        
        self._cache = {}

    def remove_atoms(self, indices: list[int]) -> None:
        """Remove the given atoms

        Parameters
        ----------
            indices
                Indices of the atoms to remove
        """

        _remove.remove_atoms(self, indices)

    def set_bond(self, atom_list: Sequence[int], bond_type: str | int) -> None:
        """Set a bond between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the two atoms in the bond.
            bond_type
                Type of the bond (numerical or alphabetical)
        """

        _set.set_connection(self, 'bond', atom_list, bond_type)

    def set_angle(self, atom_list: Sequence[int], angle_type: str | int) -> None:
        """Set a angle between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the angle, in the right order.
            angle_type
                Type of the angle (numerical or alphabetical)
        """

        _set.set_connection(self, 'angle', atom_list, angle_type)

    def set_dihedral(self, atom_list: Sequence[int], dihedral_type: str | int) -> None:
        """Set a dihedral between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the dihedral, in the right order.
            dihedral_type
                Type of the dihedral (numerical or alphabetical)
        """

        _set.set_connection(self, 'dihedral', atom_list, dihedral_type)

    def set_improper(self, atom_list: Sequence[int], improper_type: str | int) -> None:
        """Set a improper between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the improper, in the right order.
            improper_type
                Type of the improper (numerical or alphabetical)
        """

        _set.set_connection(self, 'improper', atom_list, improper_type)

    
    def remove_bond(self, bond: int | list[int]) -> None:
        """Remove the given bond.
        Parameters
        ----------
            bond: int or list
                Index of the bond or list of the indices of the atoms in the bond.
        """

        _remove.remove_connection(self, 'bonds', bond)

    def remove_angle(self, angle: int | list[int]) -> None:
        """Remove the given angle.
        Parameters
        ----------
            angle: int or list
                Index of the angle or list of the indices of the atoms in the angle.
        """

        _remove.remove_connection(self, 'angles', angle)

    def remove_dihedral(self, dihedral: int | list[int]) -> None:
        """Remove the given dihedral.
        Parameters
        ----------
            dihedral: int or list
                Index of the dihedral or list of the indices of the atoms in the dihedral.
        """

        _remove.remove_connection(self, 'dihedrals', dihedral)

    def remove_improper(self, improper: int | list[int]) -> None:
        """Remove the given improper.
        Parameters
        ----------
            improper: int or list
                Index of the improper or list of the indices of the atoms in the improper.
        """

        _remove.remove_connection(self, 'impropers', improper)

    def remove_connection_types(self, 
                                bond_types:Sequence[str | int]=None, 
                                angle_types:Sequence[str | int]=None, 
                                dihedral_types:Sequence[str | int]=None, 
                                improper_types:Sequence[str | int]=None
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

        _remove.remove_connection_types(self, bond_types, angle_types, dihedral_types, improper_types)

    def keep_connection_types(self, 
                                bond_types:Sequence[str | int]=None, 
                                angle_types:Sequence[str | int]=None, 
                                dihedral_types:Sequence[str | int]=None, 
                                improper_types:Sequence[str | int]=None
                                ) -> None:
        """Keep the given bond, angle, dihedral or improper types. Remove the other.

        Parameters
        ----------
            bond_types: list
                Bond types to remove
            angle_types: list
                Angle types to remove
            dihedral_types: list
                Dihedral types to remove
            improper_types: list
                Improper types to remove

        """

        _remove.keep_connection_types(self, bond_types, angle_types, dihedral_types, improper_types)

    def remove_all_connections(self) -> None:
        """Remove all bonds, angles, dihedrals and impropers."""

        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None

    def reset_types(self, prevent: Sequence[str | int]=None) -> None:
        """Reset atom types to elements based on masses.
        Useful after a LAMMPS run.

        Parameters
        ----------
            prevent:
                Atom types that will be prevented from the reset

        """
        self._cache = {}
        _set.reset_types(self, prevent)

        return self

    def reset_topo(self, prevent: Sequence[str | int]=None) -> None:
        """Both reset atom types and remove connectivity.

        Parameters
        ----------
            prevent
                Atom types that will be prevented from the reset

        """
        self.remove_all_connections()
        self.reset_types(prevent)

        return self
    
    def set_topo_rule(self, rule: dict) -> None:
        """Apply a motif-based topology rule to a MDAnalysis Universe.

        Parameters
        ----------
        universe : MDAnalysis.core.universe.Universe
            The MDAnalysis Universe to modify.
        rule : dict
            A dictionary defining the structural motif. Expected keys:
            -'center_sel' (str): Selection string for the pivot atom.
            -'neighbors' (list of dict): Requirements for the coordination shell.
            -'create_bond' (bool): Create bonds between center and neighbors.
            -'create_angle' (bool): Create angles between neighbors (pivot = center).
            -'create_improper' (bool): Create impropers if exactly 3 neighbors are found.

        Returns
        -------
        MDAnalysis.core.universe.Universe
            The modified Universe with updated topology attributes.
        """

        univ = to_mda(self)

        _set.set_topology_rule(univ, rule)
        new_lmp_data = AtomicSystem.from_mda(univ)

        self._replace_internals(new_lmp_data)

        return self


    def set_topo(self, style: str='clayff', reset: bool=True) -> str:
        """Set the topology (atom types, bonds an angles) based on a geometrical criteria and the box parameters if given. The atom types set without specific style are:

        -Ow: oxygen in water
        -Hw: hydrogen in water
        -Oh: oxygen in hydroxide
        -Hh: hydrogen in hydroxide
        -Osi: oxygen in silicate
        -Osih: oxygen in Si-O-H
        -Hsi: hydrogen in Si-O-H
        -Oa: oxygen in aluminate
        -Oah: oxygen in Al-O-H
        -Ha: oxygen in Al-O-H
        -Ob: bridging oxygen in Si-O-Si
        -Obs: brigding oxygen in Si-O-Al
        -Oc: oxygen in carbonate
        -Os: oxygen in sulfate

        Bonds are:

        -Hw-Ow
        -Hh-Oh
        -Hsi-Osih
        -Ha-Oah

        Angles are:

        -Hw-Ow-Ow

        Parameters
        ----------
            style: str
                A key word to obtain a specific topology.
                Supported:

                -'cshff': Label interlayer calcium ions as 'Cw'
            box: list of float
                [a, b, c, alpha, beta, gamma]
            reset: bool
                Reset or not the topology.

        """

        if reset:
            self.reset_topo()

        univ: mda.Universe = to_mda(self)

        if style == 'clayff':
            univ = _set.set_topo_clayff(univ)
            new_lmp_data = AtomicSystem.from_mda(univ)


        if style == 'cshff':
            univ = _set.set_topo_clayff(univ)
            new_lmp_data = AtomicSystem.from_mda(univ)
            list_ids_cw = _set.get_ids_cw(univ)
            new_lmp_data.set_type2atoms(list_ids_cw, "Cw")

        #TODO:
        # if style == 'iff':

        self._replace_internals(new_lmp_data)


    def set_ff_from_database(self, 
                             assignments: dict[str | int, str], 
                             ff_database: str=FF_DATABASE_FILE) -> None:
        """Automatically loads the forcefield database and applies parameters.

        This method reads the Excel database, extracts the Lennard-Jones, bond,
        and angle sheets, and orchestrates the automated parameter assignment for 
        the entire atomic system based on the provided atom mapping.

        Args:
            assignments (dict): Mapping between the system's current atom types/labels
                and the forcefield database types. 
                Example: {'H': 'hspc', 'O': 'ospc'}
            ff_database (str, optional): Absolute or relative path to the Excel 
                database file (.xls/.xlsx). Defaults to FF_DATABASE_PATH.

        Returns:
            Self: The instance of the system class (allows method chaining).

        Raises:
            FileNotFoundError: If the specified database file does not exist.
            ValueError: If required sheets ('lj_12-6', 'bond', 'angle') are missing.
        """
        all_sheets = pd.read_excel(ff_database, sheet_name=None)

        df_lj: pd.DataFrame | None = all_sheets.get('lj_12-6')
        df_bond: pd.DataFrame | None = all_sheets.get('bond')
        df_angle: pd.DataFrame | None = all_sheets.get('angle')

        if df_lj is None or df_bond is None or df_angle is None:
            raise ValueError(
                "Database is missing one or more required sheets ('lj_12-6', 'bond', 'angle')."
            )

        _set.set_pair_forcefield(self, assignments, df_lj)
        _set.set_bond_forcefield(self, assignments, df_bond)
        _set.set_angle_forcefield(self, assignments, df_angle)

    
    def set_pair_param(self, atom_type: str | int, pair_coeffs: list[float]) -> None:
        """Manually assigns non-bonded (L-J) parameters for a single atom type.

        Args:
            atom_type (str | int): The atom type/label present in the system 
                (e.g., 'H', 'O', or 1).
            pair_coeffs (list[float]): Lennard-Jones parameters [epsilon, sigma].

        Returns:
            Self: The instance of the system class (allows method chaining).
        """
        _set.set_atom_type_param(self, atom_type, pair_coeffs)
    
    def set_bond_param(self, bond_type: str, bond_coeffs: list[float]) -> None:
        """Manually assigns structural parameters for a single bond type.

        Args:
            bond_type (str): The bond label present in the system, 
                usually formatted with a hyphen (e.g., 'H-O').
            bond_coeffs (list[float]): Bond potential parameters [k, r0].

        Returns:
            Self: The instance of the system class (allows method chaining).
        """
        _set.set_bond_type_param(self, bond_type, bond_coeffs)

    def set_angle_param(self, 
                        angle_type: str, 
                        angle_coeffs: list[float]) -> None:
        """Manually assigns structural parameters for a single angle type.

        Args:
            angle_type (str): The angle label present in the system,
                usually formatted with hyphens (e.g., 'H-O-H').
            angle_coeffs (list[float]): Angle potential parameters [k, theta0].

        Returns:
            Self: The instance of the system class (allows method chaining).
        """
        _set.set_angle_type_param(self, angle_type, angle_coeffs)

    def add_atom(self, 
                 atype: str | int, 
                 position: list[float], 
                 charge: float=0.0, 
                 mass: float=None) -> int:
        """
        Add a single atom to the system and synchronize metadata.

        Parameters
        ----------
        atype
            The atom type or symbol (e.g., 'H', 'Ow', 'Ca').
        position
            Cartesian coordinates of the atom in Angstroms.
        charge
            Partial charge of the atom, by default 0.0.
        mass
            Atomic mass. If None, it uses mass_dic or defaults to 1.0.

        Returns
        -------
            The real index (ID) assigned to the new atom.
        """

        new_id = 1
        if not self.atoms.empty:
            new_id = self.atoms.index.max() + 1

        x, y, z = position

        new_row = {
            'type': str(atype),
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'charge': float(charge)
        }
        
        self.atoms.loc[new_id] = new_row
        self._cache = {}

        if atype not in self._masses_storage:
            # ---Update Masses ---
            if mass is not None:
                self._masses_storage[atype] = float(mass)
            elif atype in MASSES_DICT:
                self._masses_storage[atype] = MASSES_DICT[atype]
            else:
                # Default fallback if unknown
                self._masses_storage[atype] = 1.0

        return new_id

    def write(self, fout: str, atom_style: str='full', oldstyle: bool=False) -> None:
        """Write the system to a LAMMPS datafile.

        Parameters
        ----------
            d
                Input LAMMPS data
            fout
                Output file name (PDB, or LAMMPSData file)
            atom_style
                LAMMPS atom_style to adopt to format the output file

        """

        ext = os.path.splitext(fout)[1].lower()

        if ext == '.pdb':
            write_pdb(self, fout)
        else:
            write_lmp(self, fout, atom_style, oldstyle)

    def to_mda(self) -> mda.Universe:
        """
        Convert the current system to a MDAnalysis Universe object.
        """

        return to_mda(self)
    

    def to_pmg(self) -> Structure:
        """
        Convert the current system to a pymatgen Structure object.
        """

        return to_pmg(self)

    def replicate(self, factors: Sequence[int]) -> Self:
        """
        Replicate the system by integer factors along the lattice vectors (a, b, c).
        
        Parameters
        ----------
        factors :
            Number of copies along each lattice vector (nx, ny, nz).
        """
        nx, ny, nz = factors
        if nx == 1 and ny == 1 and nz == 1:
            return self

        # Prepare the basis vectors
        # lattice2vectors returns a (3, 3) matrix where each row is a vector v1, v2, v3
        vecs = np.array(lattice2vectors(self.box))
        v1, v2, v3 = vecs[0], vecs[1], vecs[2]

        original_atoms = self.atoms.copy()
        original_num_atoms = len(original_atoms)
        
        all_atoms = []
        
        # Dictionaries to store new interactions
        new_interactions = {
            'bonds': [], 'angles': [], 'dihedrals': [], 'impropers': []
        }

        # Replication loop
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if i == 0 and j == 0 and k == 0:
                        all_atoms.append(original_atoms)
                        continue
                    
                    # Calculate the shift for this cell
                    shift = i * v1 + j * v2 + k * v3
                    
                    # Copy atoms
                    new_atoms = original_atoms.copy()
                    new_atoms['x'] += shift[0]
                    new_atoms['y'] += shift[1]
                    new_atoms['z'] += shift[2]
                    
                    # Updating indices (IDs) to avoid duplicates
                    # shift the indexes by (number_atoms_orig *unique_multiplier)
                    offset = (i * ny * nz + j * nz + k) * original_num_atoms
                    new_atoms.index += offset
                    all_atoms.append(new_atoms)

                    # Replicate the interactions (Bonds, Angles, etc.)
                    for name in ['bonds', 'angles', 'dihedrals', 'impropers']:
                        df = getattr(self, name)
                        if df is not None and not df.empty:
                            new_df = df.copy()
                            # shift the index type of the interaction itself
                            new_df.index += (i * ny * nz + j * nz + k) * len(df)
                            
                            # shift the columns atom_1, atom_2, etc.
                            atom_cols = [c for c in df.columns if c.startswith('atom_')]
                            for col in atom_cols:
                                new_df[col] += offset
                            
                            new_interactions[name].append(new_df)

        # Merge Dataframes
        self.atoms = pd.concat(all_atoms)
        
        if self.velocities is not None:
            self.velocities = None

        for name in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if new_interactions[name]:
                full_list = [getattr(self, name)] + new_interactions[name]
                setattr(self, name, pd.concat(full_list))

        new_box = self.box.copy()
        new_box[0] *= nx
        new_box[1] *= ny
        new_box[2] *= nz
        self.set_box(new_box)

        return self

    def wrap(self) -> None:
            """
            Wraps all atoms back into the simulation box [0, L] using Periodic Boundary Conditions.
            Handles both orthogonal and triclinic boxes.
            """
            if self.atoms.empty:
                return self

            # Get the pass matrix (H_matrix)
            H = np.array(self._box_vectors)
            inv_H = np.linalg.inv(H)

            # Extract current coordinates (N, 3)
            coords = self.atoms[['x', 'y', 'z']].values

            # Conversion to fractional coordinates (0 to 1)
            # s = r . inv(H)
            frac_coords = np.dot(coords, inv_H)

            # Application of modulo 1.0
            # This brings everything into the range [0, 1[
            frac_coords %= 1.0

            # Return to Cartesian coordinates
            # r_new = s_new . H
            new_coords = np.dot(frac_coords, H)

            self.atoms[['x', 'y', 'z']] = new_coords
            
            return self

    def orthogonalize(self, tolerance: float=0.1, max_replica: int=10) -> bool:
        """
        Attempt to transform a triclinic (skewed) box into an orthogonal one.

        This method searches for new lattice vectors that align with the Cartesian 
        axes (X, Y, Z) by exploring linear combinations of the current basis 
        vectors. It then re-populates the new box by shifting and wrapping 
        existing atom coordinates.

        Parameters
        ----------
        tolerance
            The maximum deviation allowed from the Cartesian axes (in Å) for 
            a candidate vector to be considered orthogonal.
        max_replica
            The search range for linear combinations (m, n, o) of the original 
            lattice vectors. Higher values increase the chance of finding an 
            orthogonal cell but significantly increase computation time.

        Returns
        -------
        bool
            True if an orthogonal cell was successfully found and populated, 
            False otherwise. In case of failure, the box is 'unskewed' 
            (tilt factors removed) to maintain consistency.

        Notes
        -----
        - This operation will change the total number of atoms in the system.
        - Velocities and interactions (bonds, etc.) are typically lost or 
          invalidated during this specific geometric transformation.
        - The resulting box will have tilt factors (xy, xz, yz) set to zero.
        """

        H = np.array(lattice2vectors(self.box))
        new_vectors = []

        for i in range(3):
            best_v = None
            best_len = float('inf')
            
            for m in range(-max_replica, max_replica + 1):
                for n in range(-max_replica, max_replica + 1):
                    for o in range(-max_replica, max_replica + 1):
                        if m == 0 and n == 0 and o == 0: continue
                        
                        v_cand = m*H[0] + n*H[1] + o*H[2]
                        # check the alignment with the i axis
                        others = [v_cand[j] for j in range(3) if j != i]
                        
                        if all(abs(comp) < tolerance for comp in others):
                            v_len = abs(v_cand[i]) # We want the length on the axis
                            if 0.1 < v_len < best_len: # 0.1 to avoid zero vector
                                best_len = v_len
                                best_v = v_cand
            
            if best_v is None:
                print(f"Error: Unable to find an orthogonal vector for axis {i}")
                print(f"Unskew the box anyway")
                self.unskew() 
                return False
            new_vectors.append(best_v)

        diag_dim = [abs(new_vectors[0][0]), abs(new_vectors[1][1]), abs(new_vectors[2][2])]
        new_H = np.diag(diag_dim)
        
        search_range = np.arange(-max_replica, max_replica + 1)
        m_grid, n_grid, o_grid = np.meshgrid(search_range, search_range, search_range)
        translations = np.vstack([m_grid.ravel(), n_grid.ravel(), o_grid.ravel()]).T
        translation_vectors = translations @ H

        all_replicas = []
        eps = 1e-5

        for vec in translation_vectors:
  
            temp_df = self.atoms.copy()
            temp_df['x'] += vec[0]
            temp_df['y'] += vec[1]
            temp_df['z'] += vec[2]
            
            mask = (
                (temp_df['x'] >= -eps) & (temp_df['x'] < diag_dim[0] - eps) &
                (temp_df['y'] >= -eps) & (temp_df['y'] < diag_dim[1] - eps) &
                (temp_df['z'] >= -eps) & (temp_df['z'] < diag_dim[2] - eps)
            )
            if mask.any():
                all_replicas.append(temp_df[mask])
            
        if not all_replicas:
            return False

        self.atoms = pd.concat(all_replicas, ignore_index=True)
        
        self.atoms.index = range(1, len(self.atoms) + 1)
        if 'id' in self.atoms.columns:
            self.atoms['id'] = range(1, len(self.atoms) + 1)

        self.set_box(vectors2lattice(tuple(new_H)))
        
        return True

    def unskew(self) -> None:
        """
        Unskew the box if the non-diagonal parameters of the box matrix 
        :math:`|H_{i,j}| > H_{j,j}` for :math:`i \\neq j` by adding 
        :math:`\\pm H_{j,:}` to :math:`H_{i,:}` while :math:`|H_{i,j}| > H_{j,j}`.
        """

        boxm = np.array( lattice2vectors(self.box) )

        for i in range(3):
            for j in range(3):
                if i != j:
                    while boxm[i,j] > 0.5 * boxm[i,i]:
                        boxm[i] -= boxm[j]
                    while boxm[i,j] < - 0.5 * boxm[i,i]:
                        boxm[i] += boxm[j]

        self.set_box( vectors2lattice( (boxm[0], boxm[1], boxm[2]) ) )
        self.wrap()


    def view(self, trajectory=None):
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




 