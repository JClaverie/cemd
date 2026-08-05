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

import copy
from typing import Sequence, Any, Self, TYPE_CHECKING

import numpy as np
import pandas as pd

from ._edit import EditMixin
from ._topology import TopologyMixin
from ._io import IOMixin
from ._forcefield import ForceFieldMixin

from ._view import view

from .._constants import(
    AVOGADRO,
    MASSES_DICT, 
    INV_MASSES, 
    MASS_KEYS
)
from .._utils import (
    lammps2lattice, 
    lattice2vectors, 
)

if TYPE_CHECKING:
    from ..build import SolutionBuilder


class AtomicSystem(EditMixin, IOMixin, TopologyMixin, ForceFieldMixin):
    """
    Container storing the complete atomic system representation.

    The class stores atomic coordinates, topology information
    (bonds, angles, dihedrals and impropers), simulation box
    parameters, atomic masses and charges, and force field parameters.

    It is designed to represent the content of a LAMMPS data file and
    provides utilities for editing, analysing and exporting atomistic
    systems.

    Parameters
    ----------
    topology : dict[str, Any]
        Dictionary containing the complete system data.
        
        Required keys:
            - coordinates : np.ndarray
                Atomic coordinates (N, 3)
            - box : np.ndarray
                Simulation box dimensions (3,)
            - atom_types : list or np.ndarray
                Atom type indices
            - masses : list or np.ndarray
                Atomic masses
            - charges : list or np.ndarray
                Atomic charges
        
        Optional keys:
            - bonds : np.ndarray
                Bond connectivity (N_bonds, 2)
            - angles : np.ndarray
                Angle connectivity (N_angles, 3)
            - dihedrals : np.ndarray
                Dihedral connectivity (N_dihedrals, 4)
            - impropers : np.ndarray
                Improper connectivity (N_impropers, 4)
            - velocities : np.ndarray
                Atomic velocities (N, 3)
            - bond_params : dict
                Bond force field parameters
            - angle_params : dict
                Angle force field parameters
            - dihedral_params : dict
                Dihedral force field parameters
            - improper_params : dict
                Improper force field parameters

    Attributes
    ----------
    atoms : pandas.DataFrame
        Atomic information including coordinates, atom types and charges.
    bonds : pandas.DataFrame or None
        Bond topology information.
    angles : pandas.DataFrame or None
        Angle topology information.
    dihedrals : pandas.DataFrame or None
        Dihedral topology information.
    impropers : pandas.DataFrame or None
        Improper topology information.
    velocities : pandas.DataFrame or None
        Atomic velocities when available.
    pair_params : dict
        Pair interaction force field parameters.
    bond_params : dict
        Bond force field parameters.
    angle_params : dict
        Angle force field parameters.
    dihedral_params : dict
        Dihedral force field parameters.
    improper_params : dict
        Improper force field parameters.
    """

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
        tuple[float, float], # [xlo]
        tuple[float, float], # [yellow, this one]
        tuple[float, float], # awl, abode
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
        """
        Initialize an atomic system from topology data.

        Parameters
        ----------
        topology : dict[str, Any]
            Dictionary containing all atomic, topological and simulation
            box information.
        """
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

        self.pair_params = topology.get('pair_params', {})
        self.bond_params = topology.get('bond_params', {})
        self.angle_params = topology.get('angle_params', {})
        self.dihedral_params = topology.get('dihedral_params', {})
        self.improper_params =topology.get('improper_params', {})

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
        """Replaces internal content with that of another system, without breaking references."""
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
        """
        Create an independent deep copy of the atomic system.

        The copied object contains duplicated atomic coordinates,
        topology tables, simulation box information and force field
        parameters.

        Returns
        -------
        AtomicSystem
            Independent copy of the current system.
        """
        new = self.__class__.__new__(self.__class__)
        
        topology = {
            'atoms':     self.atoms.copy(),
            'bonds':     self.bonds.copy()     if self.bonds     is not None else None,
            'angles':    self.angles.copy()    if self.angles    is not None else None,
            'dihedrals': self.dihedrals.copy() if self.dihedrals is not None else None,
            'impropers': self.impropers.copy() if self.impropers is not None else None,
            'velocities':self.velocities.copy()if self.velocities is not None else None,
            'lmp_box':   self._lmp_box,
            'atom_types': list(self._types),
            'masses':    dict(self._masses_storage),
            'charges':   dict(self._charges_storage),
            'atom_style': self._atom_style,
        }
        
        new._assign_topology(topology)
        new._finalize_data()
        
        # Copying force field settings if defined
        new.pair_params     = dict(self.pair_params)
        new.bond_params     = dict(self.bond_params)
        new.angle_params    = dict(self.angle_params)
        new.dihedral_params = dict(self.dihedral_params)
        new.improper_params = dict(self.improper_params)
        
        if self._pmg_struct is not None:
            new._pmg_struct = copy.deepcopy(self._pmg_struct)

        return new

    @property
    def box(self) -> np.ndarray:
        """
        Return the lattice parameters.

        Returns
        -------
        numpy.ndarray
            The box lattice parameters.
        """
        return copy.copy(self._box)
        
    @property
    def volume(self) -> float:
        """
        Return the volume of the simulation box.

        Returns
        -------
        float
            Box volume in Å³.
        """
        v1, v2, v3 = self._box_vectors
        return np.dot(v1, np.cross(v2, v3))

    @property
    def masses(self) -> list[float]:
        """
        Return atomic masses associated with atom types.

        Returns
        -------
        list of float
            Mass of each atom type.
        """
        if 'masses' not in self._cache:
            mass_list = [
                float(self._masses_storage.get(t, MASSES_DICT.get(t, 1.0)))
                for t in self.atom_types
            ]
            self._cache['masses'] = mass_list
        return self._cache['masses']

    @property
    def charges(self) -> list[float]:
        """
        Return charges associated with atom types.

        Returns
        -------
        list of float
            Charge of each atom type.
        """
        if 'charges' not in self._cache:
            self._cache['charges'] = [
                float(self._charges_storage.get(atype, 0))
                for atype in self.atom_types
            ]
        return self._cache['charges']

    @property
    def elements(self) -> dict[str | int , str]:
        """
        Return a mapping of atom types to their elemental symbols.

        Returns
        -------
        dict of {str or int : str}
            Dictionary matching type ID to element symbol.
        """
        
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
    def atom_types(self) -> list[str | int]:
        """
        Return the list of unique atom types.

        Returns
        -------
        list of str or int
            Sorted list of atom types.
        """
        if 'atom_types' not in self._cache:
            self._cache['atom_types'] = sorted([str(t) for t in self.atoms.type.unique()])
        return self._cache['atom_types']

    @property
    def bond_types(self) -> list[str | int]:
        """
        Return the list of unique bond types.

        Returns
        -------
        list of str or int
            Sorted list of bond types.
        """
        if self.bonds is not None:
            return sorted(self.bonds.type.unique().tolist())
        else:
            return list()

    @property
    def angle_types(self) -> list[str | int]:
        """
        Return the list of unique angle types.

        Returns
        -------
        list of str or int
            Sorted list of angle types.
        """
        if self.angles is not None:
            return sorted(self.angles.type.unique().tolist())
        else:
            return list()

    @property
    def dihedral_types(self) -> list[str | int]:
        """
        Return the list of unique dihedral types.

        Returns
        -------
        list of str or int
            Sorted list of dihedral types.
        """
        if self.dihedrals is not None:
            return sorted(self.dihedrals.type.unique().tolist())
        else:
            return list()

    @property
    def improper_types(self) -> list[str | int]:
        """
        Return the list of unique improper types.

        Returns
        -------
        list of str or int
            Sorted list of improper types.
        """
        if self.impropers is not None:
            return sorted(self.impropers.type.unique().tolist())
        else:
            return list()

    @property
    def num_atoms(self) -> int:
        """
        Return the number of atoms.

        Returns
        -------
        int
            Number of atoms in the system.
        """
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """
        Return the number of bonds.

        Returns
        -------
        int
            Total number of bonds.
        """
        if self.bonds is None:
            return 0
        else:
            return len(self.bonds)
    @property
    def num_angles(self) -> int:
        """
        Return the number of angles.

        Returns
        -------
        int
            Total number of angles.
        """
        if self.angles is None:
            return 0
        else:
            return len(self.angles)
    @property
    def num_dihedrals(self) -> int:
        """
        Return the number of dihedrals.

        Returns
        -------
        int
            Total number of dihedrals.
        """
        if self.dihedrals is None:
            return 0
        else:
            return len(self.dihedrals)
    @property
    def num_impropers(self) -> int:
        """
        Return the number of impropers.

        Returns
        -------
        int
            Total number of impropers.
        """
        if self.impropers is None:
            return 0
        else:
            return len(self.impropers)

    @property
    def num_atom_types(self) -> int:
        """
        Return the number of atom types.

        Returns
        -------
        int
            Count of distinct atom types.
        """
        return len(self.atom_types)
    @property
    def num_bond_types(self) -> int:
        """
        Return the number of bond types.

        Returns
        -------
        int
            Count of distinct bond types.
        """
        return len(self.bond_types)
    @property
    def num_angle_types(self) -> int:
        """
        Return the number of angle types.

        Returns
        -------
        int
            Count of distinct angle types.
        """
        return len(self.angle_types)
    @property
    def num_dihedral_types(self) -> int:
        """
        Return the number of dihedral types.

        Returns
        -------
        int
            Count of distinct dihedral types.
        """
        return len(self.dihedral_types)
    @property
    def num_improper_types(self) -> int:
        """
        Return the number of improper types.

        Returns
        -------
        int
            Count of distinct improper types.
        """
        return len(self.improper_types)

    @property
    def total_charge(self) -> float:
        """
        Return the total system charge.

        Returns
        -------
        float
            Sum of all atomic charges.
        """
        return self.atoms.charge.sum()

    @property
    def total_mass(self) -> float:
        """
        Return the total system mass.

        Returns
        -------
        float
            Total mass calculated from atomic types and counts.
        """
        counts = self.atoms['type'].value_counts()
        total_mass = sum(counts.get(atype, 0) * mass for atype, mass in zip(self.atom_types, self.masses))
        return total_mass

    @property
    def density(self) -> float:
        """
        Return the system density.

        Returns
        -------
        float
            Density in g/cm³.
        """
        return self.total_mass / AVOGADRO / self.volume / 1e-24

    def _get_type_summary(self) -> pd.DataFrame:
        """
        Return a summarized DataFrame of atom types, numbers, masses and charges.
        """
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
        """
        Return a string representation of the AtomicSystem.

        Returns
        -------
        str
            Summary of box size, atom counts, charge and density.
        """
        
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
        red_df = self._get_type_summary()
        
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
        """
        Return the string representation of the object.

        Returns
        -------
        str
            Equivalent to __repr__.
        """
        return self.__repr__()

    def add_structure(self, 
                      structure_to_add: AtomicSystem,
                      distance: float = 2.0, 
                      axis: str = 'z',
                      vacuum: float = 10.0) -> Self:
        """
        Add a structure on top of this system.

        This method adds any structure (droplet, liquid layer, molecule, etc.)
        on top of the current system along the specified axis. The structure is
        aligned by center of mass in the transverse directions and placed at the
        specified distance from the surface.

        Parameters
        ----------
        structure_to_add : AtomicSystem or str
            The structure to add. Can be an AtomicSystem object or a path to a
            file that can be loaded by AtomicSystem.from_file().
        distance : float, default=2.0
            Distance between the current system surface and the structure
            in Ångströms.
        axis : str, default='z'
            Axis along which to add the structure ('x', 'y', or 'z').
        vacuum : float, default=10.0
            Vacuum space added above the structure in Ångströms.

        Returns
        -------
        Self
            The updated system with the structure added.

        Examples
        --------
        >>> from cemd import AtomicSystem
        >>> 
        >>> # Add a droplet from a file
        >>> surface = AtomicSystem.from_file("surface.lmp")
        >>> droplet = AtomicSystem.from_file("droplet.lmp")
        >>> system = surface.add_structure(droplet, distance=2.0, vacuum=10.0)
        >>> 
        >>> # Add a structure from a file path (auto-loads)
        >>> system = surface.add_structure("droplet.lmp", distance=2.0)
        >>> 
        >>> # Add a custom structure with different axis
        >>> system = surface.add_structure(molecule, axis='y', distance=3.0)
        """
        from ..build import add_structure
        
        new_system = add_structure(
            solid_system=self,
            structure_to_add=structure_to_add,
            distance=distance,
            axis=axis,
            vacuum=vacuum
        )
        self._replace_internals(new_system)
        
        return self

    def add_layer(self, blueprint: SolutionBuilder, thickness: float,
                  distance: float = 2.0, vacuum: float = 10.0, axis: str = 'z') -> Self:
        """
        Add a liquid layer on top of this system.

        This method creates a liquid layer with the composition defined by the
        blueprint and places it on top of the current system along the specified
        axis. The layer is aligned to match the transverse dimensions of the
        current system.

        Parameters
        ----------
        blueprint : SolutionBuilder
            Solution blueprint defining the liquid composition (density,
            solutes, etc.).
        thickness : float
            Thickness of the liquid layer in Ångströms.
        distance : float, default=2.0
            Distance between the current system surface and the liquid layer
            in Ångströms.
        vacuum : float, default=10.0
            Vacuum space added above the liquid layer in Ångströms.
        axis : str, default='z'
            Axis along which to add the layer ('x', 'y', or 'z').

        Returns
        -------
        Self
            The updated system with the liquid layer added.

        Examples
        --------
        >>> from cemd import AtomicSystem
        >>> from cemd.builders import SolutionBuilder
        >>> 
        >>> surface = AtomicSystem.from_file("surface.lmp")
        >>> blueprint = SolutionBuilder.from_molarities(
        ...     density=1.0,
        ...     molarities={'NaCl': 0.1}
        ... )
        >>> 
        >>> # Add a 30 Å water layer on the surface
        >>> system = surface.add_layer(blueprint, thickness=30.0, distance=2.0)
        """
        from ..build import add_liquid_layer
        new_system = add_liquid_layer(
            self, blueprint, thickness, distance, vacuum, axis
        )
        self._replace_internals(new_system)

        return self
    
    def add_droplet(self, blueprint: SolutionBuilder, radius: float,
                    distance: float = 2.0, vacuum: float = 10.0, axis: str = 'z') -> Self:
        """
        Add a hemispherical liquid droplet on top of this system.

        This method creates a hemispherical droplet with the composition defined
        by the blueprint and places it on top of the current system along the
        specified axis. The droplet sits on the surface with a flat bottom.

        Parameters
        ----------
        blueprint : SolutionBuilder
            Solution blueprint defining the droplet composition (density,
            solutes, etc.).
        radius : float
            Radius of the hemispherical droplet in Ångströms.
        distance : float, default=2.0
            Distance between the current system surface and the droplet
            in Ångströms.
        vacuum : float, default=10.0
            Vacuum space added above the droplet in Ångströms.
        axis : str, default='z'
            Axis along which the droplet sits ('x', 'y', or 'z').

        Returns
        -------
        Self
            The updated system with the droplet added.

        Examples
        --------
        >>> from cemd import AtomicSystem
        >>> from cemd.builders import SolutionBuilder
        >>> 
        >>> surface = AtomicSystem.from_file("surface.lmp")
        >>> blueprint = SolutionBuilder.from_molarities(
        ...     density=1.0,
        ...     molarities={'NaCl': 0.1}
        ... )
        >>> 
        >>> # Add a 15 Å radius water droplet on the surface
        >>> system = surface.add_droplet(blueprint, radius=15.0, distance=2.0)
        """
        from ..build import add_droplet
        new_system = add_droplet(
            self, blueprint, radius, distance, vacuum, axis
        )
        self._replace_internals(new_system)

        return self
    
    def get_count(self, symbol: str | int) -> int:
        """
        Calculate the exact number of atoms for a specific type.

        Parameters
        ----------
        symbol : str or int
            The atom type identifier to count.

        Returns
        -------
        int
            The number of atoms of the specified type.

        Raises
        ------
        ValueError
            If the symbol is not found.
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

    def view(self, trajectory: str=None) -> None:
        """
        Visualize the current system in VMD, with an optional trajectory.

        Parameters
        ----------
        trajectory : str, optional
            Path to a trajectory file to overlay onto this 
            system's topology. All formats supported by MDAnalysis can be used. Defaults to None.

        Examples
        --------
        >>> system = AtomicSystem("input.data")
        >>> system.view()
        >>> system.view(trajectory="production.dcd")
        """

        view(self, trajectory=trajectory)




 