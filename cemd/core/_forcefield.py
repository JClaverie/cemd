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
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from .._config import FF_DATABASE_FILE

if TYPE_CHECKING:
    from .atomic_system import AtomicSystem

class ForceFieldMixin:

    def set_masses(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """
        Set the masses for the atomic system.

        Parameters
        ----------
        value : Sequence of float or dict of {str or int : float}
            If a sequence is provided, it must match the current order of 
            `atom_types`. If a dictionary is provided, it maps specific 
            atom types to their corresponding masses.
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
        Set or update the charges for the atomic system and the atoms DataFrame.

        Parameters
        ----------
        value : Sequence of float or dict of {str or int : float}
            If a sequence is provided, it must match the current order of 
            `atom_types`. If a dictionary is provided, it performs a 
            partial or full update of the charge mapping.
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

        self._set_pair_forcefield(assignments, df_lj)
        self._set_bond_forcefield(assignments, df_bond)
        self._set_angle_forcefield(assignments, df_angle)

    def set_atom_type_param(self: AtomicSystem, 
                            atom_type: str | int, 
                            pair_coeffs: list[float]) -> None:
        """
        Assign the non-bond (Lennard-Jones) parameters for a specific atom type.

        Parameters
        ----------
        atom_type : str or int
            The identifier of the atom type.
        pair_coeffs : list of float
            The L-J coefficients (typically [epsilon, sigma]).

        Raises
        ------
        ValueError
            If the atom_type is not present in the system's current atoms.
        """
        # Dynamic extraction of real types present in the DataFrame
        if atom_type not in self.atoms['type'].unique():
            raise ValueError(f"Atom type '{atom_type}' does not exist in the system.")
        
        sorted_key = (atom_type, atom_type)
        self.pair_params[sorted_key] = pair_coeffs

    def set_bond_type_param(self: AtomicSystem, 
                            bond_type: str, 
                            coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific bond type.

        Parameters
        ----------
        bond_type : str
            The identifier of the bond type (e.g., 'H-O').
        coeffs : list of float
            The bond coefficients (typically [k, r0]).

        Raises
        ------
        ValueError
            If the bond_type is not initialized or does not exist.
        """
        if not hasattr(self, 'bond_types') or self.bond_types is None:
            raise ValueError("The system does not have any bond types initialized.")
        if bond_type not in self.bond_types:
            raise ValueError(f"Bond type '{bond_type}' does not exist in the system.")
        
        self.bond_params[bond_type] = coeffs

    def set_angle_type_param(self: AtomicSystem, 
                            angle_type: str, 
                            coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific angle type.

        Parameters
        ----------
        angle_type : str
            The identifier of the angle type (e.g., 'H-O-H').
        coeffs : list of float
            The angle coefficients (typically [k, theta0]).

        Raises
        ------
        ValueError
            If the angle_type is not initialized or does not exist.
        """
        if not hasattr(self, 'angle_types') or self.angle_types is None:
            raise ValueError("The system does not have any angle types initialized.")
        if angle_type not in self.angle_types:
            raise ValueError(f"Angle type '{angle_type}' does not exist in the system.")
        
        self.angle_params[angle_type] = coeffs

    def _set_pair_forcefield(
        self: AtomicSystem, 
        atom_type_assignments: dict[str | int, str], 
        df_lj: pd.DataFrame) -> None:
        """
        Automatically searches the Lennard-Jones database for assigned types,
        extracts self-interaction parameters, applies them via 'set_atom_type_param',
        and registers all cross-interaction (i != j) terms.
        
        Parameters
        ----------
        system : AtomicSystem
            The atomic system to update.
        atom_type_assignments : dict
            Mapping of {system_atom_label: forcefield_database_type} 
            (e.g., {'ob': 'ob_clayff', 'ho': 'ho_clayff'}).
        df_lj : pd.DataFrame
            The raw Lennard-Jones spreadsheet DataFrame ('lj_12-6').
        """
        if df_lj is None or df_lj.empty:
            return

        # First, clear previous pair parameters to start fresh
        self.pair_params = {}

        # Generate ALL possible pairs from the system types (self and cross-interactions)
        for label1, label2 in combinations_with_replacement(atom_type_assignments.keys(), 2):
            ff_t1 = atom_type_assignments[label1]
            ff_t2 = atom_type_assignments[label2]
            
            # Bi-directional search in the Excel sheet columns 'type 1' and 'type 2'
            mask = ((df_lj['type 1'] == ff_t1) & (df_lj['type 2'] == ff_t2)) | \
                ((df_lj['type 1'] == ff_t2) & (df_lj['type 2'] == ff_t1))
            row = df_lj[mask]
            
            if not row.empty:
                # Column 2 and 3 usually represent epsilon and sigma in your sheet
                pair_coeffs = [float(row.iloc[0].iloc[2]), float(row.iloc[0].iloc[3])]
                
                if label1 == label2:
                    # LEVEL 2 CALL: Assign charge and non-bond parameters for this specific single type
                    self.set_atom_type_param(label1, pair_coeffs)
                else:
                    # Custom cross-interaction (i != j) stored as a sorted label tuple key
                    sorted_key = tuple(sorted([label1, label2]))
                    self.pair_params[sorted_key] = pair_coeffs

    def _set_bond_forcefield(
        self: AtomicSystem, 
        atom_type_assignments: dict[str | int, str], 
        df_bond: pd.DataFrame) -> None:
        """
        Automatically maps system bond labels (e.g., 'H-O') to their assigned
        forcefield types (e.g., 'hspc-ospc') using atom_type_assignments, checks the 
        database bidirectionally, and applies the parameters.
        
        Parameters
        ----------
        system : AtomicSystem
            The atomic system containing 'bond_types'.
        atom_type_assignments : dict
            Mapping of {system_atom_label: forcefield_database_type} 
            (e.g., {'O': 'ospc', 'H': 'hspc'}).
        df_bond : pd.DataFrame
            The raw bond spreadsheet DataFrame.
        """
        if df_bond is None or df_bond.empty or not hasattr(self, 'bond_types') or self.bond_types is None:
            return

        # Cleaning database columns and chains
        df_bond_clean = df_bond.copy()
        df_bond_clean.columns = df_bond_clean.columns.str.strip()
        df_bond_clean['type 1'] = df_bond_clean['type 1'].astype(str).str.strip()
        df_bond_clean['type 2'] = df_bond_clean['type 2'].astype(str).str.strip()

        self.bond_params = {}

        for bond_str in self.bond_types:
            elements = bond_str.split('-')
            if len(elements) != 2: 
                continue
            
            sys_t1, sys_t2 = elements[0].strip(), elements[1].strip()
            
            # REMAPPING: Translation of system types to forcefield types
            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2)
            
            # If one of the two atoms has not received a forcefield assignment, we cannot search
            if not ff_t1 or not ff_t2:
                continue
                
            # Bidirectional database search with remapped types
            mask = ((df_bond_clean['type 1'] == ff_t1) & (df_bond_clean['type 2'] == ff_t2)) | \
                ((df_bond_clean['type 1'] == ff_t2) & (df_bond_clean['type 2'] == ff_t1))
            row = df_bond_clean[mask]
            
            if not row.empty:
                # Extraction of constants (k and r)
                coeffs = [float(row.iloc[0].iloc[2]), float(row.iloc[0].iloc[3])]
                
                # Applies the settings to the original system label (e.g. 'H-O')
                self.set_bond_type_param(bond_str, coeffs)

    def _set_angle_forcefield(
        self: AtomicSystem, 
        atom_type_assignments: dict[str | int, str], 
        df_angle: pd.DataFrame) -> None:
        """
        Automatically maps system angle labels (e.g., 'H-O-H') to their assigned
        forcefield types, checks the database bidirectionally (outer atoms), and applies parameters.
        """
        if df_angle is None or df_angle.empty or not hasattr(self, 'angle_types') or self.angle_types is None:
            return

        df_angle_clean = df_angle.copy()
        df_angle_clean.columns = df_angle_clean.columns.str.strip()
        df_angle_clean['type 1'] = df_angle_clean['type 1'].astype(str).str.strip()
        df_angle_clean['type 2'] = df_angle_clean['type 2'].astype(str).str.strip()
        df_angle_clean['type 3'] = df_angle_clean['type 3'].astype(str).str.strip()

        self.angle_params = {}

        for angle_str in self.angle_types:
            elements = angle_str.split('-')
            if len(elements) != 3: 
                continue
            
            sys_t1, sys_t2, sys_t3 = elements[0].strip(), elements[1].strip(), elements[2].strip()
            
            # Translation on the fly via the dictionary
            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2) # Atome central
            ff_t3 = atom_type_assignments.get(sys_t3)
            
            if not ff_t1 or not ff_t2 or not ff_t3:
                continue
                
            # Type 2 (central) remains in the middle, reading direction 1-3 can be reversed
            mask = ((df_angle_clean['type 1'] == ff_t1) & (df_angle_clean['type 2'] == ff_t2) & (df_angle_clean['type 3'] == ff_t3)) | \
                ((df_angle_clean['type 1'] == ff_t3) & (df_angle_clean['type 2'] == ff_t2) & (df_angle_clean['type 3'] == ff_t1))
            row = df_angle_clean[mask]
            
            if not row.empty:
                coeffs = [float(row.iloc[0].iloc[3]), float(row.iloc[0].iloc[4])]
                self.set_angle_type_param(angle_str, coeffs)
