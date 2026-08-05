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

from typing import TYPE_CHECKING, Sequence
from itertools import combinations_with_replacement, combinations

import numpy as np

from ..._data import get_ff_db_path
from .forcefield_database import ForceFieldDatabase
from .models import (
    LJParams, BuckinghamParams, BondParams, AngleParams
)

if TYPE_CHECKING:
    from ..atomic_system import AtomicSystem

class ForceFieldMixin:
    """Mixin for force field operations on AtomicSystem."""

    def set_ff_from_database(self, 
                             assignments: dict[str | int, str], ff_database_dir: str = None) -> None:
        """
        Apply force field parameters from the TOML database.
        """
        if ff_database_dir is None:
            ff_database_dir = get_ff_db_path()
        
        db = ForceFieldDatabase(ff_database_dir)
        
        # Resolve assignments to complete types (explicit format)
        resolved_assignments = {}
        for sys_type, db_type in assignments.items():
            # If already a full_type, we keep it
            if '.' in db_type:
                resolved_assignments[sys_type] = db_type
            else:
                # Search type in all models
                found = None
                for full_type in db.atom_types.keys():
                    if full_type.endswith(f'.{db_type}'):
                        found = full_type
                        break
                if found is None:
                    print(f"⚠️ Type '{db_type}' not found in any model")
                    continue
                resolved_assignments[sys_type] = found
        
        # Apply settings
        self._set_pair_params_from_db(resolved_assignments, db)
        self._set_bond_params_from_db(resolved_assignments, db)
        self._set_angle_params_from_db(resolved_assignments, db)
        self._update_masses_and_charges(resolved_assignments, db)

    def explore_ff_database(self, ff_database_dir: str = None, visible_rows: int = 20) -> None:
        """
        Interactive Forcefield Explorer using the prompt_toolkit UI pattern.
        
        Launches an interactive terminal user interface (TUI) that allows the user
        to browse and select force field parameters for each element and atom type
        present in the system. The explorer reads a force field database from an
        TOML files and presents matching parameter sets for user selection via
        keyboard navigation.
        
        Parameters
        ----------
        ff_database_dir : str, optional
            Path to the directory containing the force field database.
            The file must contain a sheet named 'list' with force field parameters.
            Defaults to get_ff_db_path() (global constant).
        
        visible_rows : int, optional
            Number of rows visible in the interactive table. Default is 20.
        
        Returns
        -------
        None
            The method updates the system's force field parameters in-place via
            successive calls to `set_ff_from_database()` as each selection is made.
        
        Raises
        ------
        FileNotFoundError
            If the database file does not exist.
        ValueError
            If required sheets are missing from the database.
        
        Notes
        -----
        The interactive UI provides:
            -Up/Down arrows: Navigate through parameter options
            -Enter/Space: Select the highlighted parameter
            -Ctrl+C/Escape/q: Cancel selection and skip element
            -p: Open the reference DOI in a web browser
        
        The function filters the database for parameters matching each element
        present in the system (self.elements). For each unique element, the user is
        presented with a list of available parameter sets to choose from.
        
        Examples
        --------
        >>> system = AtomicSystem(...)
        >>> system.explore_ff_database()
        Assigned: H -> clayff.h_star
        Assigned: O -> clayff.o_star
        
        >>> # Using custom database
        >>> system.explore_ff_database('/path/to/database')
        """

        from prompt_toolkit import Application
        from prompt_toolkit.layout import Layout, HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import ScrollOffsets
        import webbrowser
        
        if ff_database_dir is None:
            ff_database_dir = get_ff_db_path()
        
        db = ForceFieldDatabase(ff_database_dir)
        dfs = db.to_dataframes()
        
        if "list" not in dfs or dfs["list"].empty:
            print("No atom types found in database.")
            return
        
        df_list = dfs["list"]
        
        assignments = {}
        system_elements = self.elements
        system_types = self.atom_types
        
        for el, t in zip(system_elements, system_types):
            subset = df_list[df_list['element'] == el]
            if subset.empty:
                print(f"No parameters for element {el}. Skipping.")
                continue
            
            types = []
            for _, row in subset.iterrows():
                types.append({
                    "type": row["type"],
                    "full_type": row["full_type"],
                    "model": row["model"],
                    "environment": row["environment"],
                    "ref": row["ref"],
                })
            
            index = 0
            selected = None
            scroll_top = 0
            kb = KeyBindings()
            
            ROW_FORMAT = "{cursor} {type:<20} {model:<15} | {env:<45.45}"
            
            def render():
                header = ROW_FORMAT.format(cursor=" ", type="TYPE", model="MODEL", env="ENVIRONMENT")
                separator = "-" * len(header)
                
                lines = [
                    f"--- Select Forcefield Parameters for type {t} [{index+1}/{len(types)}] ---",
                    f"Found {len(types)} options  [{index+1}/{len(types)}]",
                    "↑↓ Move   [Enter] Assign   [p] Open DOI   [q] Quit",
                    "",
                    header,
                    separator
                ]
                
                # Show only the visible window
                visible = types[scroll_top:scroll_top + visible_rows]
                
                for i, r in enumerate(visible):
                    real_index = scroll_top + i
                    cursor = "➜" if real_index == index else " "
                    lines.append(ROW_FORMAT.format(
                        cursor=cursor,
                        type=r['type'],
                        model=r['model'],
                        env=r['environment'][:45] if r['environment'] else ""
                    ))
                
                # Scroll indicator
                if len(types) > visible_rows:
                    lines.append(f"\n  ↕ {scroll_top+1}-{min(scroll_top+visible_rows, len(types))} of {len(types)}")
                
                return "\n".join(lines)
            
            text = FormattedTextControl(render)
            window = Window(
                content=text,
                always_hide_cursor=True,
                scroll_offsets=ScrollOffsets(top=1, bottom=1),
                allow_scroll_beyond_bottom=False,
                dont_extend_height=False,
            )
            
            @kb.add("up")
            def _(e):
                nonlocal index, scroll_top
                if index > 0:
                    index -= 1
                    if index < scroll_top:
                        scroll_top = index
                    e.app.invalidate()
            
            @kb.add("down")
            def _(e):
                nonlocal index, scroll_top
                if index < len(types) - 1:
                    index += 1
                    if index >= scroll_top + visible_rows:
                        scroll_top = index - visible_rows + 1
                    e.app.invalidate()
            
            @kb.add("enter")
            def _(e):
                nonlocal selected
                selected = types[index]
                e.app.exit()
            
            @kb.add("p")
            def _(e):
                ref = types[index].get('ref', '')
                if ref and ref.startswith('http'):
                    webbrowser.open(ref)
            
            @kb.add("q")
            @kb.add("escape")
            def _(e):
                e.app.exit()
            
            app = Application(
                layout=Layout(HSplit([window])),
                key_bindings=kb,
                full_screen=True,
            )
            
            app.run()
            
            if selected:
                assignments[t] = selected['full_type']
                print(f"Assigned: {t} -> {selected['full_type']} ({selected['model']})")
                self.set_ff_from_database(assignments, ff_database_dir)
            else:
                print("Selection cancelled.")
                break

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
                traceback.print_stack()  # ← to see where the call is coming from
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

    def _validate_and_convert_coeffs(self, coeffs, expected_len, name):
        """Validates and converts a list of coefficients."""
        if len(coeffs) != expected_len:
            raise ValueError(
                f"Invalid number of coefficients for {name}: "
                f"expected {expected_len}, got {len(coeffs)}."
            )
        try:
            return [float(c) for c in coeffs]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid coefficient values for {name}: "
                f"expected numbers, got {coeffs}"
            ) from e

    def set_pair_params(self: AtomicSystem, 
                            atom_type1: str | int, 
                            atom_type2: str | int = None,
                            coeffs: list[float] = None,
                            potential_type: str = 'lj') -> None:
        """
        Assign non-bond parameters for a pair of atom types (self or cross).

        Parameters
        ----------
        atom_type1 : str or int
            The identifier of the first atom type.
        atom_type2 : str or int, optional
            The identifier of the second atom type. If None, assumes self-interaction.
        coeffs : list of float
            The potential coefficients.
            -LJ: [epsilon (kcal/mol), sigma (Å)]
            -Buckingham: [A (kcal/mol), rho (Å), C (kcal/mol·Å⁶)]
        potential_type : str, optional
            Type of potential: 'lj' (Lennard-Jones) or 'buckingham'. Default is 'lj'.

        Raises
        ------
        ValueError
            If the atom types are not present in the system or coeffs length is invalid.
        """
        # Si atom_type2 est None, on fait une auto-interaction
        if atom_type2 is None:
            atom_type2 = atom_type1
        
        # Check that both types exist
        if atom_type1 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type1}' does not exist in the system.")
        if atom_type2 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type2}' does not exist in the system.")
        
        sorted_key = tuple(sorted([atom_type1, atom_type2]))
        
        if potential_type == 'lj':
            eps, sigma = self._validate_and_convert_coeffs(
                coeffs, 2, 
                f"LJ potential for pair '{atom_type1}-{atom_type2}'"
            )
            self.pair_params[sorted_key] = LJParams(
                epsilon=eps,
                sigma=sigma,
            )
        elif potential_type == 'buckingham':
            A, rho, C = self._validate_and_convert_coeffs(
                coeffs, 3, 
                f"Buckingham potential for pair '{atom_type1}-{atom_type2}'"
            )
            self.pair_params[sorted_key] = BuckinghamParams(
                A=A,
                rho=rho,
                C=C,
            )
        else:
            raise ValueError(f"Unknown potential type: {potential_type}. Use 'lj' or 'buckingham'.")


    def set_bond_params(self: AtomicSystem, 
                            bond_type: str, 
                            coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific bond type.

        Parameters
        ----------
        bond_type : str
            The identifier of the bond type (e.g., 'H-O').
        coeffs : list of float
            The bond coefficients [k (kcal/(mol·Å²)), r0 (Å)].

        Raises
        ------
        ValueError
            If the bond_type is not initialized or does not exist.
            If coeffs length is invalid.
        """
        if not hasattr(self, 'bond_types') or self.bond_types is None:
            raise ValueError("The system does not have any bond types initialized.")
        if bond_type not in self.bond_types:
            raise ValueError(f"Bond type '{bond_type}' does not exist in the system.")
        
        k, r0 = self._validate_and_convert_coeffs(coeffs, 2, f"bond '{bond_type}'")
        
        if k <= 0:
            raise ValueError(f"k must be positive for bond '{bond_type}', got {k}")
        if r0 <= 0:
            raise ValueError(f"r0 must be positive for bond '{bond_type}', got {r0}")
        
        self.bond_params[bond_type] = BondParams(k=k, r0=r0)


    def set_angle_params(self: AtomicSystem, 
                            angle_type: str, 
                            coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific angle type.

        Parameters
        ----------
        angle_type : str
            The identifier of the angle type (e.g., 'H-O-H').
        coeffs : list of float
            The angle coefficients [k (kcal/(mol·rad²)), theta0 (deg)].

        Raises
        ------
        ValueError
            If the angle_type is not initialized or does not exist.
            If coeffs length is invalid.
        """
        if not hasattr(self, 'angle_types') or self.angle_types is None:
            raise ValueError("The system does not have any angle types initialized.")
        if angle_type not in self.angle_types:
            raise ValueError(f"Angle type '{angle_type}' does not exist in the system.")
        
        k, theta0 = self._validate_and_convert_coeffs(coeffs, 2, f"angle '{angle_type}'")
        
        if k <= 0:
            raise ValueError(f"k must be positive for angle '{angle_type}', got {k}")
        if theta0 < 0 or theta0 > 180:
            raise ValueError(
                f"theta0 must be between 0 and 180 for angle '{angle_type}', got {theta0}"
            )
        
        self.angle_params[angle_type] = AngleParams(k=k, theta0=theta0)


    def set_dihedral_params(self: AtomicSystem, 
                                dihedral_type: str, 
                                coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific dihedral type.

        Parameters
        ----------
        dihedral_type : str
            The identifier of the dihedral type (e.g., 'C-C-C-C').
        coeffs : list of float
            The dihedral coefficients. Number depends on the dihedral style.

        Raises
        ------
        ValueError
            If the dihedral_type is not initialized or does not exist.
            If coeffs length is invalid.
        """
        if not hasattr(self, 'dihedral_types') or self.dihedral_types is None:
            raise ValueError("The system does not have any dihedral types initialized.")
        if dihedral_type not in self.dihedral_types:
            raise ValueError(f"Dihedral type '{dihedral_type}' does not exist in the system.")
        
        # For dihedrals, we do not validate the number of coeffs because it depends on the style
        # (OPLS: 3 coeffs, CHARMM: 6 coeffs, etc.)
        try:
            coeffs = [float(c) for c in coeffs]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid coefficient values for dihedral '{dihedral_type}': "
                f"expected numbers, got {coeffs}"
            ) from e
        
        self.dihedral_params[dihedral_type] = coeffs  # Or a dataclass if defined


    def set_improper_params(self: AtomicSystem, 
                                improper_type: str, 
                                coeffs: list[float]) -> None:
        """
        Assign structural parameters for a specific improper type.

        Parameters
        ----------
        improper_type : str
            The identifier of the improper type.
        coeffs : list of float
            The improper coefficients [k, theta0].

        Raises
        ------
        ValueError
            If the improper_type is not initialized or does not exist.
            If coeffs length is invalid.
        """
        if not hasattr(self, 'improper_types') or self.improper_types is None:
            raise ValueError("The system does not have any improper types initialized.")
        if improper_type not in self.improper_types:
            raise ValueError(f"Improper type '{improper_type}' does not exist in the system.")
        
        k, theta0 = self._validate_and_convert_coeffs(coeffs, 2, f"improper '{improper_type}'")
        
        if k <= 0:
            raise ValueError(f"k must be positive for improper '{improper_type}', got {k}")
        
        self.improper_params[improper_type] = AngleParams(k=k, theta0=theta0)


    def apply_pair_mixing_rules(self, rule='arithmetic', overwrite=False) -> None:
        """
        Calculate and update cross-interaction parameters (i != j).

        This method estimates parameters for pair interactions that are not
        explicitly defined in `self.pair_params`. It uses self-interaction 
        parameters (i, i) to compute cross-interaction parameters based on 
        the LAMMPS mixing rules.

        Parameters
        ----------
        rule : str, optional
            The mixing rule to apply. Supported rules:
            
            - 'arithmetic':
                .. math:: \\epsilon_{ij} = \\sqrt{\\epsilon_i \\epsilon_j}
                .. math:: \\sigma_{ij} = \\frac{1}{2}(\\sigma_i + \\sigma_j)
            
            - 'geometric':
                .. math:: \\epsilon_{ij} = \\sqrt{\\epsilon_i \\epsilon_j}
                .. math:: \\sigma_{ij} = \\sqrt{\\sigma_i \\sigma_j}

        overwrite : bool, optional
            If True, existing cross-interaction parameters are recalculated.
            If False, only missing interactions are computed. Default is False.

        Raises
        ------
        ValueError
            If self-interaction parameters are missing for some atom types.
            If the pair parameters are not LJ parameters.

        References
        ----------
        .. [1] LAMMPS Pair Modify Documentation. https://docs.lammps.org/pair_modify.html
        """

        for t in self.atom_types:
            if (t, t) not in self.pair_params:
                self.pair_params[(t, t)] = LJParams(
                    epsilon=0.0,
                    sigma=0.0,
                    ref="zero",
                    model="default"
                )

        missing_self_params = []
        for t in self.atom_types:
            if (t, t) not in self.pair_params:
                missing_self_params.append(t)
                
        if missing_self_params:
            raise ValueError(
                f"Missing parameters for self-interaction of the following types: {missing_self_params}. "
                "You must set these parameters with 'set_pair_params' before blending."
            )

        for t in self.atom_types:
            params = self.pair_params[(t, t)]
            if not isinstance(params, LJParams):
                raise ValueError(
                    f"Self-interaction for type '{t}' is not LJ parameters (got {type(params).__name__}). "
                    "Mixing rules only apply to LJ parameters."
                )

        for t1, t2 in combinations(self.atom_types, 2):
            key_cross = tuple(sorted((t1, t2)))
            
            if key_cross not in self.pair_params or overwrite:
                params1 = self.pair_params[(t1, t1)]
                params2 = self.pair_params[(t2, t2)]
                
                # Extract LJ settings
                eps1, sig1 = params1.epsilon, params1.sigma
                eps2, sig2 = params2.epsilon, params2.sigma
                
                # Apply the chosen rule
                if rule == 'arithmetic':
                    eps_mixed = (eps1 * eps2) ** 0.5
                    sig_mixed = (sig1 + sig2) / 2.0
                elif rule == 'geometric':
                    eps_mixed = (eps1 * eps2) ** 0.5
                    sig_mixed = (sig1 * sig2) ** 0.5
                else:
                    raise ValueError(
                        f"Unknown mixing rule: '{rule}'. "
                        "Supported rules: 'arithmetic', 'geometric'."
                    )
                
                # Store result as LJParams
                self.pair_params[key_cross] = LJParams(
                    epsilon=eps_mixed,
                    sigma=sig_mixed,
                    ref="mixed",
                    model="mixed",
                )
                
    def _set_pair_params_from_db(self: AtomicSystem, 
                                 atom_type_assignments: dict[str | int, str], db: ForceFieldDatabase) -> None:
        """
        Search the database for pair parameters and apply them.
        """
        self.pair_params = {}
        
        for label1, label2 in combinations_with_replacement(atom_type_assignments.keys(), 2):
            ff_t1 = atom_type_assignments[label1]
            ff_t2 = atom_type_assignments[label2]
            
            # Try LJ, then Buckingham
            params = db.get_lj(ff_t1, ff_t2) or db.get_buckingham(ff_t1, ff_t2)
            
            if params is not None:
                self.pair_params[tuple(sorted([label1, label2]))] = params

    def _set_bond_params_from_db(
        self: AtomicSystem, 
        atom_type_assignments: dict[str | int, str], 
        db: ForceFieldDatabase) -> None:
        """Search the database for bond parameters and apply them."""
        if not hasattr(self, 'bond_types') or self.bond_types is None:
            return
        
        self.bond_params = {}
        
        for bond_str in self.bond_types:
            elements = bond_str.split('-')
            if len(elements) != 2:
                continue
            
            sys_t1, sys_t2 = elements[0].strip(), elements[1].strip()
            
            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2)
            
            if not ff_t1 or not ff_t2:
                continue
            
            bond_params = db.get_bond(ff_t1, ff_t2)
            if bond_params is not None:
                self.bond_params[bond_str] = bond_params

    def _set_angle_params_from_db(
        self: AtomicSystem, 
        atom_type_assignments: dict[str | int, str], 
        db: ForceFieldDatabase) -> None:
        """Search the database for angle parameters and apply them."""
        if not hasattr(self, 'angle_types') or self.angle_types is None:
            return
        
        self.angle_params = {}
        
        for angle_str in self.angle_types:
            elements = angle_str.split('-')
            if len(elements) != 3:
                continue
            
            sys_t1, sys_t2, sys_t3 = elements[0].strip(), elements[1].strip(), elements[2].strip()
            
            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2)
            ff_t3 = atom_type_assignments.get(sys_t3)
            
            if not ff_t1 or not ff_t2 or not ff_t3:
                continue
            
            angle_params = db.get_angle(ff_t1, ff_t2, ff_t3)
            if angle_params is not None:
                self.angle_params[angle_str] = angle_params

    def _update_masses_and_charges(self, assignments: dict, db: ForceFieldDatabase) -> None:
        """Update masses and charges from the database."""
        masses_update = {}
        charges_update = {}
        
        for sys_type, db_type in assignments.items():
            atom_type = db.get_atom_type(db_type)
            if atom_type:
                if atom_type.mass is not None:
                    masses_update[sys_type] = atom_type.mass
                charges_update[sys_type] = atom_type.charge
        
        if masses_update:
            self.set_masses(masses_update)
        if charges_update:
            self.set_charges(charges_update)