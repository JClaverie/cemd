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

import warnings
from collections.abc import Sequence
from itertools import combinations, combinations_with_replacement
from typing import Any

import numpy as np

from ..forcefield._config import get_ff_db_path
from ..forcefield.forcefield_database import ForceFieldDatabase
from ..forcefield.models import LJParams


class ForceFieldMixin:
    """Mixin for force field operations on AtomicSystem."""

    atom_types: list[str]
    pair_params: dict
    bond_params: dict
    angle_params: dict
    dihedral_params: dict
    improper_params: dict
    bondbond_params: dict
    bondangle_params: dict
    bond_types: list[str] | None
    angle_types: list[str] | None
    dihedral_types: list[str] | None
    improper_types: list[str] | None

    def set_ff_from_database(
        self, assignments: dict[str | int, str], ff_database_dir: str = None
    ) -> None:
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
            if "." in db_type:
                resolved_assignments[sys_type] = db_type
            else:
                # Search type in all models
                found = None
                for full_type in db.atom.keys():
                    if full_type.endswith(f".{db_type}"):
                        found = full_type
                        break
                if found is None:
                    warnings.warn(
                        f"Type '{db_type}' not found in any model in database",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    continue
                resolved_assignments[sys_type] = found

        # Apply settings
        self._set_pair_params_from_db(resolved_assignments, db)
        self._set_bond_params_from_db(resolved_assignments, db)
        self._set_angle_params_from_db(resolved_assignments, db)
        self._update_masses_and_charges(resolved_assignments, db)

    def explore_ff_database(
        self, ff_database_dir: str = None, visible_rows: int = 20
    ) -> None:
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

        import webbrowser

        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, ScrollOffsets, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

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
            subset = df_list[df_list["element"] == el]
            if subset.empty:
                print(f"No parameters for element {el}. Skipping.")
                continue

            types = []
            for _, row in subset.iterrows():
                types.append(
                    {
                        "type": row["type"],
                        "full_type": row["full_type"],
                        "model": row["model"],
                        "environment": row["environment"],
                        "ref": row["ref"],
                    }
                )

            index = 0
            selected = None
            scroll_top = 0
            kb = KeyBindings()

            row_format = "{cursor} {type:<20} {model:<15} | {env:<45.45}"

            def render():
                header = row_format.format(
                    cursor=" ", type="TYPE", model="MODEL", env="ENVIRONMENT"
                )
                separator = "-" * len(header)

                lines = [
                    f"--- Select Forcefield Parameters for type {t} [{index + 1}/{len(types)}] ---",
                    f"Found {len(types)} options  [{index + 1}/{len(types)}]",
                    "↑↓ Move   [Enter] Assign   [p] Open DOI   [q] Quit",
                    "",
                    header,
                    separator,
                ]

                # Show only the visible window
                visible = types[scroll_top : scroll_top + visible_rows]

                for i, r in enumerate(visible):
                    real_index = scroll_top + i
                    cursor = "➜" if real_index == index else " "
                    lines.append(
                        row_format.format(
                            cursor=cursor,
                            type=r["type"],
                            model=r["model"],
                            env=r["environment"][:45] if r["environment"] else "",
                        )
                    )

                # Scroll indicator
                if len(types) > visible_rows:
                    lines.append(
                        f"\n  ↕ {scroll_top + 1}-{min(scroll_top + visible_rows, len(types))} of {len(types)}"
                    )

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
                ref = types[index].get("ref", "")
                if ref and ref.startswith("http"):
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
                assignments[t] = selected["full_type"]
                print(f"Assigned: {t} -> {selected['full_type']} ({selected['model']})")
                self.set_ff_from_database(assignments, ff_database_dir)
            else:
                print("Selection cancelled.")
                break

    def set_masses(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """Set or update the atomic masses for the system and the atoms DataFrame.

        Parameters
        ----------
        value : Sequence of float or dict of {str or int : float}
            If a sequence is provided, it must match the current order and
            length of unique `atom_types`. If a dictionary is provided, it
            performs a partial or full update of the mass mapping.

        Raises
        ------
        ValueError
            If the sequence length does not match `atom_types` or if any mass is <= 0.
        TypeError
            If `value` is neither a dictionary nor a supported sequence type.
        """
        if isinstance(value, dict):
            current_types = set(self.atom_types)
            missing_types = [
                atype for atype in value.keys() if atype not in current_types
            ]

            if missing_types:
                warnings.warn(
                    f"Target atom types not present in current system: {missing_types}",
                    UserWarning,
                )

            if any(m <= 0 for m in value.values()):
                raise ValueError("All masses must be strictly positive (> 0).")

            self._masses_storage.update(value)

        elif isinstance(value, (list, np.ndarray, tuple)):
            current_types = list(self.atom_types)

            if len(value) != len(current_types):
                raise ValueError(
                    f"Sequence length ({len(value)}) does not match "
                    f"the number of atom types ({len(current_types)})."
                )

            if any(m <= 0 for m in value):
                raise ValueError("All masses must be strictly positive (> 0).")

            new_map = dict(zip(current_types, value))
            self._masses_storage.update(new_map)

        else:
            raise TypeError(
                f"Unsupported argument type: {type(value).__name__}. "
                "Expected dict or sequence of floats."
            )

        if hasattr(self, "atoms") and self.atoms is not None:
            mapped_masses = self.atoms["type"].map(self._masses_storage)

            if "mass" in self.atoms.columns:
                self.atoms["mass"] = mapped_masses.fillna(self.atoms["mass"])
            else:
                self.atoms["mass"] = mapped_masses

            if self.atoms["mass"].isna().any():
                unassigned = self.atoms[self.atoms["mass"].isna()]["type"].unique()
                warnings.warn(
                    f"Some atom types have no assigned mass: {list(unassigned)}",
                    UserWarning,
                )

        # Clear internal cache
        self._cache = {}

    def set_charges(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """Set or update the charges for the atomic system and the atoms DataFrame.

        Parameters
        ----------
        value : Sequence of float or dict of {str or int : float}
            If a sequence is provided, it must match the current order and
            length of unique `atom_types`. If a dictionary is provided, it
            performs a partial or full update of the charge mapping.
        """

        if isinstance(value, dict):
            current_types = set(self.atom_types)
            missing_types = [
                atype for atype in value.keys() if atype not in current_types
            ]

            if missing_types:
                warnings.warn(
                    f"Targeted types missing from current system: {missing_types}",
                    UserWarning,
                )

            self._charges_storage.update(value)

        elif isinstance(value, (list, np.ndarray, tuple)):
            # Make sure atom_types is an ordered list
            current_types = list(self.atom_types)

            if len(value) != len(current_types):
                raise ValueError(
                    f"Sequence size ({len(value)}) incompatible with "
                    f"the number of atom types ({len(current_types)})."
                )

            new_map = dict(zip(current_types, value))
            self._charges_storage.update(new_map)

        else:
            raise TypeError(
                f"Argument type not taken into account: {type(value).__name__}."
                "Expected: dict or sequence of floats."
            )

        if hasattr(self, "atoms") and self.atoms is not None:
            mapped_charges = self.atoms["type"].map(self._charges_storage)

            if "charge" in self.atoms.columns:
                self.atoms["charge"] = mapped_charges.fillna(self.atoms["charge"])
            else:
                self.atoms["charge"] = mapped_charges

            if self.atoms["charge"].isna().any():
                unassigned = self.atoms[self.atoms["charge"].isna()]["type"].unique()
                warnings.warn(
                    f"Certain types of atoms have no assigned charge:{list(unassigned)}",
                    UserWarning,
                )

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
                f"Invalid coefficient values for {name}: expected numbers, got {coeffs}"
            ) from e

    def set_pair_params(
        self,
        atom_type1: str | int,
        atom_type2: str | int = None,
        params: Any = None,  # Idéalement : LJParams | BuckinghamParams
    ) -> None:
        """
        Assign non-bond parameter objects for a pair of atom types (self or cross).

        Parameters
        ----------
        atom_type1 : str or int
            The identifier of the first atom type.
        atom_type2 : str or int, optional
            The identifier of the second atom type. If None, assumes self-interaction.
        params : Interaction object
            The potential parameter object (e.g., LJParams or BuckinghamParams instance).
        """
        if atom_type2 is None:
            atom_type2 = atom_type1

        if atom_type1 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type1}' does not exist in the system.")
        if atom_type2 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type2}' does not exist in the system.")

        # Tri sécurisé pour un stockage unique
        sorted_key = tuple(sorted([atom_type1, atom_type2], key=str))
        self.pair_params[sorted_key] = params

    def set_bond_params(self, bond_type: str, params: Any) -> None:
        """
        Assign a structural parameter object for a specific bond type.

        Parameters
        ----------
        bond_type : str
            The identifier of the bond type (e.g., 'H-O').
        params : Bond object
            The bond parameter object (e.g., HarmonicBondParams instance).
        """
        if not hasattr(self, "bond_types") or self.bond_types is None:
            raise ValueError("The system does not have any bond types initialized.")

        # Normalisation de la liaison (H-O devient identique à O-H)
        elements = bond_type.split("-")
        if len(elements) == 2:
            normalized_bond = "-".join(
                sorted([elements[0].strip(), elements[1].strip()], key=str)
            )
        else:
            normalized_bond = bond_type

        if bond_type not in self.bond_types and normalized_bond not in self.bond_types:
            raise ValueError(
                f"Bond type '{bond_type}' (or '{normalized_bond}') does not exist in the system."
            )

        self.bond_params[normalized_bond] = params

    def set_angle_params(self, angle_type: str, params: Any) -> None:
        """
        Assign a structural parameter object for a specific angle type.

        Parameters
        ----------
        angle_type : str
            The identifier of the angle type (e.g., 'H-O-H').
        params : Angle object
            The angle parameter object (e.g., HarmonicAngleParams instance).
        """
        if not hasattr(self, "angle_types") or self.angle_types is None:
            raise ValueError("The system does not have any angle types initialized.")

        if angle_type not in self.angle_types:
            raise ValueError(f"Angle type '{angle_type}' does not exist in the system.")

        self.angle_params[angle_type] = params

    def set_dihedral_params(self, dihedral_type: str, params: Any) -> None:
        """
        Assign a structural parameter object for a specific dihedral type.

        Parameters
        ----------
        dihedral_type : str
            The identifier of the dihedral type (e.g., 'C-C-C-C').
        params : Dihedral object
            The dihedral parameter object (depends on the style).
        """
        if not hasattr(self, "dihedral_types") or self.dihedral_types is None:
            raise ValueError("The system does not have any dihedral types initialized.")

        if dihedral_type not in self.dihedral_types:
            raise ValueError(
                f"Dihedral type '{dihedral_type}' does not exist in the system."
            )

        self.dihedral_params[dihedral_type] = params

    def set_improper_params(self, improper_type: str, params: Any) -> None:
        """
        Assign a structural parameter object for a specific improper type.

        Parameters
        ----------
        improper_type : str
            The identifier of the improper type.
        params : Improper object
            The improper parameter object (e.g., HarmonicAngleParams instance).
        """
        if not hasattr(self, "improper_types") or self.improper_types is None:
            raise ValueError("The system does not have any improper types initialized.")

        if improper_type not in self.improper_types:
            raise ValueError(
                f"Improper type '{improper_type}' does not exist in the system."
            )

        self.improper_params[improper_type] = params

    def apply_pair_mixing_rules(self, rule="arithmetic", overwrite=False) -> None:
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
            if (t, t) not in self.pair_params:
                self.pair_params[(t, t)] = LJParams(
                    epsilon=0.0, sigma=0.0, ref="zero", model="default"
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
                if rule == "arithmetic":
                    eps_mixed = (eps1 * eps2) ** 0.5
                    sig_mixed = (sig1 + sig2) / 2.0
                elif rule == "geometric":
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

    def _set_pair_params_from_db(
        self,
        atom_type_assignments: dict[str | int, str],
        db: ForceFieldDatabase,
    ) -> None:
        """
        Search the database for pair parameters and apply them.
        """
        missing_pairs = []

        for label1, label2 in combinations_with_replacement(
            atom_type_assignments.keys(), 2
        ):
            ff_t1 = atom_type_assignments[label1]
            ff_t2 = atom_type_assignments[label2]

            # Try LJ, then Buckingham
            params = db.get_lj(ff_t1, ff_t2) or db.get_buckingham(ff_t1, ff_t2)

            if params is not None:
                # Secure sorting that supports str/int mixing
                pair_key = tuple(sorted([label1, label2], key=str))
                self.pair_params[pair_key] = params
            elif label1 == label2:
                # Report if a self-interaction is missing
                missing_pairs.append((label1, ff_t1))

        if missing_pairs:
            warnings.warn(
                f"Missing self-interaction pair parameters for: {missing_pairs}",
                category=UserWarning,
                stacklevel=2,
            )

    def _set_bond_params_from_db(
        self,
        atom_type_assignments: dict[str | int, str],
        db: ForceFieldDatabase,
    ) -> None:
        """Search the database for bond parameters and apply them."""
        if not hasattr(self, "bond_types") or self.bond_types is None:
            return

        missing_bonds = []

        for bond_str in self.bond_types:
            elements = bond_str.split("-")
            if len(elements) != 2:
                continue

            sys_t1, sys_t2 = elements[0].strip(), elements[1].strip()

            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2)

            if not ff_t1 or not ff_t2:
                missing_bonds.append(bond_str)
                continue

            bond_params = db.get_bond(ff_t1, ff_t2)
            if bond_params is not None:
                self.bond_params[bond_str] = bond_params
            else:
                missing_bonds.append(f"{bond_str} ({ff_t1}-{ff_t2})")

        if missing_bonds:
            warnings.warn(
                f"Missing bond parameters in database for: {missing_bonds}",
                category=UserWarning,
                stacklevel=2,
            )

    def _set_angle_params_from_db(
        self,
        atom_type_assignments: dict[str | int, str],
        db: ForceFieldDatabase,
    ) -> None:
        """Search the database for angle parameters and apply them."""
        if not hasattr(self, "angle_types") or self.angle_types is None:
            return

        missing_angles = []

        for angle_str in self.angle_types:
            elements = angle_str.split("-")
            if len(elements) != 3:
                continue

            sys_t1, sys_t2, sys_t3 = (
                elements[0].strip(),
                elements[1].strip(),
                elements[2].strip(),
            )

            ff_t1 = atom_type_assignments.get(sys_t1)
            ff_t2 = atom_type_assignments.get(sys_t2)
            ff_t3 = atom_type_assignments.get(sys_t3)

            if not ff_t1 or not ff_t2 or not ff_t3:
                missing_angles.append(angle_str)
                continue

            angle_params = db.get_angle(ff_t1, ff_t2, ff_t3)
            if angle_params is not None:
                self.angle_params[angle_str] = angle_params
            else:
                missing_angles.append(f"{angle_str} ({ff_t1}-{ff_t2}-{ff_t3})")

            bb_params = db.get_bondbond(ff_t1, ff_t2, ff_t3)
            if bb_params is not None:
                self.bondbond_params[angle_str] = bb_params

            ba_params = db.get_bondangle(ff_t1, ff_t2, ff_t3)
            if ba_params is not None:
                self.bondangle_params[angle_str] = ba_params

        if missing_angles:
            warnings.warn(
                f"Missing angle parameters in database for: {missing_angles}",
                category=UserWarning,
                stacklevel=2,
            )

    def _update_masses_and_charges(
        self, assignments: dict, db: ForceFieldDatabase
    ) -> None:
        """Update masses and charges from the database."""
        masses_update = {}
        charges_update = {}
        missing_types = []

        for sys_type, db_type in assignments.items():
            atom_type = db.get_atom_type(db_type)
            if atom_type:
                if atom_type.mass is not None:
                    masses_update[sys_type] = atom_type.mass
                charges_update[sys_type] = atom_type.charge
            else:
                missing_types.append((sys_type, db_type))

        if missing_types:
            warnings.warn(
                f"Atom types not found in database for mass/charge update: {missing_types}",
                category=UserWarning,
                stacklevel=2,
            )

        if masses_update:
            self.set_masses(masses_update)
        if charges_update:
            self.set_charges(charges_update)
