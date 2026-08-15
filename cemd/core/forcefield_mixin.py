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
from ._format import canonical_ff_type


class ForceFieldMixin:
    """Mixin for force field operations on AtomicSystem."""

    def set_ff_keys(
        self,
        atom: dict[str, str] | None = None,
        bond: dict[str, str] | None = None,
        angle: dict[str, str] | None = None,
        dihedral: dict[str, str] | None = None,
        improper: dict[str, str] | None = None,
        overwrite: bool = True,
    ) -> None:
        """Set or update force-field keys for atom and interaction types.

        Parameters
        ----------
        atom, bond, angle, dihedral, improper
            Mapping between system types and force-field keys.
        overwrite
            If True, existing assignments are replaced.
        """

        mappings = {
            "atom": atom,
            "bond": bond,
            "angle": angle,
            "dihedral": dihedral,
            "improper": improper,
        }

        for kind, values in mappings.items():
            if not values:
                continue

            target = getattr(self._forcefield_keys, kind)

            if overwrite:
                target.update(values)
            else:
                for system_type, ff_key in values.items():
                    if system_type not in target:
                        target[system_type] = ff_key

        self._cache = {}

    def set_ff_from_database(
        self,
        atom_assignments: dict[str, str] = None,
        bond_assignments: dict[str, str] = None,
        angle_assignments: dict[str, str] = None,
        dihedral_assignments: dict[str, str] = None,
        improper_assignments: dict[str, str] = None,
        ff_database_dir: str = None,
        overwrite: bool = True,
    ) -> None:

        db = ForceFieldDatabase()

        atom_assignments = atom_assignments or {}
        bond_assignments = bond_assignments or {}
        angle_assignments = angle_assignments or {}
        dihedral_assignments = dihedral_assignments or {}
        improper_assignments = improper_assignments or {}

        resolved_atom_assignments = self._resolve_atom_assignments(
            atom_assignments,
            db,
        )

        self.set_ff_keys(
            atom=resolved_atom_assignments,
            bond=bond_assignments,
            angle=angle_assignments,
            dihedral=dihedral_assignments,
            improper=improper_assignments,
            overwrite=overwrite,
        )

        self._update_masses_and_charges(db)
        self._set_pair_params_from_db(resolved_atom_assignments, db, overwrite)
        self._set_bond_params_from_db(
            bond_assignments, resolved_atom_assignments, db, overwrite
        )
        self._set_angle_params_from_db(
            angle_assignments, resolved_atom_assignments, db, overwrite
        )
        self._set_dihedral_params_from_db(
            dihedral_assignments, resolved_atom_assignments, db, overwrite
        )
        self._set_improper_params_from_db(
            improper_assignments, resolved_atom_assignments, db, overwrite
        )

    def set_atom_ff_keys(self, value: Sequence[str] | dict[str | int, str]) -> None:
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

            self._forcefield_keys.atom.update(value)

        elif isinstance(value, (list, np.ndarray, tuple)):
            current_types = list(self.atom_types)

            if len(value) != len(current_types):
                raise ValueError(
                    f"Sequence length ({len(value)}) does not match "
                    f"the number of atom types ({len(current_types)})."
                )

            new_map = dict(zip(current_types, value))
            self._forcefield_keys.atom.update(new_map)
        else:
            raise TypeError(
                f"Unsupported argument type: {type(value).__name__}. "
                "Expected dict or sequence of strings."
            )

        self._cache = {}

    def set_bond_ff_keys(self, value: Sequence[str] | dict[str, str]) -> None:
        self._apply_topology_ff_keys("bond", value)

    def set_angle_ff_keys(self, value: Sequence[str] | dict[str, str]) -> None:
        self._apply_topology_ff_keys("angle", value)

    def set_dihedral_ff_keys(self, value: Sequence[str] | dict[str, str]) -> None:
        self._apply_topology_ff_keys("dihedral", value)

    def set_improper_ff_keys(self, value: Sequence[str] | dict[str, str]) -> None:
        self._apply_topology_ff_keys("improper", value)

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

                # Utilisation des arguments nommés pour éviter l'erreur de position
                self.set_ff_from_database(
                    atom_assignments=assignments, ff_database_dir=ff_database_dir
                )
            else:
                print("Selection cancelled.")
                break

    def set_masses(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """Set or update the atomic masses for the system and the atoms DataFrame."""
        self._apply_property_mapping("mass", value, self._masses_storage)

    def set_charges(self, value: Sequence[float] | dict[str | int, float]) -> None:
        """Set or update the charges for the atomic system and the atoms DataFrame."""
        self._apply_property_mapping("charge", value, self._charges_storage)

    def set_pair_params(
        self,
        atom_type1: str | int,
        atom_type2: str | int = None,
        params: Any = None,
    ) -> None:
        """Set pair-interaction parameters for two atom types."""

        if atom_type2 is None:
            atom_type2 = atom_type1

        if atom_type1 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type1}' does not exist in the system.")

        if atom_type2 not in self.atom_types:
            raise ValueError(f"Atom type '{atom_type2}' does not exist in the system.")

        pair_key = self._normalize_binary_key(atom_type1, atom_type2)
        self._forcefield_params.pair[pair_key] = params

    def set_bond_params(self, bond_type: str, params: Any) -> None:
        self._apply_topology_param("bond", bond_type, params)

    def set_angle_params(self, angle_type: str, params: Any) -> None:
        self._apply_topology_param("angle", angle_type, params)

    def set_dihedral_params(self, dihedral_type: str, params: Any) -> None:
        self._apply_topology_param("dihedral", dihedral_type, params)

    def set_improper_params(self, improper_type: str, params: Any) -> None:
        self._apply_topology_param("improper", improper_type, params)

    def apply_pair_mixing_rules(
        self,
        rule: str = "arithmetic",
        overwrite: bool = False,
    ) -> None:
        """Calculate missing cross-interaction parameters."""

        missing_self_params = []

        for atom_type in self.atom_types:
            key = self._normalize_binary_key(atom_type, atom_type)

            if key not in self._forcefield_params.pair:
                missing_self_params.append(atom_type)

        if missing_self_params:
            raise ValueError(
                "Missing parameters for self-interaction of the following types: "
                f"{missing_self_params}. "
                "You must set these parameters with 'set_pair_params' "
                "before blending."
            )

        for atom_type in self.atom_types:
            key = self._normalize_binary_key(atom_type, atom_type)
            params = self._forcefield_params.pair[key]

            if not isinstance(params, LJParams):
                raise ValueError(
                    f"Self-interaction for type '{atom_type}' is not LJ "
                    f"parameters (got {type(params).__name__}). "
                    "Mixing rules only apply to LJ parameters."
                )

        for atom_type1, atom_type2 in combinations(self.atom_types, 2):
            pair_key = self._normalize_binary_key(atom_type1, atom_type2)

            if pair_key in self._forcefield_params.pair and not overwrite:
                continue

            params1 = self._forcefield_params.pair[
                self._normalize_binary_key(atom_type1, atom_type1)
            ]
            params2 = self._forcefield_params.pair[
                self._normalize_binary_key(atom_type2, atom_type2)
            ]

            eps1, sig1 = params1.epsilon, params1.sigma
            eps2, sig2 = params2.epsilon, params2.sigma

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

            self._forcefield_params.pair[pair_key] = LJParams(
                epsilon=eps_mixed,
                sigma=sig_mixed,
                ref="mixed",
                model="mixed",
            )

    def _apply_topology_param(self, kind: str, topo_type: str, params: Any) -> None:
        """Helper method to validate and set a single topology parameter."""
        types_list = getattr(self, f"{kind}_types", None)

        if types_list is None:
            raise ValueError(f"The system does not have any {kind} types initialized.")

        # Les liaisons nécessitent une normalisation spécifique
        if kind == "bond":
            topo_type = self._normalize_binary_key(*topo_type.split("-"))

        if topo_type not in types_list:
            raise ValueError(
                f"{kind.capitalize()} type '{topo_type}' does not exist in the system."
            )

        target_dict = getattr(self._forcefield_params, kind)
        target_dict[topo_type] = params

    def _apply_property_mapping(
        self,
        property_name: str,
        value: Sequence[float] | dict[str | int, float],
        storage: dict,
    ) -> None:
        """Helper method to validate and apply mass or charge mappings."""
        if isinstance(value, dict):
            current_types = set(self.atom_types)
            missing_types = [
                atype for atype in value.keys() if atype not in current_types
            ]

            if missing_types:
                warnings.warn(
                    f"Targeted types missing from current system for {property_name}: {missing_types}",
                    UserWarning,
                )

            if property_name == "mass" and any(m <= 0 for m in value.values()):
                raise ValueError("All masses must be strictly positive (> 0).")

            storage.update(value)

        elif isinstance(value, (list, np.ndarray, tuple)):
            current_types = list(self.atom_types)
            if len(value) != len(current_types):
                raise ValueError(
                    f"Sequence size ({len(value)}) incompatible with "
                    f"the number of atom types ({len(current_types)}) for {property_name}."
                )

            if property_name == "mass" and any(m <= 0 for m in value):
                raise ValueError("All masses must be strictly positive (> 0).")

            storage.update(dict(zip(current_types, value)))

        else:
            raise TypeError(
                f"Unsupported argument type for {property_name}: {type(value).__name__}."
                "Expected: dict or sequence of floats."
            )

        # Update self.atoms DataFrame ONLY for charges
        if (
            property_name == "charge"
            and hasattr(self, "atoms")
            and self.atoms is not None
        ):
            mapped_values = self.atoms["type"].map(storage)

            if property_name in self.atoms.columns:
                self.atoms[property_name] = mapped_values.fillna(
                    self.atoms[property_name]
                )
            else:
                self.atoms[property_name] = mapped_values

            if self.atoms[property_name].isna().any():
                unassigned = self.atoms[self.atoms[property_name].isna()][
                    "type"
                ].unique()
                warnings.warn(
                    f"Certain types of atoms have no assigned {property_name}: {list(unassigned)}",
                    UserWarning,
                )

        self._cache = {}

    def _apply_topology_ff_keys(
        self, kind: str, value: Sequence[str] | dict[str, str]
    ) -> None:
        """Helper method to set or update force field keys for a given topology kind."""
        current_types_list = getattr(self, f"{kind}_types", [])
        current_types = set(current_types_list)
        target_dict = getattr(self._forcefield_keys, kind)

        if isinstance(value, dict):
            missing_types = []
            for t in value.keys():
                normalized_t = (
                    self._normalize_binary_key(*t.split("-")) if kind == "bond" else t
                )
                if t not in current_types and normalized_t not in current_types:
                    missing_types.append(t)

            if missing_types:
                warnings.warn(
                    f"Target {kind} types not present: {missing_types}", UserWarning
                )

            for topo_type, ff_key in value.items():
                final_key = (
                    self._normalize_binary_key(*topo_type.split("-"))
                    if kind == "bond"
                    else topo_type
                )
                target_dict[final_key] = ff_key

        elif isinstance(value, (list, np.ndarray, tuple)):
            if len(value) != len(current_types_list):
                raise ValueError(
                    f"Sequence length ({len(value)}) does not match {kind} types count ({len(current_types_list)})."
                )
            target_dict.update(dict(zip(current_types_list, value)))
        else:
            raise TypeError(f"Unsupported argument type: {type(value).__name__}.")

        self._cache = {}

    def _resolve_pair_parameters(
        self,
        db: ForceFieldDatabase,
        ff_t1: str,
        ff_t2: str,
        orig_t1: str,
        orig_t2: str,
        is_self_interaction: bool,
    ) -> Any:
        """Helper method to find or derive pair parameters from the database."""

        # 1. Search with exact keys (ex: db.lj or db.buckingham)
        params = db.get_lj(ff_t1, ff_t2) or db.get_buckingham(ff_t1, ff_t2)
        if params is not None:
            return params

        # 2. Search with the original keys (without the suffix)
        params = db.get_lj(orig_t1, orig_t2) or db.get_buckingham(orig_t1, orig_t2)
        if params is not None:
            return params

        # 3. If not found, we try to calculate from individual atoms (mixing)
        atom1 = db.get_atom_type(ff_t1)
        atom2 = db.get_atom_type(ff_t2)

        if atom1 is not None and atom2 is not None:
            if hasattr(atom1, "epsilon") and hasattr(atom2, "epsilon"):
                eps_mixed = (atom1.epsilon * atom2.epsilon) ** 0.5
                sig_mixed = (atom1.sigma + atom2.sigma) / 2.0
                return LJParams(
                    epsilon=eps_mixed,
                    sigma=sig_mixed,
                    ref="mixed from atom",
                    model="gromos",
                )

        # 4. FALLback: if still not found and it is the same atom, we put 0
        if is_self_interaction:
            return LJParams(
                epsilon=0.0,
                sigma=0.0,
                ref="zero (not found in DB)",
                model="gromos",
            )

        return None

    def _set_pair_params_from_db(
        self, atom_assignments: dict, db: ForceFieldDatabase, overwrite: bool
    ) -> None:
        """Set pair parameters by searching the database or applying mixing rules."""
        if not self._forcefield_keys.atom:
            return

        missing_pairs = []

        # Using a "dictionary comprehension" to do this in 1 line
        original_types = {
            atype: atype.split("_")[0] for atype in self._forcefield_keys.atom
        }

        for atom_type1, atom_type2 in combinations_with_replacement(
            self._forcefield_keys.atom, 2
        ):
            pair_key = self._normalize_binary_key(atom_type1, atom_type2)

            if pair_key in self._forcefield_params.pair and not overwrite:
                continue

            ff_t1 = self._forcefield_keys.atom[atom_type1]
            ff_t2 = self._forcefield_keys.atom[atom_type2]

            orig_t1 = original_types[atom_type1]
            orig_t2 = original_types[atom_type2]

            is_self_interaction = atom_type1 == atom_type2

            # We call our new method tool for doing research
            params = self._resolve_pair_parameters(
                db, ff_t1, ff_t2, orig_t1, orig_t2, is_self_interaction
            )

            if params is not None:
                self._forcefield_params.pair[pair_key] = params
            elif is_self_interaction:
                missing_pairs.append((atom_type1, ff_t1))

        if missing_pairs:
            warnings.warn(
                f"Missing pair parameters for atomic types:{missing_pairs}",
                category=UserWarning,
                stacklevel=2,
            )

    def _set_topology_params_from_db(
        self,
        topo_type: str,
        assignments: dict,
        db: ForceFieldDatabase,
        overwrite: bool,
    ) -> None:
        """Generic method to retrieve topology parameters (bonds, angles, etc.) from the DB."""
        missing_items = []

        types_list = getattr(self, f"{topo_type}_types", [])
        ff_keys_dict = getattr(self._forcefield_keys, topo_type)
        db_attr = getattr(db, topo_type, None)
        db_get_method = getattr(db, f"get_{topo_type}", None)
        params_storage = getattr(self._forcefield_params, topo_type)

        for topo_str in types_list:
            params = None

            ff_key = ff_keys_dict.get(topo_str)
            if ff_key and db_attr:
                params = db_attr.get(ff_key)

            if params is None and topo_str in assignments:
                target_key = assignments[topo_str]
                if db_attr:
                    params = db_attr.get(target_key)

            if params is None and db_get_method:
                elements = [e.strip() for e in topo_str.split("-")]
                ff_types = [self._get_ff_key_for_type(e) for e in elements]

                if all(ff_types):
                    params = db_get_method(*ff_types)

            if params is not None:
                if topo_str not in params_storage or overwrite:
                    params_storage[topo_str] = params
            else:
                missing_items.append(topo_str)

        if missing_items:
            fr_names = {
                "bond": "liaisons",
                "angle": "d'angles",
                "dihedral": "de dièdres",
                "improper": "impropres",
            }
            warnings.warn(
                f"Parameters {fr_names.get(topo_type, topo_type)} not found for: {missing_items}",
                category=UserWarning,
                stacklevel=2,
            )

    def _set_bond_params_from_db(
        self,
        bond_assignments: dict,
        atom_assignments: dict,
        db: ForceFieldDatabase,
        overwrite: bool,
    ) -> None:
        self._set_topology_params_from_db("bond", bond_assignments, db, overwrite)

    def _set_angle_params_from_db(
        self,
        angle_assignments: dict,
        atom_assignments: dict,
        db: ForceFieldDatabase,
        overwrite: bool,
    ) -> None:
        self._set_topology_params_from_db("angle", angle_assignments, db, overwrite)

    def _set_dihedral_params_from_db(
        self,
        dihedral_assignments: dict,
        atom_assignments: dict,
        db: ForceFieldDatabase,
        overwrite: bool,
    ) -> None:
        self._set_topology_params_from_db(
            "dihedral", dihedral_assignments, db, overwrite
        )

    def _set_improper_params_from_db(
        self,
        improper_assignments: dict,
        atom_assignments: dict,
        db: ForceFieldDatabase,
        overwrite: bool,
    ) -> None:
        self._set_topology_params_from_db(
            "improper", improper_assignments, db, overwrite
        )

    def _resolve_atom_assignments(
        self, assignments: dict, db: ForceFieldDatabase
    ) -> dict:
        """Resolves atom abbreviations (ex: 'h_star' -> 'clayff.h_star')."""
        resolved = {}
        for sys_type, db_type in assignments.items():
            if "." in db_type:
                resolved[sys_type] = db_type
            else:
                found = next(
                    (
                        full_type
                        for full_type in db.atom.keys()
                        if full_type.endswith(f".{db_type}")
                    ),
                    None,
                )

                if found is None:
                    warnings.warn(
                        f"Atom type '{db_type}' not found in database.",
                        category=UserWarning,
                        stacklevel=3,
                    )
                    continue

                resolved[sys_type] = found
        return resolved

    @staticmethod
    def _normalize_binary_key(type1: str | int, type2: str | int) -> str:
        """Return a canonical key for a two-type interaction."""
        return "-".join(sorted((str(type1), str(type2))))

    def _update_masses_and_charges(self, db: ForceFieldDatabase) -> None:
        """Updates masses and charges from atoms and their ff_key."""

        masses_update = {}
        charges_update = {}
        missing_types = []

        for sys_type, db_type in self.ff_keys.atom.items():
            if db_type is None:
                continue

            atom_type = db.get_atom_type(db_type)

            if atom_type:
                if atom_type.mass is not None:
                    masses_update[sys_type] = atom_type.mass
                charges_update[sys_type] = atom_type.charge
            else:
                missing_types.append((sys_type, db_type))

        if missing_types:
            warnings.warn(
                f"Types introuvables pour les masses/charges : {missing_types}",
                category=UserWarning,
                stacklevel=2,
            )

        if masses_update:
            self.set_masses(masses_update)
        if charges_update:
            self.set_charges(charges_update)

    def _get_ff_key_for_type(self, sys_type: str) -> str | None:
        if hasattr(self._forcefield_keys, "atom") and self._forcefield_keys.atom:
            return self._forcefield_keys.atom.get(sys_type)
        return None

    def _remap_ff_keys(self, type_mapping: dict[str, str]) -> None:
        """Remap force-field keys after atom types have been renamed."""

        # Atom FF keys
        old_atom_keys = dict(self._forcefield_keys.atom)
        new_atom_keys = {}
        for old_type, ff_key in old_atom_keys.items():
            new_type = type_mapping.get(old_type, old_type)
            new_atom_keys[new_type] = ff_key
        self._forcefield_keys.atom = new_atom_keys

        # Interaction FF keys
        for kind in ["bond", "angle", "dihedral", "improper"]:
            old_keys = dict(getattr(self._forcefield_keys, kind))
            new_keys = {}

            for interaction_type, ff_key in old_keys.items():
                old_types = interaction_type.split("-")
                new_types = tuple(
                    type_mapping.get(atom_type, atom_type) for atom_type in old_types
                )

                # === IMPORTANT CHANGE ===
                # For the dihedrons, we keep the original order
                if kind == "dihedral":
                    new_interaction_type = "-".join(new_types)
                else:
                    new_interaction_type = canonical_ff_type(new_types, kind)

                new_keys[new_interaction_type] = ff_key

            setattr(self._forcefield_keys, kind, new_keys)

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
