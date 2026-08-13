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

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._constants import MASSES_DICT
from ..topology._apply import (
    apply_clayff_rules,
    apply_cshff_rules,
    apply_single_dihedral_rule_to_universe,
    apply_single_rule_to_universe,
)

if TYPE_CHECKING:
    from ..topology.rules import DihedralRule, TopologyRule
    from .atomic_system import AtomicSystem

RULE_SETS: dict[str, callable] = {
    "clayff": apply_clayff_rules,
    "cshff": apply_cshff_rules,
}


class TopologyMixin:
    def _remap_connection_types(self, connection_type: str):
        """Generic method to update connection types.
        connection_type must be 'bond', 'angle', 'dihedral' or 'improper'.
        """
        df = getattr(self, connection_type + "s")
        if df is None:
            return

        def get_new_type(row):
            indices = [row[f"atom_{i + 1}"] for i in range(len(row) - 1)]
            types = [str(self.atoms.loc[idx, "type"]) for idx in indices]

            if connection_type == "bond":
                types.sort()

            elif connection_type == "angle":
                if types[0] > types[2]:
                    types = [types[2], types[1], types[0]]

            return "-".join(types)

        df["type"] = df.apply(get_new_type, axis=1)
        setattr(self, connection_type + "s", df)

    def set_types(
        self, new_types: Sequence[str | int] | dict[str | int, str | int]
    ) -> AtomicSystem:
        """Assign types to atoms, supporting both full lists and partial dictionaries.

        Parameters
        ----------
        new_types
            If a list: New types to assign (must match the length of current atom_types).
            If a dict: Partial mapping of {old_type: new_type} to update specific types.
        """

        self._cache = {}

        old_types_snapshot = list(map(str, self.atom_types))
        if isinstance(self.masses, dict):
            old_masses_dict = {str(k): v for k, v in self.masses.items()}
        else:
            old_masses_dict = dict(zip(old_types_snapshot, self.masses))

        if isinstance(new_types, Sequence):
            if len(new_types) != len(old_types_snapshot):
                raise ValueError(
                    "The new list of atom types must match the current number of atom types."
                )
            mapping = dict(zip(old_types_snapshot, map(str, new_types)))

        elif isinstance(new_types, dict):
            for k in new_types.keys():
                if str(k) not in old_types_snapshot:
                    print(
                        f"WARNING: Type '{k}' targeted in set_types is not currently in the system."
                    )

            mapping = {t: t for t in old_types_snapshot}
            for k, v in new_types.items():
                mapping[str(k)] = str(v)

        else:
            raise TypeError("new_types must be either a list or a dict.")

        self.atoms["type"] = self.atoms["type"].astype(str).replace(mapping)

        for connection in ["bond", "angle", "dihedral", "improper"]:
            self._remap_connection_types(connection)

        new_masses_dic = {}
        for old_name, m in old_masses_dict.items():
            new_name = mapping[old_name]
            if new_name not in new_masses_dic or m > 0:
                new_masses_dic[new_name] = m

        self.set_masses(new_masses_dic)

        return self

    def set_types_from_elements(
        self, prevent: Sequence[str | int] = None
    ) -> AtomicSystem:
        """
        Reset atom types to their corresponding chemical elements.
        """

        self._cache = {}
        prevent = set(prevent or [])

        new_types = [
            atype if atype in prevent else self.elements[int(atype)]
            for atype in self.atom_types
        ]

        return self.set_types(new_types)

    def set_type2atoms(
        self, indices: Sequence[int], atom_type: str | int
    ) -> AtomicSystem:
        """
        Assign a new type to a specific subset of atoms.

        Parameters
        ----------
        indices : Sequence[int]
            The indices of the atoms to modify.
        atom_type : str | int
            The new type to assign to these atoms.

        Returns
        -------
        AtomicSystem
            The updated system.
        """

        self._cache = {}

        if not np.issubdtype(type(atom_type), type(self.atom_types[0])):
            raise TypeError(
                "The atom type must be of the same type (string or integer) than existing atom types."
            )

        data_masses_dic = self.masses
        old_type = self.atoms.loc[indices, "type"].iloc[0]

        # Edit in place
        self.atoms.loc[indices, "type"] = atom_type

        updated_atom_types = sorted(self.atoms["type"].unique().tolist())

        for t in updated_atom_types:
            if t not in data_masses_dic:
                if t in MASSES_DICT:
                    data_masses_dic[t] = MASSES_DICT[t]
                else:
                    data_masses_dic[t] = data_masses_dic.get(old_type, 1.0)

        # CHANGE: pass a dict instead of a list to avoid order/length issues
        self.set_masses(data_masses_dic)

        return self

    def add_bond(
        self, atom_list: Sequence[int], bond_type: str | int | None = None
    ) -> None:
        """Set a bond between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the two atoms in the bond.
            bond_type
                Type of the bond (numerical or alphabetical)
        """

        self._set_connection("bond", atom_list, bond_type=None)

    def add_angle(
        self, atom_list: Sequence[int], angle_type: str | int | None = None
    ) -> None:
        """Set a angle between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the angle, in the right order.
            angle_type
                Type of the angle (numerical or alphabetical)
        """

        self._set_connection("angle", atom_list, angle_type=None)

    def add_dihedral(
        self, atom_list: Sequence[int], dihedral_type: str | int | None = None
    ) -> None:
        """Set a dihedral between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the dihedral, in the right order.
            dihedral_type
                Type of the dihedral (numerical or alphabetical)
        """

        self._set_connection("dihedral", atom_list, dihedral_type=None)

    def add_improper(
        self, atom_list: Sequence[int], improper_type: str | int | None = None
    ) -> None:
        """Set a improper between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the improper, in the right order.
            improper_type
                Type of the improper (numerical or alphabetical)
        """

        self._set_connection("improper", atom_list, improper_type=None)

    def _set_connection(
        self,
        connection_class: str,
        atom_list: list[int],
        connection_type: str | int | None,
    ) -> AtomicSystem:
        """
        Factorized version to manage connections dynamically.
        """

        import pandas as pd

        # 1. Setting up connections
        conn_config = {
            "bond": {"cols": 2, "target": "bonds"},
            "angle": {"cols": 3, "target": "angles"},
            "dihedral": {"cols": 4, "target": "dihedrals"},
            "improper": {"cols": 4, "target": "impropers"},
        }

        if connection_class not in conn_config:
            raise ValueError(f"Unknown connection class: {connection_class}")

        cfg = conn_config[connection_class]
        df_connections = getattr(self, cfg["target"], pd.DataFrame())

        # 2. Automatic type logic
        if connection_type is None:
            if isinstance(self.atom_types[0], int):
                raise ValueError(
                    f"Please provide a {connection_class} type in integer format."
                )

            # Type reconstruction based on atom names
            atom_types = [self.atoms.loc[i, "type"] for i in atom_list]
            connection_type = "-".join(sorted(atom_types))

        # 3. Preparing the new line
        new_row = {"type": connection_type}
        for i, idx in enumerate(atom_list):
            new_row[f"atom_{i + 1}"] = idx

        # 4. Existence check (generic)
        atom_cols = [f"atom_{i + 1}" for i in range(cfg["cols"])]

        if not df_connections.empty:
            # Checks if the row already exists (simple column comparison)
            exists = (df_connections[atom_cols].values == atom_list).all(axis=1).any()
            if exists:
                print(f"This {connection_class} already exists.")
                return self

        # 5. Adding and cleaning
        new_row_df = pd.DataFrame([new_row])
        df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)

        # Conversion typing
        if isinstance(self.atom_types[0], int):
            df_connections = df_connections.astype(int)
        else:
            df_connections[atom_cols] = df_connections[atom_cols].astype(int)

        df_connections.index = range(1, len(df_connections) + 1)
        setattr(self, cfg["target"], df_connections)

        return self

    def remove_connection_types(
        self,
        bond_types: Sequence[str | int] = None,
        angle_types: Sequence[str | int] = None,
        dihedral_types: Sequence[str | int] = None,
        improper_types: Sequence[str | int] = None,
    ) -> None:
        """
        Remove specific connection types from the system.

        Parameters
        ----------
        bond_types : Sequence[str | int], optional
            Bond types to remove (e.g., ['1', 'O-Si']).
        angle_types : Sequence[str | int], optional
            Angle types to remove.
        dihedral_types : Sequence[str | int], optional
            Dihedral types to remove.
        improper_types : Sequence[str | int], optional
            Improper types to remove.
        """

        if bond_types is None:
            bond_types = []
        if angle_types is None:
            angle_types = []
        if dihedral_types is None:
            dihedral_types = []
        if improper_types is None:
            improper_types = []

        if len(bond_types) != 0:
            # list of remaining bond types
            remaining_bonds_list = list(set(self.bond_types) - set(bond_types))

            # remove bonds
            self.bonds = self.bonds[~self.bonds.type.isin(bond_types)]

            # update bonds indices
            self.bonds.index = list(range(1, len(self.bonds) + 1))

            # update bonds types range
            self.bonds.type.replace(
                self.bonds.type.unique(), remaining_bonds_list, inplace=True
            )

            # update system info
            if len(self.bonds) == 0:
                self.bonds = None

        if len(angle_types) != 0:
            # list of remaining angle types
            remaining_angles_list = list(set(self.angle_types) - set(angle_types))

            # remove angles
            self.angles = self.angles[~self.angles.type.isin(angle_types)]

            # update angles indices
            self.angles.index = list(range(1, len(self.angles) + 1))

            # update angles types range
            self.angles.type.replace(
                self.angles.type.unique(), remaining_angles_list, inplace=True
            )

            # update system info
            if len(self.angles) == 0:
                self.angles = None

        # remove dihedrals
        if len(dihedral_types) != 0:
            # list of remaining dihedrals types
            remaining_dihedrals_list = list(
                set(self.dihedral_types) - set(dihedral_types)
            )

            # remove dihedrals
            self.dihedrals = self.dihedrals[self.dihedrals.type.isin(dihedral_types)]

            # update dihedrals indices
            self.dihedrals.index = list(range(1, len(self.dihedrals) + 1))

            # update dihedrals types range
            self.dihedrals.type.replace(
                self.dihedrals.type.unique(), remaining_dihedrals_list, inplace=True
            )

            # update system info
            if len(self.dihedrals) == 0:
                self.dihedrals = None

        # remove impropers
        if len(improper_types) != 0:
            # list of remaining improper types
            remaining_impropers_list = list(
                set(self.improper_types) - set(improper_types)
            )

            # remove impropers
            self.impropers = self.impropers[self.impropers.type.isin(improper_types)]

            # update impropers indices
            self.impropers.index = list(range(1, len(self.impropers) + 1))

            # update impropers types range
            self.impropers.type.replace(
                self.impropers.type.unique(), remaining_impropers_list, inplace=True
            )

            # update system info
            if len(self.impropers) == 0:
                self.impropers = None

    def keep_connection_types(
        self,
        bond_types: Sequence[str | int] = None,
        angle_types: Sequence[str | int] = None,
        dihedral_types: Sequence[str | int] = None,
        improper_types: Sequence[str | int] = None,
    ) -> None:
        """
        Keep only the specified connection types and remove all others.

        This method filters the topology of the system by retaining only the
        specified bond, angle, dihedral, and improper types. All connection
        types not listed will be removed from the system.

        Parameters
        ----------
        bond_types : Sequence[str | int], optional
            Bond types to keep. Can be specified as type names (str) or type
            indices (int). If None, all bonds are removed.
        angle_types : Sequence[str | int], optional
            Angle types to keep. Can be specified as type names (str) or type
            indices (int). If None, all angles are removed.
        dihedral_types : Sequence[str | int], optional
            Dihedral types to keep. Can be specified as type names (str) or type
            indices (int). If None, all dihedrals are removed.
        improper_types : Sequence[str | int], optional
            Improper types to keep. Can be specified as type names (str) or type
            indices (int). If None, all impropers are removed.

        Notes
        -----
        -If a connection type is specified as a string, it will match the
          type name in the topology.
        -If specified as an integer, it will match the type index (1-based).
        -This operation modifies the system in place.

        Examples
        --------
        >>> # Keep only Si-O bonds and H-O-H angles
        >>> system.keep_connection_types(
        ...     bond_types=['Si-O'],
        ...     angle_types=['H-O-H']
        ... )
        >>>
        >>> # Keep bonds of type 1 and 2 only
        >>> system.keep_connection_types(bond_types=[1, 2])
        >>>
        >>> # Remove all bonds but keep all angles
        >>> system.keep_connection_types(bond_types=[], angle_types=None)
        """

        if bond_types is None:
            bond_types = []
        if angle_types is None:
            angle_types = []
        if dihedral_types is None:
            dihedral_types = []
        if improper_types is None:
            improper_types = []

        bondtypes2remove = []
        angletypes2remove = []
        dihedraltypes2remove = []
        impropertypes2remove = []

        if self.num_bond_types != 0:
            bondtypes2remove = list(set(self.bond_types) - set(bond_types))

        if self.num_angle_types != 0:
            angletypes2remove = list(set(self.angle_types) - set(angle_types))

        if self.num_dihedral_types != 0:
            dihedraltypes2remove = list(set(self.dihedral_types) - set(dihedral_types))

        if self.num_improper_types != 0:
            impropertypes2remove = list(set(self.improper_types) - set(improper_types))

        self.remove_connection_types(
            bond_types=bondtypes2remove,
            angle_types=angletypes2remove,
            dihedral_types=dihedraltypes2remove,
            improper_types=impropertypes2remove,
        )

    def remove_all_connections(self) -> None:
        """
        Apply a predefined topology style or a custom connectivity rule.

        Parameters
        ----------
        r : dict | str
            If a string, the name of a predefined rule set (e.g., 'clayff', 'cshff').
            If a dict, a single custom rule dictionary.

        Returns
        -------
        AtomicSystem
            The updated system with new topology.
        """

        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None

    def set_topology(
        self,
        r: str | TopologyRule | DihedralRule | list[TopologyRule | DihedralRule],
    ) -> AtomicSystem:
        """
        Apply a predefined style or custom rule(s) to the system.

        Parameters
        ----------
        r : str | TopologyRule | DihedralRule | list[TopologyRule | DihedralRule]
            -str: Name of predefined rule set (e.g., 'clayff', 'cshff')
            -TopologyRule: Single topology rule
            -DihedralRule: Single dihedral rule
            - list[TopologyRule | DihedralRule]: Mixed list of rules
        """
        universe = self.to_mda()
        actions = {}

        if isinstance(r, str):
            style = r.lower()
            if style not in RULE_SETS:
                supported = ", ".join(RULE_SETS.keys())
                raise ValueError(
                    f"Topology style '{r}' not recognized. Available styles: {supported}"
                )

            universe, actions = RULE_SETS[style](universe)
        elif isinstance(r, TopologyRule):
            universe = apply_single_rule_to_universe(universe, r)
        elif isinstance(r, DihedralRule):
            universe = apply_single_dihedral_rule_to_universe(universe, r)
        elif isinstance(r, list):
            for rule in r:
                if isinstance(rule, TopologyRule):
                    universe = apply_single_rule_to_universe(universe, rule)
                elif isinstance(rule, DihedralRule):
                    universe = apply_single_dihedral_rule_to_universe(universe, rule)
                else:
                    raise TypeError(
                        f"Expected TopologyRule or DihedralRule, got {type(rule)}"
                    )
        else:
            raise TypeError(
                f"Expected str, TopologyRule, or DihedralRule, got {type(r)}"
            )

        new_lmp_data = AtomicSystem.from_mda(universe)
        self._replace_internals(new_lmp_data)

        if "rename_atoms" in actions:
            atom_ids, new_type = actions["rename_atoms"]
            self.set_type2atoms(atom_ids, new_type)

        for connection in ["bond", "angle", "dihedral", "improper"]:
            self._remap_connection_types(connection)

        return self

    def guess_connections(
        self,
        guess_angles: bool = True,
        guess_dihedrals: bool = True,
        guess_impropers: bool = True,
    ) -> AtomicSystem:
        """Guess angles, dihedrals and/or impropers directly from existing bonds.

        Parameters
        ----------
        guess_angles : bool, default=True
            If True, guess angles.
        guess_dihedrals : bool, default=True
            If True, guess dihedrals.
        guess_impropers : bool, default=True
            If True, guess impropers.

        Returns
        -------
        AtomicSystem
            The updated system.
        """
        if self.bonds is None or self.bonds.empty:
            print(
                "WARNING: No bonds found in system. Cannot guess higher-order topology."
            )
            return self

        # 1. Dictionnaire d'accès rapide pour les types d'atomes (Supporte l'indexation par ID ou position)
        atom_type_map = dict(zip(self.atoms.index, self.atoms["type"].astype(str)))

        # 2. Construction du graphe d'adjacence
        graph = {}
        for _, row in self.bonds.iterrows():
            a1, a2 = int(row["atom_1"]), int(row["atom_2"])
            graph.setdefault(a1, set()).add(a2)
            graph.setdefault(a2, set()).add(a1)

        # 3. Traitement des Angles et Impropers
        if guess_angles or guess_impropers:
            angles_list = []
            impropers_list = []

            for center, neighbors in graph.items():
                # Angles (A - B - C)
                if guess_angles and len(neighbors) >= 2:
                    for a1, a3 in itertools.combinations(sorted(neighbors), 2):
                        t1 = atom_type_map.get(a1, "X")
                        t2 = atom_type_map.get(center, "X")
                        t3 = atom_type_map.get(a3, "X")

                        # Standardisation de l'ordre canonique des types
                        if t1 > t3:
                            conn_type = f"{t3}-{t2}-{t1}"
                        else:
                            conn_type = f"{t1}-{t2}-{t3}"

                        angles_list.append(
                            {
                                "type": conn_type,
                                "atom_1": a1,
                                "atom_2": center,
                                "atom_3": a3,
                            }
                        )

                # Impropers (B central lié à A, C, D)
                if guess_impropers and len(neighbors) == 3:
                    a, c, d = sorted(neighbors)
                    t_center = atom_type_map.get(center, "X")
                    t_a = atom_type_map.get(a, "X")
                    t_c = atom_type_map.get(c, "X")
                    t_d = atom_type_map.get(d, "X")
                    conn_type = f"{t_center}-{t_a}-{t_c}-{t_d}"

                    impropers_list.append(
                        {
                            "type": conn_type,
                            "atom_1": center,
                            "atom_2": a,
                            "atom_3": c,
                            "atom_4": d,
                        }
                    )

            if guess_angles and angles_list:
                df_angles = pd.DataFrame(angles_list)
                df_angles.index = range(1, len(df_angles) + 1)
                self.angles = df_angles

            if guess_impropers and impropers_list:
                df_imp = pd.DataFrame(impropers_list)
                df_imp.index = range(1, len(df_imp) + 1)
                self.impropers = df_imp

        # 4. Traitement des Dièdres (A - B - C - D)
        if guess_dihedrals:
            dihedrals_list = []
            for _, row in self.bonds.iterrows():
                b, c = int(row["atom_1"]), int(row["atom_2"])
                neighbors_b = graph[b] - {c}
                neighbors_c = graph[c] - {b}

                for a in neighbors_b:
                    for d in neighbors_c:
                        if a != d:
                            t_a = atom_type_map.get(a, "X")
                            t_b = atom_type_map.get(b, "X")
                            t_c = atom_type_map.get(c, "X")
                            t_d = atom_type_map.get(d, "X")
                            conn_type = f"{t_a}-{t_b}-{t_c}-{t_d}"

                            dihedrals_list.append(
                                {
                                    "type": conn_type,
                                    "atom_1": a,
                                    "atom_2": b,
                                    "atom_3": c,
                                    "atom_4": d,
                                }
                            )

            if dihedrals_list:
                df_dih = pd.DataFrame(dihedrals_list)
                df_dih.index = range(1, len(df_dih) + 1)
                self.dihedrals = df_dih

        return self

    def guess_angles(self) -> AtomicSystem:
        """Guess only angles from existing bonds."""
        return self.guess_connections(
            guess_angles=True,
            guess_dihedrals=False,
            guess_impropers=False,
        )

    def guess_dihedrals(self) -> AtomicSystem:
        """Guess only dihedrals from existing bonds."""
        return self.guess_connections(
            guess_angles=False,
            guess_dihedrals=True,
            guess_impropers=False,
        )

    def guess_impropers(self) -> AtomicSystem:
        """Guess only impropers from existing bonds."""
        return self.guess_connections(
            guess_angles=False,
            guess_dihedrals=False,
            guess_impropers=True,
        )

    # Aliases
    set_topo = set_topology
    guess_topo = guess_connections
