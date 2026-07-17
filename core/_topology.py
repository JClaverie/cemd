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

from typing import TYPE_CHECKING, Sequence
from itertools import combinations

import pandas as pd
import numpy as np
import MDAnalysis as mda

from cemd.builders._silicate_helpers import get_interlayer_ca_indices
from .._constants import MASSES_DICT

if TYPE_CHECKING:
    from atomic_system import AtomicSystem

def _build_rule(center: str,
                neighbors: list[tuple],
                new_type: str = None,
                bonds: bool = False,
                angles: bool = False,
                impropers: bool = False) -> dict:
    """
    Build a connectivity rule dictionary.
    
    Parameters
    ----------
    neighbors: List of tuples (sel, cutoff, n, exact, new_type=None)
        - exact: True (exact count) or False (at least n)
    """
    neighbor_dicts = []
    
    for n in neighbors:
        # On décompose le tuple avec la nouvelle position de 'exact'
        sel, cutoff, n_val, exact = n[0], n[1], n[2], n[3]
        new_n_type = n[4] if len(n) > 4 else None
        
        neighbor_dicts.append({
            "sel":      sel,
            "cutoff":   cutoff,
            "n":        n_val,
            "exact":    exact,
            "new_type": new_n_type,
        })
        
    return {
        "center_sel":      center,
        "new_type":        new_type,
        "neighbors":       neighbor_dicts,
        "create_bond":     bonds,
        "create_angle":    angles,
        "create_improper": impropers,
    }

CLAYFF_RULES: list[dict] = [
 
    # ---Silicate oxygens ---
    _build_rule("type O", [("type Si", 1.85, 2, True)],               new_type="Ob"),   # bridging Si-O-Si
    _build_rule("type O", [("type Si", 1.85, 1, True),
                      ("type Al", 1.85, 1, True)],               new_type="Obs"),  # bridging Si-O-Al
    _build_rule("type O",   [("type Si", 1.85, 1, True)],               new_type="Osi"),
    # ---Aluminate oxygens ---
    _build_rule("type O",   [("type Al", 1.85, 1, True)],               new_type="Oa"),
 
    # ---Water molecules (O bonded to exactly 2 H) ---
    _build_rule("type O Oa Osi",
         [("type H", 1.2, 2, False, "Hw")],
         new_type="Ow", bonds=True, angles=True),
 
    # ---Si-O-H silanols ---
    _build_rule("type Osi", [("type H", 1.2, 1, True, "Hsi")],         new_type="Osih",  bonds=True),
 
    # ---Al-O-H aluminols ---
    _build_rule("type Oa",  [("type H", 1.2, 1, True, "Ha")],          new_type="Oah",   bonds=True),
 
    # ---Hydroxide (generic O-H) ---
    _build_rule("type O",   [("type H", 1.2, 1, True, "Hh")],          new_type="Oh",    bonds=True),
 
    # ---Carbonate oxygens ---
    _build_rule("type O",   [("type C", 1.6, 1, True)],                 new_type="Oc"),
 
    # ---Sulfate oxygens ---
    _build_rule("type O",   [("type S", 1.6, 1, True)],                 new_type="Os"),
]

def _apply_single_rule_to_mda(universe: mda.Universe, r: dict) -> mda.Universe:
    """Applies a single geometry-based connectivity rule in-place on a MDAnalysis Universe.
    
    This function avoids converting back and forth to AtomicSystem.
    """
    center_atoms = universe.select_atoms(r["center_sel"])

    new_bonds     = []
    new_angles    = []
    new_impropers = []

    for c in center_atoms:
        matched_neighbors = []   
        neighbor_type_tasks = [] 
        is_valid = True

        for n_rule in r["neighbors"]:
            found = universe.select_atoms(
                f"({n_rule['sel']}) and "
                f"(around {n_rule['cutoff']} index {c.index}) and "
                f"not index {c.index}"
            )

            if n_rule.get("exact", True):
                if len(found) != n_rule["n"]:
                    is_valid = False
                    break
            else:
                if len(found) < n_rule["n"]: # "Au moins n"
                    is_valid = False
                    break

            matched_neighbors.extend(list(found))

            if n_rule.get("new_type") is not None:
                neighbor_type_tasks.append((found, n_rule["new_type"]))

        if not is_valid:
            continue

        if r.get("new_type") is not None:
            c.type = r["new_type"]

        for atoms, n_type in neighbor_type_tasks:
            atoms.types = n_type

        if r.get("create_bond"):
            for n in matched_neighbors:
                new_bonds.append(tuple(sorted((c.index, n.index))))

        if r.get("create_angle") and len(matched_neighbors) >= 2:
            for n1, n2 in combinations(matched_neighbors, 2):
                new_angles.append((n1.index, c.index, n2.index))

        if r.get("create_improper") and len(matched_neighbors) >= 3:
            for n1, n2, n3 in combinations(matched_neighbors, 3):
                new_impropers.append((c.index, n1.index, n2.index, n3.index))

    # Updating the MDAnalysis topology
    def _update_topo(univ, attr, new_entries):
        if not new_entries:
            return
        existing = list(getattr(univ, attr).indices) if hasattr(univ, attr) else []
        combined = list({tuple(e) for e in existing} | {tuple(e) for e in new_entries})
        if hasattr(univ, attr):
            univ.del_TopologyAttr(attr)
        univ.add_TopologyAttr(attr, combined)

    _update_topo(universe, "bonds",     new_bonds)
    _update_topo(universe, "angles",    new_angles)
    _update_topo(universe, "impropers", new_impropers)

    return universe

def _apply_clayff_rules(universe: mda.Universe) -> tuple[mda.Universe, dict]:
    """Applies ClayFF to the universe and flips the universe + no secondary actions."""
    for rule in CLAYFF_RULES:
        universe = _apply_single_rule_to_mda(universe, rule)
    return universe, {}

def _apply_cshff_rules(universe: mda.Universe) -> tuple[mda.Universe, dict]:
    """Applies CSHFF to the universe and returns the calcium indices to modify."""

    universe, _ = _apply_clayff_rules(universe)
    
    list_ids_cw = get_interlayer_ca_indices(universe)
    
    actions = {"rename_atoms": (list_ids_cw, "Cw")}
    return universe, actions

RULE_SETS: dict[str, list[dict]] = {
    "clayff": _apply_clayff_rules,
    "cshff": _apply_cshff_rules,
}

class TopologyMixin:

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
    
    def remove_atoms(self: AtomicSystem, indices: list[int] | int) -> None:
        """Remove the given atoms

        Parameters
        ----------
            indices: list
                Indices of the atoms to remove

        """

        # create a copy of the atoms DataFrame
        df_atoms = self.atoms.copy()
        old_types = self.atom_types

        if isinstance(indices, int):
            indices = [indices]

        indices_valides = [idx for idx in indices if idx in df_atoms.index]

        if not indices_valides:
            print(f"None of the {indices} atoms were found in the current system.")
            return

        # remove atoms that are in the 'indices' list
        df_atoms = df_atoms.drop(indices)

        # create list to remap the atom indices in the DataFrame
        old_ids = df_atoms.index
        new_ids = np.arange(1, len(df_atoms) + 1)
        df_atoms.set_index(new_ids, inplace=True)
        self.atoms = df_atoms

        new_types = self.atom_types

        for i, t in enumerate(old_types):
            if t not in new_types:
                self.masses.pop(i)

        if self.velocities is not None:
            df_vel = self.velocities.copy()
            df_vel = df_vel.drop(indices)
            df_vel.set_index(new_ids, inplace=True)
            self.velocities = df_vel

        id_map = dict(zip(old_ids, new_ids))

        for name, n_cols in [('bonds', 2), ('angles', 3), ('dihedrals', 4), ('impropers', 4)]:
            df = getattr(self, name)
            if df is None:
                continue
            atom_cols = [f'atom_{i}' for i in range(1, n_cols + 1)]
            df = df.loc[~df[atom_cols].isin(indices).any(axis=1)].copy()
            df.index = np.arange(1, len(df) + 1)
            for col in atom_cols:
                df[col] = df[col].map(id_map)
            setattr(self, name, df if len(df) > 0 else None)

    def _remap_connection_types(self, connection_type: str):
        """Generic method to update connection types.
        connection_type must be 'bond', 'angle', 'dihedral' or 'improper'.
        """
        df = getattr(self, connection_type + 's') # Get self.bonds, self.angles, etc.
        if df is None:
            return

        # We dynamically reconstruct the type name from the atom types
        def get_new_type(row):
            # Get the indices of the atoms in the line
            indices = [row[f'atom_{i+1}'] for i in range(len(row) - 1)]
            # Get the type of each atom in the DataFrame self.atoms
            types = [self.atoms.loc[idx, 'type'] for idx in indices]
            return '-'.join(map(str, types))

        # Apply the transformation to the 'type' column
        df['type'] = df.apply(get_new_type, axis=1)
        setattr(self, connection_type + 's', df)

    def set_types(self: AtomicSystem, new_types: Sequence[str | int] | dict[str | int, str | int]) -> AtomicSystem:
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
                raise ValueError("The new list of atom types must match the current number of atom types.")
            mapping = dict(zip(old_types_snapshot, map(str, new_types)))
            
        elif isinstance(new_types, dict):
            for k in new_types.keys():
                if str(k) not in old_types_snapshot:
                    print(f"WARNING: Type '{k}' targeted in set_types is not currently in the system.")
                    
            mapping = {t: t for t in old_types_snapshot}
            for k, v in new_types.items():
                mapping[str(k)] = str(v)
                
        else:
            raise TypeError("new_types must be either a list or a dict.")

        self.atoms['type'] = self.atoms['type'].astype(str).replace(mapping)

        for connection in ['bond', 'angle', 'dihedral', 'improper']:
            self._remap_connection_types(connection)

        new_masses_dic = {}
        for old_name, m in old_masses_dict.items():
            new_name = mapping[old_name]
            if new_name not in new_masses_dic or m > 0:
                new_masses_dic[new_name] = m
        
        self.set_masses(new_masses_dic)
        
        return self
    
    def reset_types(self: AtomicSystem, 
                    prevent: Sequence[str | int]=None) -> AtomicSystem:
        """
        Reset atom types to elements based on masses.
        Finds the closest element in masses_dic to avoid proximity errors (eg: Ar/Ca).
        """

        self._cache = {}

        if prevent is None:
            prevent = []

        new_types = []

        for mass, atype in zip(self.masses, self.atom_types):
            # If the type is protected, we keep it as is
            if atype in prevent:
                new_types.append(atype)
                continue
            
            # look for the element whose mass is closest
            best_element = None
            min_diff = float('inf')

            for element, val in MASSES_DICT.items():
                diff = abs(mass - val)
                if diff < min_diff:
                    min_diff = diff
                    best_element = element
            
            if best_element and (min_diff / mass) < 0.05:
                new_types.append(best_element)
                if (min_diff / mass) > 0.01:
                    print(f"Imprecise match for type {atype} ({mass:.3f} u) -> Assigned to {best_element} (diff: {min_diff:.3f})")
            else:
                new_types.append(str(atype))
                print(f"No credible match for mass {mass:.3f} (type {atype}). Retaining the original.")

        return self.set_types(new_types)

    def set_type2atoms(self: AtomicSystem, 
                    indices: Sequence[int], 
                    atom_type: str | int) -> AtomicSystem:
        
        self._cache = {}

        if not np.issubdtype(type(atom_type), type(self.atom_types[0])):
            raise TypeError("The atom type must be of the same type (string or integer) than existing atom types.")

        data_masses_dic = {t: m for t, m in zip(self.atom_types, self.masses)}
        old_type = self.atoms.loc[indices, 'type'].iloc[0]

        # Edit in place
        self.atoms.loc[indices, 'type'] = atom_type

        updated_atom_types = sorted(self.atoms['type'].unique().tolist())

        for t in updated_atom_types:
            if t not in data_masses_dic:
                if t in MASSES_DICT:
                    data_masses_dic[t] = MASSES_DICT[t]
                else:
                    data_masses_dic[t] = data_masses_dic.get(old_type, 1.0)

        # CHANGE: pass a dict instead of a list to avoid order/length issues
        self.set_masses(data_masses_dic)

        return self
    
    def set_bond(self, atom_list: Sequence[int], bond_type: str | int) -> None:
        """Set a bond between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the two atoms in the bond.
            bond_type
                Type of the bond (numerical or alphabetical)
        """

        self._set_connection('bond', atom_list, bond_type)

    def set_angle(self, atom_list: Sequence[int], angle_type: str | int) -> None:
        """Set a angle between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the angle, in the right order.
            angle_type
                Type of the angle (numerical or alphabetical)
        """

        self._set_connection('angle', atom_list, angle_type)

    def set_dihedral(self, atom_list: Sequence[int], dihedral_type: str | int) -> None:
        """Set a dihedral between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the dihedral, in the right order.
            dihedral_type
                Type of the dihedral (numerical or alphabetical)
        """

        self._set_connection('dihedral', atom_list, dihedral_type)

    def set_improper(self, atom_list: Sequence[int], improper_type: str | int) -> None:
        """Set a improper between two atoms.

        Parameters
        ----------
            atom_list
                list of the indices of the three atoms in the improper, in the right order.
            improper_type
                Type of the improper (numerical or alphabetical)
        """

        self._set_connection('improper', atom_list, improper_type)

    def _set_connection(self: AtomicSystem,
                        connection_class: str, 
                        atom_list: list[int], 
                        connection_type: str | int = None) -> AtomicSystem:
        """
        Factorized version to manage connections dynamically.
        """
        # 1. Setting up connections
        conn_config = {
            'bond':     {'cols': 2, 'target': 'bonds'},
            'angle':    {'cols': 3, 'target': 'angles'},
            'dihedral': {'cols': 4, 'target': 'dihedrals'},
            'improper': {'cols': 4, 'target': 'impropers'}
        }

        if connection_class not in conn_config:
            raise ValueError(f"Unknown connection class: {connection_class}")

        cfg = conn_config[connection_class]
        df_connections = getattr(self, cfg['target'], pd.DataFrame())
        
        # 2. Automatic type logic
        if connection_type is None:
            if isinstance(self.atom_types[0], int):
                raise ValueError(f"Please provide a {connection_class} type in integer format.")
            
            # Type reconstruction based on atom names
            atom_types = [self.atoms.loc[i, 'type'] for i in atom_list]
            connection_type = '-'.join(sorted(atom_types))

        # 3. Preparing the new line
        new_row = {'type': connection_type}
        for i, idx in enumerate(atom_list):
            new_row[f'atom_{i+1}'] = idx
        
        # 4. Existence check (generic)
        atom_cols = [f'atom_{i+1}' for i in range(cfg['cols'])]
        
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
        setattr(self, cfg['target'], df_connections)

        return self
    
    def remove_connection_types(self: AtomicSystem, 
                                bond_types: Sequence[str | int]=None, 
                                angle_types: Sequence[str | int]=None, 
                                dihedral_types: Sequence[str | int]=None, improper_types: Sequence[str | int]=None
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
            remaining_bonds_list = list(set(self.bond_types) - set(bond_types))

            # remove bonds
            self.bonds = self.bonds[~self.bonds.type.isin(bond_types)]

            # update bonds indices
            self.bonds.index = list(range(1, len(self.bonds) + 1))

            # update bonds types range
            self.bonds.type.replace(self.bonds.type.unique(), remaining_bonds_list, inplace=True)

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
            self.angles.type.replace(self.angles.type.unique(), remaining_angles_list, inplace=True)

            # update system info
            if len(self.angles) == 0:
                self.angles = None

        # remove dihedrals
        if len(dihedral_types) != 0:
            # list of remaining dihedrals types
            remaining_dihedrals_list = list(set(self.dihedral_types) - set(dihedral_types))

            # remove dihedrals
            self.dihedrals = self.dihedrals[self.dihedrals.type.isin(dihedral_types)]

            # update dihedrals indices
            self.dihedrals.index = list(range(1, len(self.dihedrals) + 1))

            # update dihedrals types range
            self.dihedrals.type.replace(
                self.dihedrals.type.unique(),
                remaining_dihedrals_list, inplace=True)

            # update system info
            if len(self.dihedrals) == 0:
                self.dihedrals = None

        # remove impropers
        if len(improper_types) != 0:
            # list of remaining improper types
            remaining_impropers_list = list(set(self.improper_types) - set(improper_types))

            # remove impropers
            self.impropers = self.impropers[self.impropers.type.isin(improper_types)]

            # update impropers indices
            self.impropers.index = list(range(1, len(self.impropers) + 1))

            # update impropers types range
            self.impropers.type.replace(
                self.impropers.type.unique(),
                remaining_impropers_list, inplace=True)

            # update system info
            if len(self.impropers) == 0:
                self.impropers = None


    def keep_connection_types(self: AtomicSystem, 
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
            improper_types=impropertypes2remove)

    def remove_all_connections(self) -> None:
        """Remove all bonds, angles, dihedrals and impropers."""

        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None

    def set_topology(self: AtomicSystem, r: dict | str) -> AtomicSystem:
        """Apply a predefined style or a single custom rule to the system."""
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
        else:
            universe = _apply_single_rule_to_mda(universe, r)

        AtomicSystemClass = self.__class__
        new_lmp_data = AtomicSystemClass.from_mda(universe)
        self._replace_internals(new_lmp_data)

        if "rename_atoms" in actions:
            atom_ids, new_type = actions["rename_atoms"]
            self.set_type2atoms(atom_ids, new_type)

        return self
    
    def reset_topology(self, prevent: Sequence[str | int]=None) -> None:
        """Both reset atom types and remove connectivity.

        Parameters
        ----------
            prevent
                Atom types that will be prevented from the reset

        """
        self.remove_all_connections()
        self.reset_types(prevent)

        return self
    
    # Aliases
    set_topo = set_topology
    reset_topo = reset_topology
    
    