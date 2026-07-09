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

from typing import Sequence, TYPE_CHECKING
from itertools import combinations, combinations_with_replacement

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance

from .._utils import grouped_average
from .._constants import MASSES_DICT

if TYPE_CHECKING:
    from atomic_system import AtomicSystem


def set_topology_rule(universe: mda.Universe, rule: dict) -> mda.Universe:
    """Applies a local geometric pattern-matching rule to update atom types and topology.

    This function scans a Universe for specific central atoms, verifies if their local 
    environment meets the specified neighbor criteria (distance cutoff and minimum count),
    and if valid, updates atom types and appends new bonds, angles, or impropers.

    Args:
        universe (mda.Universe): The MDAnalysis Universe to modify.
        rule (dict): A dictionary defining the matching pattern and topology actions.
            Expected structure:
                {
                    "center_sel" (str): MDAnalysis selection string for central atoms
                        (e.g., "type O").
                    "new_type" (str, optional): New atom type for the central atom 
                        if the rule matches.
                    "create_bond" (bool, optional): If True, creates bonds between the 
                        center and all found neighbors.
                    "create_angle" (bool, optional): If True, creates angles for all 
                        neighbor pairs sharing this center.
                    "neighbors" (list of dict): A list of neighbor requirements:
                        [
                            {
                                "sel" (str): Selection string for neighbors (e.g., "type H").
                                "cutoff" (float): Radius threshold in Angstroms.
                                "n" (int): Minimum number of neighbors required within cutoff.
                                "new_type" (str, optional): New atom type for these neighbors.
                            },
                            ...
                        ]
                }

    Returns:
        mda.Universe: The modified MDAnalysis Universe with updated types and topology attributes.

    Raises:
        NameError: If `combinations` is not imported from `itertools`.

    Example:
        >>> water_rule = {
        ...     "center_sel": "type O",
        ...     "new_type": "OW",
        ...     "neighbors": [{"sel": "type H", "cutoff": 1.2, "n": 2, "new_type": "HW"}],
        ...     "create_bond": True,
        ...     "create_angle": True
        ... }
        >>> u = set_topology_rule(u, water_rule)
    """
    
    center_atoms = universe.select_atoms(rule["center_sel"])
    new_bonds = []
    new_angles = []
    new_impropers = []
            

    for c in center_atoms:
        current_motif_neighbors = []
        is_valid = True
        
        # store the neighbors by rule to be able to apply a specific type to them
        neighbor_type_tasks = []

        for n_rule in rule["neighbors"]:
            # Use index {c.index} to find neighbors around the specific central atom
            found = universe.select_atoms(f"({n_rule['sel']}) and (around {n_rule['cutoff']} index {c.index}) and not index {c.index}")
            if len(found) < n_rule["n"]:
                is_valid = False
                break
            
            current_motif_neighbors.extend(list(found))
            # store the information: “these atoms will have to change type if the pattern is complete”
            if "new_type" in n_rule:
                if n_rule["new_type"] is not None:
                    neighbor_type_tasks.append((found, n_rule["new_type"]))

        if is_valid:
            # Change of types
            if "new_type" in rule:
                if rule["new_type"] is not None:
                    c.type = rule["new_type"]
            
            for atoms, n_type in neighbor_type_tasks:
                atoms.types = n_type

            # Bonds
            if rule.get("create_bond"):
                for n in current_motif_neighbors:
                    new_bonds.append(tuple(sorted((c.index, n.index))))

            # Angles
            if rule.get("create_angle") and len(current_motif_neighbors) >= 2:
                for n1, n2 in combinations(current_motif_neighbors, 2):
                    new_angles.append((n1.index, c.index, n2.index))

    
    # Bonds management
    if new_bonds:
        # We recover the existing one (if present)
        existing = list(universe.bonds.indices) if hasattr(universe, 'bonds') else []
        # Merge + Remove duplicates
        combined = list(set([tuple(sorted(b)) for b in existing] + new_bonds))
        # We force the update (MDAnalysis sometimes requires deleting the attr before)
        if hasattr(universe, 'bonds'): universe.del_TopologyAttr('bonds')
        universe.add_TopologyAttr('bonds', combined)

    # Angles management
    if new_angles:
        existing = list(universe.angles.indices) if hasattr(universe, 'angles') else []
        combined = list(set([tuple(a) for a in existing] + new_angles))
        if hasattr(universe, 'angles'): universe.del_TopologyAttr('angles')
        universe.add_TopologyAttr('angles', combined)

    # Impropers managements
    if new_impropers:
        existing = list(universe.impropers.indices) if hasattr(universe, 'impropers') else []
        combined = list(set([tuple(i) for i in existing] + new_impropers))
        if hasattr(universe, 'impropers'): universe.del_TopologyAttr('impropers')
        universe.add_TopologyAttr('impropers', combined)

    return universe


def set_topo_clayff(universe: mda.Universe) -> mda.Universe:
    """Return a MDAnalysis Universe with a topology."""

    def n_inearest(atom, sel, n):
        '''Return the indices of the n atoms with the indices the closest from the index of a given atom.'''

        atom_index = atom.index
        sel_indices = sel.indices

        dif = sel_indices - atom_index
        sorted_dif = np.argsort(dif)
        n_smallest_indices = sorted_dif[:n]  # Get the indices of the n smallest values

        if (n>len(sel)):
            raise ValueError("n cannot be bigger than len(sel).")

        indices = sel.indices[n_smallest_indices]

        return ' '.join([str(i) for i in indices])

    def n_nearest(atom, sel, n, cutoff):
        '''Return the indices of the n nearest atoms within a selection from a given atom.'''

        def get_indices_of_n_smallest_values(arr, n):
            sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
            n_smallest_indices = sorted_indices[:n]  # Get the indices of the n smallest values
            return n_smallest_indices

        if (n>len(sel)):
            raise ValueError("n cannot be bigger than len(sel).")

        _, dist = capped_distance(atom.position, sel.positions,
            box=sel.dimensions, max_cutoff=cutoff)
        
        n_smallest_indices = get_indices_of_n_smallest_values(dist, n)

        indices = sel.indices[n_smallest_indices]

        return ' '.join([str(i) for i in indices])

    bonds = []
    angles = []

    # detect oxygen in silicate
    sel = universe.select_atoms("type O and around 1.85 type Si")
    sel.types = ['Osi'] * len(sel)
    for oxygen in sel:
        selsi = universe.select_atoms(f"type Si and around 1.85 index {oxygen.index}")
        selal = universe.select_atoms(f"type Al and around 1.85 index {oxygen.index}")

        if len(selsi) == 2:
            oxygen.type = 'Ob'
        if len(selsi) == 1 and len(selal) == 1:
            oxygen.type = 'Obs'

    # detect oxygen in aluminate
    sel = universe.select_atoms("type O and around 1.85 type Al")
    sel.types = ['Oa'] * len(sel)

    # detect oxygen and hydrogen in water
    sel = universe.select_atoms("type O Oa Osi and around 1.2 type H")
    for oxygen in sel:
        selh = universe.select_atoms(f"type H and around 1.2 index {oxygen.index}")

        if len(selh) > 2:
            str_nearest_indices = n_inearest(oxygen, selh, 2)
            selh = universe.select_atoms(f"index {str_nearest_indices}")
        if len(selh) == 2:
            selh.types = 'Hw'
            oxygen.type = 'Ow'
            bonds.append((oxygen.index, selh[0].index))
            bonds.append((oxygen.index, selh[1].index))
            angles.append((selh[0].index, oxygen.index, selh[1].index))

    # detect hydrogens in (Si-O-H) groups
    sel = universe.select_atoms("type Osi")
    for oxygen in sel:
        selh = universe.select_atoms(f"type H and around 1.2 index {oxygen.index}")

        if len(selh) == 1:
            selh.types = 'Hsi'
            oxygen.type = 'Osih'
            bonds.append((oxygen.index, selh[0].index))

    # detect hydrogens in (Al-O-H) groups
    sel = universe.select_atoms("type Oa")
    for oxygen in sel:
        selh = universe.select_atoms(f"type H and around 1.2 index {oxygen.index}")

        if len(selh) == 1:
            selh.types = 'Ha'
            oxygen.type = 'Oah'
            bonds.append((oxygen.index, selh[0].index))

    # detect hydrogens in hydroxide
    sel = universe.select_atoms("type O")
    for oxygen in sel:
        selh = universe.select_atoms(f"type H and around 1.2 index {oxygen.index}")

        if len(selh) == 1:
            selh.types = 'Hh'
            oxygen.type = 'Oh'
            bonds.append((oxygen.index, selh[0].index))

    # detect oxygen in carbonate
    sel = universe.select_atoms("type O and around 1.6 type C")
    sel.types = 'Oc'

    # detect oxygen in sulface
    sel = universe.select_atoms("type O and around 1.6 type S")
    sel.types = 'Os'

    universe.add_TopologyAttr('bonds', bonds)
    universe.add_TopologyAttr('angles', angles)

    return universe


def get_ids_cw(universe: mda.Universe) -> np.ndarray:
    """Get the ids of interlayer calcium in a C-S-H structure as MDAnalysis Universe."""

    box = universe.dimensions
    si_sel = universe.select_atoms("type Si")
    sel = universe.select_atoms("all")
    si_posz = si_sel.positions[:,2]

    # loop on tol until len(si_layers_pos) = 4 (cell with 2 pores and 4 silicate layers)
    si_layers_pos, _, list_num = grouped_average(si_posz, 8)
    indices = np.argsort(list_num)[-4:]
    list_pairing = sorted(np.array(si_layers_pos)[indices])
    
    # Get indices of calcium atoms between silicates layers separated by 5 angs. or less
    indices_cw = []
    for i in range( len(list_pairing) - 1 ):
        distance_btw_layers = list_pairing[i+1] - list_pairing[i]
        if distance_btw_layers > 5:
            sel_string = (
                f"(type Ca) and (prop z > {list_pairing[i]}) and "
                f"(prop z < {list_pairing[i+1]})"
                )
            sel = universe.select_atoms(sel_string)
            indices_cw.append(sel.ids)

    # Get indices of calcium atoms between silicates layers separated by 5
    # angströms or less within periodic boundaries
    distance_btw_layers = abs(list_pairing[-1] - list_pairing[0] - box[2])
    if distance_btw_layers > 5:
        sel_string = (
            f"(type Ca) and ((prop z > {list_pairing[-1]}) or "
            f"(prop z < {list_pairing[0]}))"
        )
        sel = universe.select_atoms(sel_string)
        indices_cw.append(sel.ids)

    indices_cw  = np.concatenate(indices_cw)

    return indices_cw

def set_types(atomic_system: AtomicSystem, new_types: Sequence[str | int] | dict[str | int, str | int]) -> AtomicSystem:
    """Assign types to atoms, supporting both full lists and partial dictionaries.

    Parameters
    ----------
    atomic_system
        The atomic system object to modify.
    new_types
        If a list: New types to assign (must match the length of current atom_types).
        If a dict: Partial mapping of {old_type: new_type} to update specific types.
    """
    # Take a snapshot of current types and masses before any modification occurs
    old_types_snapshot = list(map(str, atomic_system.atom_types))
    if isinstance(atomic_system.masses, dict):
        old_masses_dict = {str(k): v for k, v in atomic_system.masses.items()}
    else:
        old_masses_dict = dict(zip(old_types_snapshot, atomic_system.masses))

    # 1. Build a comprehensive mapping from current types to destination types
    if isinstance(new_types, Sequence):
        if len(new_types) != len(old_types_snapshot):
            raise ValueError("The new list of atom types must match the current number of atom types.")
        mapping = dict(zip(old_types_snapshot, map(str, new_types)))
        
    elif isinstance(new_types, dict):
        # Optional IPython check: warn the user if a targeted old type doesn't exist
        for k in new_types.keys():
            if str(k) not in old_types_snapshot:
                print(f"WARNING: Type '{k}' targeted in set_types is not currently in the system.")
                
        # Initialize identity mapping for all types, then update with partial choices
        mapping = {t: t for t in old_types_snapshot}
        for k, v in new_types.items():
            mapping[str(k)] = str(v)
            
    else:
        raise TypeError("new_types must be either a list or a dict.")

    # 2. Remap atom types in the atoms DataFrame
    atomic_system.atoms['type'] = atomic_system.atoms['type'].astype(str).replace(mapping)

    # 3. Remap bonds
    if atomic_system.bonds is not None:
        new_bond_types = []
        for i in atomic_system.bond_types:
            new_type = []
            i, idx1, idx2 = atomic_system.bonds[atomic_system.bonds.type == i].iloc[0]
        
            new_type.append(atomic_system.atoms.loc[idx1, 'type'])
            new_type.append(atomic_system.atoms.loc[idx2, 'type'])

            new_bond_types.append('-'.join(map(str, new_type)))
        
        atomic_system.bonds.type.replace(atomic_system.bond_types, new_bond_types, inplace=True)

    # 4. Remap angles
    if atomic_system.angles is not None:
        new_angle_types = []
        for i in atomic_system.angle_types:
            new_type = []
            i, idx1, idx2, idx3 = atomic_system.angles[atomic_system.angles.type == i].iloc[0]
        
            new_type.append(atomic_system.atoms.loc[idx1, 'type'])
            new_type.append(atomic_system.atoms.loc[idx2, 'type'])
            new_type.append(atomic_system.atoms.loc[idx3, 'type'])

            new_angle_types.append('-'.join(map(str, new_type)))

        atomic_system.angles.type.replace(atomic_system.angle_types, new_angle_types, inplace=True)

    # 5. Remap dihedrals
    if atomic_system.dihedrals is not None:
        new_dihedral_types = []
        for i in atomic_system.dihedral_types:
            new_type = []
            i, idx1, idx2, idx3, idx4 = atomic_system.dihedrals[atomic_system.dihedrals.type == i].iloc[0]
        
            new_type.append(atomic_system.atoms.loc[idx1, 'type'])
            new_type.append(atomic_system.atoms.loc[idx2, 'type'])
            new_type.append(atomic_system.atoms.loc[idx3, 'type'])
            new_type.append(atomic_system.atoms.loc[idx4, 'type'])

            new_dihedral_types.append('-'.join(map(str, new_type)))

        atomic_system.dihedrals.type.replace(atomic_system.dihedral_types, new_dihedral_types, inplace=True)

    # 6. Remap impropers
    if atomic_system.impropers is not None:
        new_improper_types = []
        for i in atomic_system.improper_types:
            new_type = []
            i, idx1, idx2, idx3, idx4 = atomic_system.impropers[atomic_system.impropers.type == i].iloc[0]
        
            new_type.append(atomic_system.atoms.loc[idx1, 'type'])
            new_type.append(atomic_system.atoms.loc[idx2, 'type'])
            new_type.append(atomic_system.atoms.loc[idx3, 'type'])
            new_type.append(atomic_system.atoms.loc[idx4, 'type'])

            new_improper_types.append('-'.join(map(str, new_type)))

        atomic_system.impropers.type.replace(atomic_system.improper_types, new_improper_types, inplace=True)

    # 7. Remap masses safely using the comprehensive mapping dict
    new_masses_dic = {}
    for old_name, m in old_masses_dict.items():
        new_name = mapping[old_name]
        if new_name not in new_masses_dic or m > 0:
            new_masses_dic[new_name] = m
    
    atomic_system.set_masses(new_masses_dic)
    
    return atomic_system

def set_type2atoms(lmp_data: AtomicSystem, 
                   indices: Sequence[int], 
                   atom_type: str | int) -> AtomicSystem:

    if not np.issubdtype(type(atom_type), type(lmp_data.atom_types[0])):
        raise TypeError("The atom type must be of the same type (string or integer) than existing atom types.")

    data_masses_dic = {t: m for t, m in zip(lmp_data.atom_types, lmp_data.masses)}
    old_type = lmp_data.atoms.loc[indices, 'type'].iloc[0]

    # Modification en place
    lmp_data.atoms.loc[indices, 'type'] = atom_type

    updated_atom_types = sorted(lmp_data.atoms['type'].unique().tolist())

    for t in updated_atom_types:
        if t not in data_masses_dic:
            if t in MASSES_DICT:
                data_masses_dic[t] = MASSES_DICT[t]
            else:
                data_masses_dic[t] = data_masses_dic.get(old_type, 1.0)

    # CHANGEMENT : passer un dict au lieu d'une liste pour éviter les problèmes d'ordre/longueur
    lmp_data.set_masses(data_masses_dic)

    return lmp_data

def set_coordinates(lmp_data: AtomicSystem, 
                    index: int, 
                    position: Sequence[float]) -> AtomicSystem:
    """
    Modify the coordinates of a single atom using a vector (x, y, z).
    """
    if len(position) != 3:
        raise ValueError("Position must be an iterable of 3 floats (x, y, z)")

    if index not in lmp_data.atoms.index:
        print(f"Warning: Index {index} not found.")
        return lmp_data

    x, y, z = position
    
    lmp_data.atoms.at[index, 'x'] = float(x)
    lmp_data.atoms.at[index, 'y'] = float(y)
    lmp_data.atoms.at[index, 'z'] = float(z)

    return lmp_data

def reset_types(lmp_data: AtomicSystem, 
                prevent: Sequence[str | int]=None) -> AtomicSystem:
    """
    Reset atom types to elements based on masses.
    Finds the closest element in masses_dic to avoid proximity errors (eg: Ar/Ca).
    """
    if prevent is None:
        prevent = []

    new_types = []

    for mass, atype in zip(lmp_data.masses, lmp_data.atom_types):
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

    return set_types(lmp_data, new_types)

def set_connection(lmp_data: AtomicSystem,
                    connection_class: str, 
                    atom_list: Sequence[int], 
                    connection_type: str | int=None) -> AtomicSystem:
    """Set a connection between a list of atoms.

    Parameters
    ----------
        atom_list
            List of the indices of the two atoms in the bond.
        connection_type
            Type of the connection (numerical or alphabetical)

    """

    df_atoms = lmp_data.atoms

    if connection_class == 'bond' and lmp_data.num_bonds != 0:
        connection_types = lmp_data.bond_types
        df_connections = lmp_data.bonds.copy()
    elif connection_class == 'angle' and lmp_data.num_angles != 0:
        connection_types = lmp_data.angle_types
        df_connections = lmp_data.angles.copy()
    elif connection_class== 'dihedral' and lmp_data.num_dihedrals != 0:
        connection_types = lmp_data.dihedral_types
        df_connections = lmp_data.dihedrals.copy()
    elif connection_class == 'improper' and lmp_data.num_impropers != 0:
        connection_types = lmp_data.improper_types
        df_connections = lmp_data.impropers.copy()
    else:
        connection_types = []
        df_connections = pd.DataFrame()
    
    if connection_type is None:
        if isinstance(lmp_data.atom_types[0], int):
            raise ValueError(f"Please provide a {connection_class} type in the integer format.")

        #TODO: get automatically the write connection type at the integer format and create new one if not exist.

        if isinstance(lmp_data.atom_types[0], str):
            connection_list = []
            for i in atom_list:
                connection_list.append(df_atoms.loc[i].type)

            if len(connection_types) != 0:
                if '-'.join(connection_list) in connection_types:
                    connection_type = '-'.join(connection_list)
                if '-'.join(connection_list.reverse()) in connection_types:
                    connection_type = '-'.join(connection_list)
            else:
                connection_type = '-'.join(sorted(connection_list))

    new_row = {'type': connection_type}
    for i, idx in enumerate(atom_list):
        new_row[f'atom_{i+1}'] = idx

    if connection_class == 'bond' and lmp_data.num_bonds != 0: 
        if ~(((df_connections['atom_1'] == atom_list[0]) & (df_connections['atom_2'] == atom_list[1])) | ((df_connections['atom_1'] == atom_list[1]) & (df_connections['atom_2'] == atom_list[0]))).any():
            new_row_df = pd.DataFrame([new_row])
            df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)
        else:
            print(f"This {connection_class} already exists in the LAMMPSData.")

    elif connection_class == 'angle' and lmp_data.num_angles != 0:
        if ~((df_connections['atom_1'] == atom_list[0]) & (df_connections['atom_2'] == atom_list[1]) & (df_connections['atom_3'] == atom_list[2])).any():
            new_row_df = pd.DataFrame([new_row])
            df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)
        else:
            print(f"This {connection_class} already exists in the LAMMPSData.")

    elif connection_class == 'dihedral' and lmp_data.num_dihedrals != 0:
        if ~((df_connections['atom_1'] == atom_list[0]) & (df_connections['atom_2'] == atom_list[1]) & (df_connections['atom_3'] == atom_list[2]) & (df_connections['atom_4'] == atom_list[3])).any():
            new_row_df = pd.DataFrame([new_row])
            df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)
        else:
            print(f"This {connection_class} already exists in the LAMMPSData.")

    elif connection_class == 'improper' and lmp_data.num_impropers != 0:
        if ~((df_connections['atom_1'] == atom_list[0]) & (df_connections['atom_2'] == atom_list[1]) & (df_connections['atom_3'] == atom_list[2]) & (df_connections['atom_4'] == atom_list[3])).any():
            new_row_df = pd.DataFrame([new_row])
            df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)
        else:
            print(f"This {connection_class} already exists in the LAMMPSData.")

    else:
            new_row_df = pd.DataFrame([new_row])
            df_connections = pd.concat([df_connections, new_row_df], ignore_index=True)
    if isinstance(lmp_data.atom_types[0], int):
        df_connections = df_connections.astype(int)
    else:
        columns_to_convert = df_connections.columns[:-1]
        df_connections[columns_to_convert] = df_connections[columns_to_convert] .astype(int)

    df_connections.index = range(1, len(df_connections) + 1)

    if connection_class == 'bond':
        lmp_data.bonds = df_connections
    if connection_class == 'angle':
        lmp_data.angles = df_connections
    if connection_class == 'dihedral':
        lmp_data.dihedrals = df_connections
    if connection_class == 'improper':
        lmp_data.impropers = df_connections

    return lmp_data

def set_atom_type_param(system: AtomicSystem, 
                        atom_type: str | int, 
                        pair_coeffs: list[float]) -> None:
    """Assigns the non-bond parameters (L-J) for one atom type."""
    # Dynamic extraction of real types present in the DataFrame
    if atom_type not in system.atoms['type'].unique():
        raise ValueError(f"Atom type '{atom_type}' does not exist in the system.")
    
    sorted_key = (atom_type, atom_type)
    system.pair_params[sorted_key] = pair_coeffs

def set_bond_type_param(system: AtomicSystem, 
                        bond_type: str, 
                        coeffs: list[float]) -> None:
    """Assigns the structural parameters [k, r0] for one specific bond type.
    
    Example: set_bond_type_param(system, "H-O", [554.1349, 1.0])
    """
    if not hasattr(system, 'bond_types') or system.bond_types is None:
        raise ValueError("The system does not have any bond types initialized.")
    if bond_type not in system.bond_types:
        raise ValueError(f"Bond type '{bond_type}' does not exist in the system.")
    
    system.bond_params[bond_type] = coeffs

def set_angle_type_param(system: AtomicSystem, 
                         angle_type: str, 
                         coeffs: list[float]) -> None:
    """Assigns the structural parameters [k, theta0] for one specific angle type.
    
    Example: set_angle_type_param(system, "H-O-H", [45.7696, 109.47])
    """
    if not hasattr(system, 'angle_types') or system.angle_types is None:
        raise ValueError("The system does not have any angle types initialized.")
    if angle_type not in system.angle_types:
        raise ValueError(f"Angle type '{angle_type}' does not exist in the system.")
    
    system.angle_params[angle_type] = coeffs


def set_pair_forcefield(
    system: AtomicSystem, 
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
    system.pair_params = {}

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
                set_atom_type_param(system, label1, pair_coeffs)
            else:
                # Custom cross-interaction (i != j) stored as a sorted label tuple key
                sorted_key = tuple(sorted([label1, label2]))
                system.pair_params[sorted_key] = pair_coeffs


def set_bond_forcefield(
    system: AtomicSystem, 
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
    if df_bond is None or df_bond.empty or not hasattr(system, 'bond_types') or system.bond_types is None:
        return

    # Nettoyage des colonnes et des chaînes de la BDD
    df_bond_clean = df_bond.copy()
    df_bond_clean.columns = df_bond_clean.columns.str.strip()
    df_bond_clean['type 1'] = df_bond_clean['type 1'].astype(str).str.strip()
    df_bond_clean['type 2'] = df_bond_clean['type 2'].astype(str).str.strip()

    system.bond_params = {}

    for bond_str in system.bond_types:
        elements = bond_str.split('-')
        if len(elements) != 2: 
            continue
        
        sys_t1, sys_t2 = elements[0].strip(), elements[1].strip()
        
        # REMAPPING : Traduction des types du système vers les types du forcefield
        ff_t1 = atom_type_assignments.get(sys_t1)
        ff_t2 = atom_type_assignments.get(sys_t2)
        
        # Si l'un des deux atomes n'a pas reçu d'assignation de forcefield, on ne peut pas chercher
        if not ff_t1 or not ff_t2:
            continue
            
        # Recherche bidirectionnelle dans la base de données avec les types remappés
        mask = ((df_bond_clean['type 1'] == ff_t1) & (df_bond_clean['type 2'] == ff_t2)) | \
               ((df_bond_clean['type 1'] == ff_t2) & (df_bond_clean['type 2'] == ff_t1))
        row = df_bond_clean[mask]
        
        if not row.empty:
            # Extraction des constantes (k et r)
            coeffs = [float(row.iloc[0].iloc[2]), float(row.iloc[0].iloc[3])]
            
            # Applique les paramètres sur le label d'origine du système (ex: 'H-O')
            set_bond_type_param(system, bond_str, coeffs)

def set_angle_forcefield(
    system: AtomicSystem, 
    atom_type_assignments: dict[str | int, str], 
    df_angle: pd.DataFrame) -> None:
    """
    Automatically maps system angle labels (e.g., 'H-O-H') to their assigned
    forcefield types, checks the database bidirectionally (outer atoms), and applies parameters.
    """
    if df_angle is None or df_angle.empty or not hasattr(system, 'angle_types') or system.angle_types is None:
        return

    df_angle_clean = df_angle.copy()
    df_angle_clean.columns = df_angle_clean.columns.str.strip()
    df_angle_clean['type 1'] = df_angle_clean['type 1'].astype(str).str.strip()
    df_angle_clean['type 2'] = df_angle_clean['type 2'].astype(str).str.strip()
    df_angle_clean['type 3'] = df_angle_clean['type 3'].astype(str).str.strip()

    system.angle_params = {}

    for angle_str in system.angle_types:
        elements = angle_str.split('-')
        if len(elements) != 3: 
            continue
        
        sys_t1, sys_t2, sys_t3 = elements[0].strip(), elements[1].strip(), elements[2].strip()
        
        # Traduction à la volée via le dictionnaire
        ff_t1 = atom_type_assignments.get(sys_t1)
        ff_t2 = atom_type_assignments.get(sys_t2) # Atome central
        ff_t3 = atom_type_assignments.get(sys_t3)
        
        if not ff_t1 or not ff_t2 or not ff_t3:
            continue
            
        # Le type 2 (central) reste au milieu, le sens de lecture 1-3 peut s'inverser
        mask = ((df_angle_clean['type 1'] == ff_t1) & (df_angle_clean['type 2'] == ff_t2) & (df_angle_clean['type 3'] == ff_t3)) | \
               ((df_angle_clean['type 1'] == ff_t3) & (df_angle_clean['type 2'] == ff_t2) & (df_angle_clean['type 3'] == ff_t1))
        row = df_angle_clean[mask]
        
        if not row.empty:
            coeffs = [float(row.iloc[0].iloc[3]), float(row.iloc[0].iloc[4])]
            set_angle_type_param(system, angle_str, coeffs)
