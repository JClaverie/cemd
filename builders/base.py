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
import re
import warnings
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.constants as cst
import MDAnalysis as mda
from pymatgen.core import Composition
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .. import AtomicSystem
from .._config import STRUCTURES_DIR
from .._constants import MASSES_DICT
from .._utils import (
    lattice2lammps, 
    lattice2vectors, 
    vectors2lattice
)
from ._packmol import (
    add_packmol_structure, 
    get_structure_path,
    run_packmol
)

warnings.filterwarnings("ignore")

def _sanitize_path(full_path: str) -> str:
    """
    Takes a full path, sanitizes ONLY the filename part to be 
    Packmol-safe, and returns the full reconstructed absolute path.
    """
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    
    name, ext = os.path.splitext(filename)
    
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    
    safe_filename = f"{name}{ext}"
    return os.path.abspath(os.path.join(directory, safe_filename))

import numpy as np
from scipy.spatial import cKDTree

import numpy as np
import MDAnalysis as mda

def rebuild_silicates(u, si_o_cutoff=1.9, oh_dist=0.96, si_o_dist=1.6):
    """
    Reconstruit la coordination des Si < 4 en ajoutant des groupes silanol (OH).
    Utilise les outils de sélection et de calcul de distance de MDAnalysis.
    """
    new_atoms_data = []
    
    # 1. Sélectionner tous les siliciums
    silicons = u.select_atoms("type Si")
    
    for si in silicons:
        # 2. Compter les oxygènes voisins dans le rayon cutoff (en utilisant PBC)
        # 'around' est l'outil natif de MDAnalysis pour le voisinage
        neighbors = u.select_atoms(f"type O and around {si_o_cutoff} index {si.index}")
        
        n_bonds = len(neighbors)
        
        if n_bonds < 4:
            missing = 4 - n_bonds
            print(f"Si ID {si.id} a {n_bonds} voisins. Ajout de {missing} groupe(s) OH.")
            
            for i in range(missing):
                # Calcul d'une direction simple (vers l'extérieur du tétraèdre)
                # Note: Dans un vrai cas, on calculerait le vecteur opposé au barycentre
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                
                # Position du nouvel Oxygène
                o_pos = si.position + (direction * si_o_dist)
                # Position du nouvel Hydrogène
                h_pos = o_pos + (direction * oh_dist)
                
                new_atoms_data.append({'type': 'O', 'pos': o_pos})
                new_atoms_data.append({'type': 'H', 'pos': h_pos})
                
    return new_atoms_data

def _build_atomic_system_from_coords(positions: np.ndarray, 
                                      types: list[str], 
                                      box: np.ndarray = None) -> AtomicSystem:
    """
    Builds a basic AtomicSystem object directly from atomic coordinates and atom types.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing atomic coordinates (x, y, z).
    types : list[str]
        List of strings specifying the type/name of each atom.
    box : np.ndarray, optional
        Simulation box parameters [a, b, c, alpha, beta, gamma].

    Returns
    -------
    AtomicSystem
        The constructed AtomicSystem instance.
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'type': types,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2]
    })
    
    system = AtomicSystem(df)
    if box is not None:
        system.set_box(box)
    return system

def _cap_broken_framework_bonds(u: mda.Universe,
                                 framework_bonds: list[tuple[int, int]],
                                 si_al_types=("Si", "Al"),
                                 o_bond_length: float = 1.75,
                                 oh_bond_length: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """
    For each broken Si/Al-O bond (si_idx, o_idx), on the ALREADY MOVED
    universe u:
      - add one H on the O side (-> existing O becomes a silanol)
      - add one O+H on the Si/Al side (-> new silanol replacing the lost O)
    Returns arrays of new positions and new (generic) type labels, ready
    to be merged into the final system.
    """
    new_pos = []
    new_types = []

    for a, b in framework_bonds:
        atom_a, atom_b = u.atoms[a], u.atoms[b]
        if atom_a.type in si_al_types:
            si_atom, o_atom = atom_a, atom_b
        else:
            si_atom, o_atom = atom_b, atom_a

        # --- Cap the surviving bridging O with an H ---
        # place the H roughly along the (now vacant) Si->O direction,
        # extended past O
        direction = o_atom.position - si_atom.position
        direction /= np.linalg.norm(direction)
        h_on_o = o_atom.position + direction * oh_bond_length
        new_pos.append(h_on_o)
        new_types.append("H")  # bonded to the existing O -> becomes Osih

        # --- Cap the under-coordinated Si with a new O-H ---
        si_neighbors = u.select_atoms(f"around 2.0 index {si_atom.index}")
        si_neighbors = si_neighbors.select_atoms("type O*")  # remaining O's
        if len(si_neighbors) > 0:
            to_neighbors = si_neighbors.positions - si_atom.position
            to_neighbors /= np.linalg.norm(to_neighbors, axis=1, keepdims=True)
            new_o_dir = -np.sum(to_neighbors, axis=0)
            new_o_dir /= np.linalg.norm(new_o_dir)
        else:
            new_o_dir = -direction  # fallback: opposite of the broken bond

        new_o_pos = si_atom.position + new_o_dir * o_bond_length
        new_h_pos = new_o_pos + new_o_dir * oh_bond_length
        new_pos.extend([new_o_pos, new_h_pos])
        new_types.extend(["Osih", "Hsi"])

    return np.array(new_pos), new_types


def build_surface(data: AtomicSystem, 
                 miller_indices: Sequence[int], 
                 min_slab_size: float=25.0, 
                 min_vacuum_size: float=15.0
                 ) -> tuple[list[AtomicSystem],
                            list[float],
                            list[float],
                            int]:

    if hasattr(data, '_pmg_struct'):
        struct = data._pmg_struct
    else:
        struct = data.to_pmg()
        
    struct.add_oxidation_state_by_guess()
    
    slabgen = SlabGenerator(struct, miller_indices, min_slab_size, min_vacuum_size)

    bonds_dic = {
        ('Al4+', 'O2-') : 1.8,
        ('Al', 'O'): 1.8,
        ('Si4+', 'O2-') : 1.8,
        ('Si', 'O') : 1.8,
        ('C4+', 'O2-') : 1.4,
        ('C', 'O') : 1.4, 
        ('H+', 'O2-') : 1.1,
        ('H', 'O') : 1.1
    }

    # Finding the minimum number of broken links
    i = 0
    slabs_list_pmg = []
    while len(slabs_list_pmg) == 0 and i < 20:
        slabs_list_pmg = slabgen.get_slabs(bonds=bonds_dic, 
                                       max_broken_bonds=i)
        if len(slabs_list_pmg) > 0:
            actual_broken = i
            break
        i += 1
    
    dipole_list = []
    shift_list = []
    for slab in slabs_list_pmg:
        dipole_list.append(abs(slab.dipole[2]))
        shift_list.append(slab.shift)

    slabs_list_as = []
    for i, raw_slab in enumerate(slabs_list_pmg):
        analyzer = SpacegroupAnalyzer(raw_slab)
        slab = analyzer.get_refined_structure()
        slabs_list_as.append(AtomicSystem.from_pymatgen(slab))
    
    return slabs_list_as, shift_list, dipole_list, actual_broken

def build_solution(box: Sequence[float] | np.ndarray = [30,30,30], 
                  density: float=1.0, 
                  solutes_dict: dict[str: int]=None,
                  structures_dict: dict[str: AtomicSystem]=None
                  ) -> AtomicSystem:
    """
    Generate an AtomicSystem representing a liquid solution using Packmol.

    This function automatically determines the number of solvent molecules (H2O) required to reach the target density, accounting for the mass of all 
    provided solutes.

    Parameters
    ----------
    box
        Dimensions of the simulation box in Angstroms [a, b, c]. 
        Defaults to [30, 30, 30].
    density
        Target density of the solution in g/cm³. Defaults to 1.0.
    solutes_dict
        Dictionary mapping solute names (keys) to their respective counts (values). 
        Keys must match either known monatomic ions or entries in structures_dict.
    structures_dict
        Dictionary mapping solute names to their corresponding AtomicSystem objects for complex molecules or custom structures.

    Returns
    -------
    AtomicSystem
        The final system geometry generated by Packmol, with configured 
        box parameters and topology.

    Raises
    ------
    ValueError
        If a solute key is neither a known ion nor found in structures_dict, 
        or if the total mass of solutes exceeds the total mass allowed by 
        the target density and box volume.

    Notes
    -----
    The function currently assumes water (H2O) as the default solvent. 
    Packmol constraints are applied using a 5% safety margin on the box dimensions.
    """

    boxa, boxb, boxc = box[:3]
    vol = boxa * boxb * boxc

    mass_solutes_total_uma = 0
    if solutes_dict:
        for key, num in solutes_dict.items():
            if structures_dict and key in structures_dict:
                mass_solutes_total_uma += num * structures_dict[key].total_mass
            elif key in MASSES_DICT:
                mass_solutes_total_uma += num * MASSES_DICT[key]
            else:
                raise ValueError(f"The structure '{key}' is neither a known ion nor an open tab.")
            
    # calculate the target total mass
    mass_total_target_g = density * vol * 1e-24 
    
    # Convert mass of ions to grams
    mass_ions_g = (mass_solutes_total_uma / cst.Avogadro)
    
    # Deduce the remaining mass of water
    mass_water_g = mass_total_target_g - mass_ions_g
    
    if mass_water_g <= 0:
        raise ValueError("The density or volume is too low to hold all these ions!")

    # Number of water molecules
    water_molarmass = 2 * MASSES_DICT['H'] + MASSES_DICT['O']
    num_water = round((mass_water_g * cst.Avogadro) / water_molarmass)

    with tempfile.TemporaryDirectory(dir='.') as tmp:
        # write packmol input
        structures = []

        h2o_path = get_structure_path('H2O', tmp)
        structures.append(add_packmol_structure(
            h2o_path, num_water, 
             (f"inside box 0 0 0 {boxa * 0.95:.4f} {boxb * 0.95:.4f} "
            f"{boxc * 0.95:.4f}")))

        if solutes_dict:
            for key, num in solutes_dict.items():
                if structures_dict and key in structures_dict:
                    safe_path = f"custom_{id(structures_dict[key])}.pdb"
                    struct_path = os.path.join(tmp, safe_path)
                    structures_dict[key].write(struct_path)

                else:
                    struct_path = get_structure_path(key, tmp)

                struct_path = _sanitize_path(struct_path)
                structures.append(add_packmol_structure(
                    struct_path, num,
                        "center",
                        (f"inside box 0 0 0 {boxa*0.95:.4f} {boxb*0.95:.4f} "
                        f"{boxc*0.95:.4f}")))

        data = run_packmol(structures)

    # Print box parameters and number of atoms
    print("\nSolution created")
    print(f"Box size: {boxa:.1f} Å x {boxb:.1f} Å x {boxc:.1f} Å")
    print(f"{num_water} water molecules")
    if solutes_dict:
        for key, num in solutes_dict.items():
            print(f"{num} {key}")

    # Set the box parameters
    if not isinstance(box, list):
        box = box.tolist()

    data.set_box(box + [90, 90, 90])

    return data

def build_glass(box: Sequence[float] | np.ndarray, 
               density: float, 
               stoichiometry_dic: dict[str, float]) -> AtomicSystem:
    """
    Creates an atomic system of glass from a composition of oxides, molecules or atoms.
    """
    
    boxa, boxb, boxc = box
    vol = boxa * boxb * boxc

    total_pattern_mass = 0.0
    atomic_composition = {} 

    for formula, coeff in stoichiometry_dic.items():

        pdb_molecule = os.path.join(STRUCTURES_DIR, f"{formula.lower()}.pdb")
        
        if os.path.exists(pdb_molecule):
            # calculate the mass of the molecule via Pymatgen for the density
            comp = Composition(formula)
            m_part = sum(MASSES_DICT.get(el, 0) * amt for el, amt in comp.get_el_amt_dict().items())
            
            atomic_composition[formula] = atomic_composition.get(formula, 0) + coeff
            total_pattern_mass += m_part * coeff
            
        else:
            # No global PDB, we decompose (e.g.: SiO2, Al2O3)
            try:
                breakdown = Composition(formula).as_dict()
                m_part = 0.0
                for element, amount in breakdown.items():
                    if element not in MASSES_DICT:
                       raise ValueError(f"Missing mass for: {element}")
                    
                    m_part += amount * MASSES_DICT[element]
                    # insert each element separately (eg: “If”, “O”)
                    atomic_composition[element] = atomic_composition.get(element, 0) + (amount * coeff)
                
                total_pattern_mass += m_part * coeff
            except Exception:
                raise ValueError(f"Component '{formula}' not found (no PDB or valid formula)")

    # Calculation of the number of units for the target density
    # n = (rho *V *Na) /M
    num_units = int(np.round((density * vol * 1e-24 * cst.Avogadro) / total_pattern_mass))
    if num_units == 0: num_units = 1

    rho_real = (num_units * total_pattern_mass) / (cst.Avogadro * vol * 1e-24)
    err = (rho_real - density) / density

    # Calling Packmol via temporary directory
    with tempfile.TemporaryDirectory(dir='.') as tmp:

        # Packmol config
        structures = []

        for comp, coeff_in_motif in atomic_composition.items():
            count = int(np.round(coeff_in_motif * num_units))
            if count <= 0: continue
            
            print(f" -> {comp:6s}: {count} unités")
            
            path2structure = get_structure_path(comp, tmp)
            
            structures.append(add_packmol_structure(
                path2structure, count,
                "center",
                f"inside box 1 1 1 {boxa-1:.4f} {boxb-1:.4f} {boxc-1:.4f}\n"
            ))

        data = run_packmol(structures)
        
    # Finalization
    data.set_box(box + [90, 90, 90])
    
    print(f"\nGlass structure generated ({density} g/cm3)")
    print(f"Real density: {rho_real:.3f} g/cm3 (Error: {err*100:.1f}%)")
    

    return data

def add_droplet(solid_system: AtomicSystem, 
                radius: float, 
                density: float = 1.0,
                distance: float = 2.0,
                solutes_dict: dict[str, int] = None, 
                structures_dict: dict[str, AtomicSystem] = None,
                vacuum: float = 10.0) -> AtomicSystem:
    """
    Adds a hemispherical droplet of solution onto a solid surface using 
    add_structure for positioning and merging.
    """
    if isinstance(solid_system, str):
        solid_system = AtomicSystem.from_file(solid_system)

    vol = 2 * np.pi * radius**3 / 3

    mass_solutes_total_uma = 0
    if solutes_dict:
        for key, num in solutes_dict.items():
            if structures_dict and key in structures_dict:
                mass_solutes_total_uma += num * structures_dict[key].total_mass
            elif key in MASSES_DICT:
                mass_solutes_total_uma += num * MASSES_DICT[key]
            else:
                raise ValueError(f"Structure '{key}' mass unknown.")

    mass_total_target_g = density * vol * 1e-24 
    mass_solutes_g = (mass_solutes_total_uma / cst.Avogadro)
    mass_water_g = mass_total_target_g - mass_solutes_g
    
    if mass_water_g <= 0:
        raise ValueError("Density or volume too low for these ions/solutes!")

    water_molarmass = 2 * MASSES_DICT['H'] + MASSES_DICT['O']
    num_water = round((mass_water_g * cst.Avogadro) / water_molarmass)

    with tempfile.TemporaryDirectory(dir='.') as tmp:
        structures = []
        h2o_path = get_structure_path("H2O", tmp)
        
        structures.append(add_packmol_structure(h2o_path, num_water, 
            f"inside sphere 0. 0. 0. {radius:.4f}",
            "over plane 0. 0. 1. 0."
        ))

        if solutes_dict:
            for key, num in solutes_dict.items():
                if structures_dict and key in structures_dict:
                    struct_path = os.path.join(tmp, f"custom_{id(structures_dict[key])}.pdb")
                    structures_dict[key].write(struct_path)
                else:
                    struct_path = get_structure_path(key, tmp)

                struct_path = _sanitize_path(struct_path)
                structures.append(add_packmol_structure(struct_path, num, 
                        f"inside sphere 0. 0. 0. {radius:.4f}",
                        "over plane 0. 0. 1. 0."
                    ))

        liquid_data = run_packmol(structures)

    data = add_structure(
        solid_system=solid_system, 
        structure_to_add=liquid_data, 
        distance=distance, 
        axis='z', 
        vacuum=vacuum
    )

    print(f"Droplet added: {num_water} H2O, Radius: {radius} A, Density: {density}")
    
    return data

def add_structure(solid_system: AtomicSystem, 
                  structure_to_add: AtomicSystem, 
                  distance: float = 2.0, 
                  axis: str = 'z',
                  vacuum: float = 10.0) -> AtomicSystem:
    """
    Adds a structure to a solid surface at a given distance, 
    aligning their Centers of Mass (COM) on the transverse axes.
    """
    if isinstance(solid_system, str):
        solid_system = AtomicSystem.from_file(solid_system)
    if isinstance(structure_to_add, str):
        structure_to_add = AtomicSystem.from_file(structure_to_add)
    else:
        structure_to_add = structure_to_add.copy()

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_names = ['x', 'y', 'z']
    idx = axis_map[axis.lower()]
    axis_name = axis_names[idx]
    
    trans_indices = [i for i in [0, 1, 2] if i != idx]
    com_solid = solid_system.get_center_of_mass()
    com_struct = structure_to_add.get_center_of_mass()

    for i in trans_indices:
        t_name = axis_names[i]
        shift_trans = com_solid[i] - com_struct[i]
        structure_to_add.atoms[t_name] += shift_trans

    solid_surface = solid_system.atoms[axis_name].max()
    struct_base = structure_to_add.atoms[axis_name].min()
    
    shift_main = (solid_surface - struct_base) + distance
    structure_to_add.atoms[axis_name] += shift_main

    new_box = solid_system.box.copy()
    solid_min = solid_system.atoms[axis_name].min()
    struct_top = structure_to_add.atoms[axis_name].max()
    
    new_box[idx] = (struct_top - solid_min) + vacuum + distance

    data = merge(solid_system, structure_to_add, new_box)
    data.wrap()

    print(f"Structure added on {axis_name}:")
    
    return data

def add_liquid(solid_system: AtomicSystem, 
               thickness: float, 
               density: float = 1.0,
               distance: float = 2.0,
               solutes_dict: dict[str, int] = None,
               structures_dict: dict[str, AtomicSystem] = None,
               vacuum: float = 0.0, 
               axis: str = 'z') -> AtomicSystem:
    """
    Creates a solid/liquid interface by generating a liquid block and 
    positioning it using the add_structure logic.
    """
    if isinstance(solid_system, str):
        solid_system = AtomicSystem.from_file(solid_system)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_map[axis.lower()]
    transverse_indices = [i for i in [0, 1, 2] if i != idx]

    boxv_solid = lattice2vectors(solid_system.box)
    dim_transverse_1 = boxv_solid[transverse_indices[0]][transverse_indices[0]]
    dim_transverse_2 = boxv_solid[transverse_indices[1]][transverse_indices[1]]

    liquid_dims = [0, 0, 0]
    liquid_dims[transverse_indices[0]] = dim_transverse_1
    liquid_dims[transverse_indices[1]] = dim_transverse_2
    liquid_dims[idx] = thickness
    
    liquid_data = build_solution(
        box=[liquid_dims[0], liquid_dims[1], liquid_dims[2]], 
        density=density, 
        solutes_dict=solutes_dict, 
        structures_dict=structures_dict
    )

    return add_structure(
        solid_system=solid_system, 
        structure_to_add=liquid_data, 
        distance=distance, 
        axis=axis, 
        vacuum=vacuum
    )

# def split(solid_system: AtomicSystem, 
#           axis=2,
#           coordinate: float=None,
#           gap_size: float=20.0,
#           tolerance: float=2.0,
#           add_solution: bool=True, 
#           density: float=1.0,
#           solutes_dict: dict[str, int]=None,
#           structures_dict: dict[str, AtomicSystem]=None) -> AtomicSystem:
#     """
#     Splits the atomic system along the specified axis at a given coordinate and optionally inserts a liquid solution in the created gap.
#     """
#     if isinstance(solid_system, str):
#         solid_system = AtomicSystem.from_file(solid_system)

#     # GEOMETRY PREPARATION
#     vec_list = list(lattice2vectors(solid_system.box))
#     target_vec = vec_list[axis]
#     unit_norm = target_vec / np.linalg.norm(target_vec)
#     L_original = np.linalg.norm(target_vec)
    
#     if coordinate is None:
#         coordinate = L_original / 2.0
    
#     # SPLIT SOLID (Using MDAnalysis fragments to keep molecules intact)
#     u = solid_system.to_mda()
#     shift_vector = unit_norm * gap_size
    
#     # Verification: Are there any connections in the system?
#     has_bonds = hasattr(u.atoms, 'bonds') and len(u.atoms.bonds) > 0

#     if has_bonds:
#         # CASE 1: move by fragments (molecules)
#         print("Bonds detected: movement by molecular fragments.")
#         for frag in u.atoms.fragments:
#             if np.dot(frag.centroid(), unit_norm) >= coordinate:
#                 frag.positions += shift_vector
#     else:
#         # CASE 2: No bonds -> move the atoms individually
#         print("No bonds detected: atom by atom movement.")
#         # Selection of atoms whose projected position is beyond the split
#         mask = np.dot(u.atoms.positions, unit_norm) >= coordinate
#         u.atoms.positions[mask] += shift_vector
            
#     # Update box dimensions
#     vec_list[axis] = target_vec + shift_vector
#     extended_box = vectors2lattice(tuple(vec_list))
    
#     # Create the expanded solid system
#     final_system = AtomicSystem.from_mda(u)
#     final_system.set_box(extended_box)

#     # ADD SOLUTION (Only if requested)
#     if add_solution:
#         dims = [np.linalg.norm(v) for v in vec_list]
#         liquid_thickness = gap_size - (2 * tolerance)
        
#         if liquid_thickness <= 0:
#             raise ValueError(f"Gap size ({gap_size} Å) is too small for the requested tolerance ({tolerance} Å x 2).")
        
#         # Determine transverse dimensions for the liquid box
#         other_axes = [i for i in range(3) if i != axis]
#         liquid_box_dims = [dims[other_axes[0]], dims[other_axes[1]], liquid_thickness]
        
#         # Generate the solution (ensure make_solution is imported)
#         liquid_data = make_solution(liquid_box_dims, density, solutes_dict, structures_dict)

#         # Positioning: Center the liquid in the middle of the new gap
#         gap_center = coordinate + (gap_size / 2.0)
        
#         # Project liquid positions on the separation axis
#         axis_name = ['x', 'y', 'z'][axis]
#         liquid_pos = liquid_data.atoms[axis_name].values
#         liquid_center = (np.max(liquid_pos) + np.min(liquid_pos)) / 2.0
        
#         # Apply shift to center the liquid block
#         liquid_data.atoms[axis_name] += (gap_center - liquid_center)

#         # Merge solid and liquid (ensure merge is imported)
#         final_system = merge(final_system, liquid_data, extended_box)
        
#     return final_system



# def split(solid_system: AtomicSystem, 
#           axis=2,
#           coordinate: float=None,
#           gap_size: float=20.0,
#           tolerance: float=2.0,
#           add_solution: bool=True, 
#           density: float=1.0,
#           solutes_dict: dict[str, int]=None,
#           structures_dict: dict[str, AtomicSystem]=None) -> AtomicSystem:
#     """
#     Splits the atomic system along the specified axis at a given coordinate 
#     and optionally inserts a liquid solution in the created gap.
#     """
#     if isinstance(solid_system, str):
#         solid_system = AtomicSystem.from_file(solid_system)

#     # GEOMETRY PREPARATION
#     vec_list = list(lattice2vectors(solid_system.box))
#     target_vec = vec_list[axis]
#     unit_norm = target_vec / np.linalg.norm(target_vec)
#     L_original = np.linalg.norm(target_vec)
    
#     if coordinate is None:
#         coordinate = L_original / 2.0
    
#     u = solid_system.to_mda()
#     shift_vector = unit_norm * gap_size
    
#     has_bonds = hasattr(u.atoms, 'bonds') and len(u.atoms.bonds) > 0
#     broken_framework_bonds = []

#     if has_bonds:
#         print("Bonds detected: filtering boundary-crossing bonds...")
        
#         # 1. Sélectionner les liaisons qui traversent le plan de coupe
#         pos1 = u.atoms.bonds.atom1.positions
#         pos2 = u.atoms.bonds.atom2.positions
        
#         proj1 = np.dot(pos1, unit_norm)
#         proj2 = np.dot(pos2, unit_norm)
        
#         # Une liaison traverse le plan si un atome est < coordinate et l'autre >= coordinate
#         crosses_plane = ((proj1 < coordinate) & (proj2 >= coordinate)) | \
#                         ((proj1 >= coordinate) & (proj2 < coordinate))
        
#         # 2. Conserver uniquement les liaisons qui NE traversent PAS la coupure
#         valid_bonds = u.atoms.bonds[~crosses_plane]
        
#         # Reconstruire la topologie temporaire sans les liaisons traversantes
#         u.delete_bonds(u.atoms.bonds)
#         u.add_bonds(valid_bonds.to_indices())

#         # 3. Déplacer par fragments (désormais séparés au niveau du plan)
#         print("Moving fragments...")
#         for frag in u.atoms.fragments:
#             if np.dot(frag.centroid(), unit_norm) >= coordinate:
#                 frag.positions += shift_vector
#     else:
#         # CASE 2: Pas de liaisons -> Déplacement atome par atome
#         print("No bonds detected: atom by atom movement.")
#         mask = np.dot(u.atoms.positions, unit_norm) >= coordinate
#         u.atoms.positions[mask] += shift_vector
            
#     # Mise à jour de la boîte
#     vec_list[axis] = target_vec + shift_vector
#     extended_box = vectors2lattice(tuple(vec_list))
    
#     final_system = AtomicSystem.from_mda(u)
#     final_system.set_box(extended_box)

#     # ADD SOLUTION
#     if add_solution:
#         dims = [np.linalg.norm(v) for v in vec_list]
#         liquid_thickness = gap_size - (2 * tolerance)
        
#         if liquid_thickness <= 0:
#             raise ValueError(f"Gap size ({gap_size} Å) is too small for requested tolerance.")
        
#         other_axes = [i for i in range(3) if i != axis]
#         liquid_box_dims = [dims[other_axes[0]], dims[other_axes[1]], liquid_thickness]
        
#         liquid_data = make_solution(liquid_box_dims, density, solutes_dict, structures_dict)

#         gap_center = coordinate + (gap_size / 2.0)
        
#         axis_name = ['x', 'y', 'z'][axis]
#         liquid_pos = liquid_data.atoms[axis_name].values
#         liquid_center = (np.max(liquid_pos) + np.min(liquid_pos)) / 2.0
        
#         liquid_data.atoms[axis_name] += (gap_center - liquid_center)

#         final_system = merge(final_system, liquid_data, extended_box)
        
#     return final_system

# def split(solid_system: AtomicSystem,
#           axis: int = 2,
#           coordinate: float = None,
#           gap_size: float = 20.0,
#           tolerance: float = 2.0,
#           add_solution: bool = False,
#           density: float = 1.0,
#           solutes_dict: dict[str, int] = None,
#           structures_dict: dict[str, AtomicSystem] = None,
#           topo_style: str = 'cshff') -> AtomicSystem:
#     """
#     Splits a solid system along a specified axis at a given coordinate,
#     caps severed framework bonds (Si/Al - O) with silanol groups, and optionally
#     inserts a liquid solution into the created gap.

#     Parameters
#     ----------
#     solid_system : AtomicSystem or str
#         The original solid system object or file path.
#     axis : int, optional
#         Cutting axis (0 for X, 1 for Y, 2 for Z). Default is 2.
#     coordinate : float, optional
#         Coordinate position along the axis where the cut is made.
#         Defaults to the midpoint of the simulation box.
#     gap_size : float, optional
#         Width of the created gap in Angstroms (Å). Default is 20.0.
#     tolerance : float, optional
#         Safety distance buffer between liquid and solid surfaces in Å. Default is 2.0.
#     add_solution : bool, optional
#         If True, fills the generated gap with a liquid solution. Default is False.
#     density : float, optional
#         Target liquid density in g/cm³. Default is 1.0.
#     solutes_dict : dict, optional
#         Dictionary mapping solute names to their requested counts.
#     structures_dict : dict, optional
#         Dictionary containing solute AtomicSystem structures.
#     topo_style : str, optional
#         Topology style used to rebuild bonds ('cshff', etc.). Default is 'cshff'.

#     Returns
#     -------
#     AtomicSystem
#         The final split, capped, and optionally solvated atomic system.
#     """
#     if isinstance(solid_system, str):
#         solid_system = AtomicSystem.from_file(solid_system)

#     # --- 1. GEOMETRY PREPARATION ---
#     vec_list   = list(lattice2vectors(solid_system.box))
#     target_vec = vec_list[axis]
#     unit_norm  = target_vec / np.linalg.norm(target_vec)
#     L_original = np.linalg.norm(target_vec)

#     if coordinate is None:
#         coordinate = L_original / 2.0

#     shift_vector = unit_norm * gap_size

#     # --- 2. EXTEND BOX AND MOVE UPPER FRAGMENT ---
#     u = solid_system.to_mda()

#     # Extend the box first so positions stay meaningful
#     vec_list[axis] = target_vec + shift_vector
#     extended_box   = vectors2lattice(tuple(vec_list))
#     u.dimensions   = np.array([*[np.linalg.norm(v) for v in vec_list],
#                                 *solid_system.box[3:]], dtype=float)

#     has_bonds = hasattr(u.atoms, 'bonds') and len(u.atoms.bonds) > 0

#     if has_bonds:
#         print("Bonds detected: moving fragments...")
#         for frag in u.atoms.fragments:
#             if np.dot(frag.centroid(), unit_norm) >= coordinate:
#                 frag.positions += shift_vector
#     else:
#         print("No bonds detected: moving atom by atom.")
#         mask = np.dot(u.atoms.positions, unit_norm) >= coordinate
#         u.atoms.positions[mask] += shift_vector

#     # --- 3. DETECT BROKEN FRAMEWORK BONDS BY STRETCHED LENGTH ---
#     broken_framework_bonds = []

#     if has_bonds:
#         pos1 = u.atoms.bonds.atom1.positions
#         pos2 = u.atoms.bonds.atom2.positions

#         # Real Euclidean distances after displacement — no PBC correction needed
#         # because the box is already extended and atoms are at absolute positions
#         lengths = np.linalg.norm(pos2 - pos1, axis=1)

#         # Framework bond types: Si/Al bonded to any oxygen
#         types1 = u.atoms.bonds.atom1.types.astype(str)
#         types2 = u.atoms.bonds.atom2.types.astype(str)

#         framework_types = {"Si", "Al"}
#         atom1_is_fw = np.isin(types1, list(framework_types))
#         atom2_is_fw = np.isin(types2, list(framework_types))
#         atom1_is_o  = np.char.startswith(types1, "O")
#         atom2_is_o  = np.char.startswith(types2, "O")

#         is_framework_bond = (atom1_is_fw & atom2_is_o) | (atom2_is_fw & atom1_is_o)

#         # A normal Si-O / Al-O bond is 1.6–1.85 Å.
#         # After moving one fragment by gap_size, a severed bond will be ~gap_size Å long.
#         # A threshold of 2.5 Å safely separates intact from broken bonds.
#         STRETCH_THRESHOLD = 2.5
#         is_stretched = lengths > STRETCH_THRESHOLD

#         broken_mask = is_framework_bond & is_stretched

#         if broken_mask.any():
#             broken_framework_bonds = u.atoms.bonds[broken_mask].to_indices()
#             print(f"Found {len(broken_framework_bonds)} broken framework bonds.")

#             # Remove broken bonds from the topology
#             valid_bonds = u.atoms.bonds[~broken_mask]
#             u.delete_bonds(u.atoms.bonds)
#             u.add_bonds(valid_bonds.to_indices())

#     # --- 4. CAP BROKEN FRAMEWORK BONDS ---
#     if len(broken_framework_bonds) > 0:
#         print(f"CAPPING: Adding silanol groups to {len(broken_framework_bonds)} broken bonds...")

#         new_pos, new_types = _cap_broken_framework_bonds(u, broken_framework_bonds)

#         n_capping  = len(new_types)
#         capping_u  = mda.Universe.empty(n_capping, trajectory=True)
#         capping_u.add_TopologyAttr('id',   np.arange(1, n_capping + 1))
#         capping_u.add_TopologyAttr('type', new_types)
#         capping_u.add_TopologyAttr('name', new_types)
#         capping_u.atoms.positions = new_pos
#         capping_u.add_TopologyAttr(
#             'mass', [MASSES_DICT.get(t, 1.0) for t in new_types]
#         )

#         u = mda.Merge(u.atoms, capping_u.atoms)
#         u.atoms.ids = np.arange(1, len(u.atoms) + 1)

#     # --- 5. CONVERT BACK TO AtomicSystem ---
#     final_system = AtomicSystem.from_mda(u)
#     final_system.set_box(extended_box)

#     # --- 6. INSERT LIQUID PHASE (OPTIONAL) ---
#     if add_solution:
#         print("Generating and inserting liquid solution...")
#         dims = [np.linalg.norm(v) for v in vec_list]
#         liquid_thickness = gap_size - 2 * tolerance

#         if liquid_thickness <= 0:
#             raise ValueError(
#                 f"Gap size ({gap_size} Å) is too small for the requested "
#                 f"tolerance ({tolerance} Å)."
#             )

#         other_axes     = [i for i in range(3) if i != axis]
#         liquid_box_dims = [dims[other_axes[0]], dims[other_axes[1]], liquid_thickness]
#         liquid_data    = build_solution(liquid_box_dims, density, solutes_dict, structures_dict)

#         gap_center  = coordinate + gap_size / 2.0
#         axis_name   = ['x', 'y', 'z'][axis]
#         liquid_pos  = liquid_data.atoms[axis_name].values
#         liquid_center = (np.max(liquid_pos) + np.min(liquid_pos)) / 2.0
#         liquid_data.atoms[axis_name] += gap_center - liquid_center

#         final_system = merge(final_system, liquid_data, extended_box)

#     return final_system

# def split(solid_system: AtomicSystem, 
#           axis: int = 2,
#           coordinate: float = None,
#           gap_size: float = 20.0,
#           tolerance: float = 2.0,
#           add_solution: bool = False, 
#           density: float = 1.0,
#           solutes_dict: dict[str, int] = None,
#           structures_dict: dict[str, AtomicSystem] = None) -> AtomicSystem:
#     """
#     Splits a solid system along a specified axis at a given coordinate,
#     caps severed framework bonds (Si/Al - O) with silanol groups, and optionally
#     inserts a liquid solution into the created gap.

#     Parameters
#     ----------
#     solid_system : AtomicSystem or str
#         The original solid system object or file path.
#     axis : int, optional
#         Cutting axis (0 for X, 1 for Y, 2 for Z). Default is 2.
#     coordinate : float, optional
#         Coordinate position along the axis where the cut is made. 
#         Defaults to the midpoint of the simulation box.
#     gap_size : float, optional
#         Width of the created gap in Angstroms (Å). Default is 20.0.
#     tolerance : float, optional
#         Safety distance buffer between liquid and solid surfaces in Å. Default is 2.0.
#     add_solution : bool, optional
#         If True, fills the generated gap with a liquid solution. Default is True.
#     density : float, optional
#         Target liquid density in g/cm³. Default is 1.0.
#     solutes_dict : dict, optional
#         Dictionary mapping solute names to their requested counts.
#     structures_dict : dict, optional
#         Dictionary containing solute AtomicSystem structures.
#     topo_style : str, optional
#         Topology style used to rebuild bonds ('cshff', etc.). Default is 'cshff'.

#     Returns
#     -------
#     AtomicSystem
#         The final split, capped, and optionally solvated atomic system.
#     """
#     if isinstance(solid_system, str):
#         solid_system = AtomicSystem.from_file(solid_system)

#     # 1. GEOMETRY AND BOX PREPARATION
#     vec_list = list(lattice2vectors(solid_system.box))
#     target_vec = vec_list[axis]
#     unit_norm = target_vec / np.linalg.norm(target_vec)
#     L_original = np.linalg.norm(target_vec)
    
#     if coordinate is None:
#         coordinate = L_original / 2.0
    
#     u = solid_system.to_mda()
#     shift_vector = unit_norm * gap_size
    
#     has_bonds = hasattr(u.atoms, 'bonds') and len(u.atoms.bonds) > 0
#     broken_framework_bonds = []

#     # 2. BOND DETECTION AND FRAGMENT MOVEMENT
#     if has_bonds:
#         print("Bonds detected: analyzing boundary-crossing bonds...")
        
#         pos1 = u.atoms.bonds.atom1.positions
#         pos2 = u.atoms.bonds.atom2.positions
        
#         proj1 = np.dot(pos1, unit_norm)
#         proj2 = np.dot(pos2, unit_norm)
        
#         # Check which bonds cross the cutting plane
#         crosses_plane = ((proj1 < coordinate) & (proj2 >= coordinate)) | \
#                         ((proj1 >= coordinate) & (proj2 < coordinate))
        
#         # Identify Si/Al - O framework bonds crossing the plane
#         framework_types = {"Si", "Al"}
#         types1 = u.atoms.bonds.atom1.types.astype(str)
#         types2 = u.atoms.bonds.atom2.types.astype(str)

#         atom1_is_fw = np.isin(types1, list(framework_types))
#         atom2_is_fw = np.isin(types2, list(framework_types))
#         atom1_is_o = np.char.startswith(types1, "O")
#         atom2_is_o = np.char.startswith(types2, "O")

#         is_framework_bond = (atom1_is_fw & atom2_is_o) | (atom2_is_fw & atom1_is_o)
#         crosses_framework = crosses_plane & is_framework_bond

#         if crosses_framework.any():
#             # Store index pairs (i, j) of broken framework bonds
#             broken_framework_bonds = u.atoms.bonds[crosses_framework].to_indices()

#         # Remove all crossing bonds to cleanly separate fragments
#         valid_bonds = u.atoms.bonds[~crosses_plane]
#         u.delete_bonds(u.atoms.bonds)
#         u.add_bonds(valid_bonds.to_indices())

#         # Shift upper fragments
#         print("Moving solid fragments...")
#         for frag in u.atoms.fragments:
#             if np.dot(frag.centroid(), unit_norm) >= coordinate:
#                 frag.positions += shift_vector
#     else:
#         print("No bonds detected: moving atom by atom.")
#         mask = np.dot(u.atoms.positions, unit_norm) >= coordinate
#         u.atoms.positions[mask] += shift_vector

#     # Update simulation box dimensions
#     vec_list[axis] = target_vec + shift_vector
#     extended_box = vectors2lattice(tuple(vec_list))

#     # 3. CAPPING SEVERED FRAMEWORK BONDS
#     if len(broken_framework_bonds) > 0:
#         print(f"CAPPING: Adding silanol groups to {len(broken_framework_bonds)} broken bonds...")
#         # Calculate positions and types for capping atoms
#         new_pos, new_types = _cap_broken_framework_bonds(u, broken_framework_bonds)
        
#         # Append capping atoms directly to the MDAnalysis Universe
#         n_capping = len(new_types)
#         capping_u = mda.Universe.empty(n_capping, trajectory=True)
#         capping_u.add_TopologyAttr('id', np.arange(1, n_capping + 1))
#         capping_u.add_TopologyAttr('type', new_types)
#         capping_u.add_TopologyAttr('name', new_types)
#         capping_u.atoms.positions = new_pos

#         capping_masses = [MASSES_DICT.get(t, 1.0) for t in new_types]
#         capping_u.add_TopologyAttr('mass', capping_masses)
        
#         # Merge the original universe u with the new capping atoms
#         u = mda.Merge(u.atoms, capping_u.atoms)

#         u.atoms.ids = np.arange(1, len(u.atoms) + 1)

#     # Convert updated MDAnalysis structure back to AtomicSystem
#     final_system = AtomicSystem.from_mda(u)
#     final_system.set_box(extended_box)

    # 4. INSERTING LIQUID PHASE (OPTIONAL)
    if add_solution:
        print("Generating and inserting liquid solution...")
        dims = [np.linalg.norm(v) for v in vec_list]
        liquid_thickness = gap_size - (2 * tolerance)
        
        if liquid_thickness <= 0:
            raise ValueError(f"Gap size ({gap_size} Å) is too small for requested tolerance ({tolerance} Å).")
        
        other_axes = [i for i in range(3) if i != axis]
        liquid_box_dims = [dims[other_axes[0]], dims[other_axes[1]], liquid_thickness]
        
        liquid_data = build_solution(liquid_box_dims, density, solutes_dict, structures_dict)

        gap_center = coordinate + (gap_size / 2.0)
        
        axis_name = ['x', 'y', 'z'][axis]
        liquid_pos = liquid_data.atoms[axis_name].values
        liquid_center = (np.max(liquid_pos) + np.min(liquid_pos)) / 2.0
        
        liquid_data.atoms[axis_name] += (gap_center - liquid_center)

        final_system = merge(final_system, liquid_data, extended_box)

    return final_system


def split(solid_system: AtomicSystem, 
                 axis: int = 2,
                 coordinate: float = None,
                 gap_size: float = 20.0,
                 add_solution: bool = False, 
                 density: float = 1.0,
                 solutes_dict: dict[str, int] = None,
                 structures_dict: dict[str, AtomicSystem] = None) -> AtomicSystem:
    """
    Sépare un système solide en deux fragments le long d'un axe sans gérer les liaisons.
    """
    # 1. PRÉPARATION
    vec_list = list(lattice2vectors(solid_system.box))
    target_vec = vec_list[axis]
    unit_norm = target_vec / np.linalg.norm(target_vec)
    
    if coordinate is None:
        coordinate = np.linalg.norm(target_vec) / 2.0
    
    # On travaille sur les positions directement via le système
    # (en supposant que solid_system permet l'accès aux positions)
    universe = solid_system.to_mda()
    pos = universe.atoms.positions
    
    # 2. DÉPLACEMENT DES FRAGMENTS
    # On identifie les atomes situés après la coordonnée de coupe
    mask = np.dot(pos, unit_norm) >= coordinate
    shift_vector = unit_norm * gap_size
    pos[mask] += shift_vector
    
    # Mise à jour des positions dans le système
    universe.atoms.positions = pos
    
    # 3. MISE À JOUR DE LA BOÎTE
    vec_list[axis] = target_vec + shift_vector
    solid_system.set_box(vectors2lattice(tuple(vec_list)))
    
    # 4. INSERTION DE LIQUIDE (OPTIONNEL)
    if add_solution:
        # Logique simplifiée pour insérer le liquide au centre du gap
        liquid_data = build_solution([vec_list[0][0], vec_list[1][1], gap_size], density, solutes_dict, structures_dict)
        # (Ajoute ici ta logique de centrage spécifique à ton outil)
        solid_system = merge(solid_system, liquid_data)
        
    return solid_system

# def split(solid_system: AtomicSystem, 
#           axis: int = 2,
#           coordinate: float = None,
#           gap_size: float = 20.0,
#           tolerance: float = 2.0,
#           add_solution: bool = False, 
#           density: float = 1.0,
#           solutes_dict: dict[str, int] = None,
#           structures_dict: dict[str, AtomicSystem] = None,
#           topo_style: str = 'cshff') -> AtomicSystem:
#     """
#     Splits a solid system along a specified axis at a given coordinate,
#     caps severed framework bonds (Si/Al - O) with silanol groups, and optionally
#     inserts a liquid solution into the created gap.

#     Parameters
#     ----------
#     solid_system : AtomicSystem or str
#         The original solid system object or file path.
#     axis : int, optional
#         Cutting axis (0 for X, 1 for Y, 2 for Z). Default is 2.
#     coordinate : float, optional
#         Coordinate position along the axis where the cut is made. 
#         Defaults to the midpoint of the simulation box.
#     gap_size : float, optional
#         Width of the created gap in Angstroms (Å). Default is 20.0.
#     tolerance : float, optional
#         Safety distance buffer between liquid and solid surfaces in Å. Default is 2.0.
#     add_solution : bool, optional
#         If True, fills the generated gap with a liquid solution. Default is True.
#     density : float, optional
#         Target liquid density in g/cm³. Default is 1.0.
#     solutes_dict : dict, optional
#         Dictionary mapping solute names to their requested counts.
#     structures_dict : dict, optional
#         Dictionary containing solute AtomicSystem structures.
#     topo_style : str, optional
#         Topology style used to rebuild bonds ('cshff', etc.). Default is 'cshff'.

#     Returns
#     -------
#     AtomicSystem
#         The final split, capped, and optionally solvated atomic system.
#     """
#     if isinstance(solid_system, str):
#         solid_system = AtomicSystem.from_file(solid_system)

#     # 1. GEOMETRY AND BOX PREPARATION
#     vec_list = list(lattice2vectors(solid_system.box))
#     target_vec = vec_list[axis]
#     unit_norm = target_vec / np.linalg.norm(target_vec)
#     L_original = np.linalg.norm(target_vec)
    
#     if coordinate is None:
#         coordinate = L_original / 2.0
    
#     u = solid_system.to_mda()
#     shift_vector = unit_norm * gap_size
    
#     has_bonds = hasattr(u.atoms, 'bonds') and len(u.atoms.bonds) > 0
#     broken_framework_bonds = []

#     # 2. BOND DETECTION AND FRAGMENT MOVEMENT
#     if has_bonds:
#         print("Bonds detected: analyzing boundary-crossing bonds...")
        
#         pos1 = u.atoms.bonds.atom1.positions
#         pos2 = u.atoms.bonds.atom2.positions
        
#         proj1 = np.dot(pos1, unit_norm)
#         proj2 = np.dot(pos2, unit_norm)
        
#         # Check which bonds cross the cutting plane
#         crosses_plane = ((proj1 < coordinate) & (proj2 >= coordinate)) | \
#                         ((proj1 >= coordinate) & (proj2 < coordinate))
        
#         # Identify Si/Al - O framework bonds crossing the plane
#         framework_types = {"Si", "Al"}
#         types1 = u.atoms.bonds.atom1.types.astype(str)
#         types2 = u.atoms.bonds.atom2.types.astype(str)

#         atom1_is_fw = np.isin(types1, list(framework_types))
#         atom2_is_fw = np.isin(types2, list(framework_types))
#         atom1_is_o = np.char.startswith(types1, "O")
#         atom2_is_o = np.char.startswith(types2, "O")

#         is_framework_bond = (atom1_is_fw & atom2_is_o) | (atom2_is_fw & atom1_is_o)
#         crosses_framework = crosses_plane & is_framework_bond

#         if crosses_framework.any():
#             # Store index pairs (i, j) of broken framework bonds
#             broken_framework_bonds = u.atoms.bonds[crosses_framework].to_indices()

#         # Remove all crossing bonds to cleanly separate fragments
#         valid_bonds = u.atoms.bonds[~crosses_plane]
#         u.delete_bonds(u.atoms.bonds)
#         u.add_bonds(valid_bonds.to_indices())

#         # Shift upper fragments
#         print("Moving solid fragments...")
#         for frag in u.atoms.fragments:
#             if np.dot(frag.centroid(), unit_norm) >= coordinate:
#                 frag.positions += shift_vector
#     else:
#         print("No bonds detected: moving atom by atom.")
#         mask = np.dot(u.atoms.positions, unit_norm) >= coordinate
#         u.atoms.positions[mask] += shift_vector

#     # Update simulation box dimensions
#     vec_list[axis] = target_vec + shift_vector
#     extended_box = vectors2lattice(tuple(vec_list))

#     # 3. CAPPING SEVERED FRAMEWORK BONDS
#     if len(broken_framework_bonds) > 0:
#         print(f"CAPPING: Adding silanol groups to {len(broken_framework_bonds)} broken bonds...")
#         # Run capping function on the ALREADY MOVED MDAnalysis universe
#         new_pos, new_types = _cap_broken_framework_bonds(u, broken_framework_bonds)
        
#         # Convert generated capping atoms to an AtomicSystem
#         cap_system = _build_atomic_system_from_coords(new_pos, new_types)
        
#         # Merge moved solid structure with capping atoms
#         final_system = merge(AtomicSystem.from_mda(u), cap_system, extended_box)
#     else:
#         final_system = AtomicSystem.from_mda(u)

#     final_system.set_box(extended_box)

#     # 4. INSERTING LIQUID PHASE (OPTIONAL)
#     if add_solution:
#         print("Generating and inserting liquid solution...")
#         dims = [np.linalg.norm(v) for v in vec_list]
#         liquid_thickness = gap_size - (2 * tolerance)
        
#         if liquid_thickness <= 0:
#             raise ValueError(f"Gap size ({gap_size} Å) is too small for requested tolerance ({tolerance} Å).")
        
#         other_axes = [i for i in range(3) if i != axis]
#         liquid_box_dims = [dims[other_axes[0]], dims[other_axes[1]], liquid_thickness]
        
#         liquid_data = make_solution(liquid_box_dims, density, solutes_dict, structures_dict)

#         gap_center = coordinate + (gap_size / 2.0)
        
#         axis_name = ['x', 'y', 'z'][axis]
#         liquid_pos = liquid_data.atoms[axis_name].values
#         liquid_center = (np.max(liquid_pos) + np.min(liquid_pos)) / 2.0
        
#         liquid_data.atoms[axis_name] += (gap_center - liquid_center)

#         final_system = merge(final_system, liquid_data, extended_box)

#     # 5. GLOBAL TOPOLOGY REGENERATION
#     if topo_style is not None:
#         print(f"Regenerating global topology ({topo_style})...")
#         final_system.set_topo(style=topo_style)

#     return final_system

def merge(lmp_data_a: AtomicSystem, 
          lmp_data_b: AtomicSystem, 
          box:np.ndarray | Sequence[float]=None
          ) -> AtomicSystem:
    '''Merge two LAMMPSData objects and deal with topology issues.'''

    output_topology = {}

    # Set box
    if box is None:
        output_topology['lmp_box'] = lmp_data_a._lmp_box
    else:
        output_topology['lmp_box'] = lattice2lammps(box)
    
    # Merge atoms
    indices_a = lmp_data_a.atoms.index.to_numpy()
    indices_b = lmp_data_b.atoms.index.to_numpy() + lmp_data_a.num_atoms
    merged_indices = np.concatenate([indices_a, indices_b])
    merged_atoms_df = pd.concat([lmp_data_a.atoms, lmp_data_b.atoms], ignore_index=True)
    merged_atoms_df.index = merged_indices
    output_topology['atoms'] = merged_atoms_df

    if lmp_data_a.velocities is not None and lmp_data_b.velocities is not None:
        merged_velocities_df = np.concatenate([lmp_data_a.velocities, lmp_data_b.velocities])
        merged_velocities_df.index = merged_indices
    else:
        output_topology['velocities'] = None

    # Merge masses
    masses_dict_a = {}
    masses_dict_b = {}

    for t, m in zip(lmp_data_a.atom_types, lmp_data_a.masses):
        masses_dict_a[t] = m
    for t, m in zip(lmp_data_b.atom_types, lmp_data_b.masses):
        masses_dict_b[t] = m

    masses_dict_a.update(masses_dict_b)
    sorted_masses_dict_a = dict(sorted(masses_dict_a.items()))
    output_topology['masses'] = sorted_masses_dict_a

    # Merge charges
    charges_dict_a = {}
    charges_dict_b = {}

    for t, m in zip(lmp_data_a.atom_types, lmp_data_a.charges):
        charges_dict_a[t] = m
    for t, m in zip(lmp_data_b.atom_types, lmp_data_b.charges):
        charges_dict_b[t] = m

    charges_dict_a.update(masses_dict_b)
    sorted_charges_dict_a = dict(sorted(charges_dict_a.items()))
    output_topology['charges'] = sorted_charges_dict_a

    output_topology['atom_types'] = list(sorted_masses_dict_a.keys())

    # Get topology info
    connectivity_keys = ['bonds', 'angles', 'dihedrals', 'impropers']
    a_connectivity_dic = {
        'bonds': lmp_data_a.bonds,
        'angles': lmp_data_a.angles,
        'dihedrals': lmp_data_a.dihedrals,
        'impropers': lmp_data_a.impropers
    }
    b_connectivity_dic = {
        'bonds': lmp_data_b.bonds,
        'angles': lmp_data_b.angles,
        'dihedrals': lmp_data_b.dihedrals,
        'impropers': lmp_data_b.impropers
    }

    for key in connectivity_keys:
        if b_connectivity_dic[key] is not None:
            b_connectivity_dic[key].iloc[:,1:] = b_connectivity_dic[key].iloc[:,1:].astype(int) + lmp_data_a.num_atoms
        else:
            b_connectivity_dic[key] = pd.DataFrame()

    for key in connectivity_keys:
        if a_connectivity_dic[key] is None:
            a_connectivity_dic[key] = pd.DataFrame()

    # Set new topology
    for key in connectivity_keys:
        df_merged = pd.concat([a_connectivity_dic[key], b_connectivity_dic[key]], ignore_index=True)

        if key == 'bonds':
            if len(df_merged) != 0:
                output_topology['bonds'] = df_merged
                output_topology['bonds'].index = range(1, len(df_merged) + 1)
            else:
                output_topology['bonds'] = None
            
        if key == 'angles':
            if len(df_merged) != 0:
                output_topology['angles'] = df_merged
                output_topology['angles'].index = range(1, len(df_merged) + 1)
            else:
                output_topology['angles'] = None

        if key == 'dihedrals':
            if len(df_merged) != 0:
                output_topology['dihedrals'] = df_merged
                output_topology['dihedrals'].index = range(1, len(df_merged) + 1)
            else:
                output_topology['dihedrals'] = None

        if key == 'impropers':
            if len(df_merged) != 0:
                output_topology['impropers'] = df_merged
                output_topology['impropers'].index = range(1, len(df_merged) + 1)
            else:
                output_topology['impropers'] = None

    return AtomicSystem(output_topology)

def protonate(data: AtomicSystem, 
              atom_index: int, 
              bond_length: float=1.0) -> AtomicSystem:
    """
    Add a proton (hydrogen atom) to a specific atom in the system.

    The position of the new proton is calculated to be opposite to the 
    center of mass of the target atom's current neighbors to respect 
    local geometry (e.g., VSEPR logic).

    Parameters
    ----------
    data : AtomicSystem
        The atomic system object to modify.
    atom_index : int
        The 0-based integer position (iloc) of the target atom in the 
        `data.atoms` DataFrame.
    bond_length : float, default 1.0
        The distance between the target atom and the new proton in Ångströms.
    proton_type : str, default 'H'
        The type label for the newly created atom.

    Returns
    -------
    data : AtomicSystem
        The modified atomic system with the added proton and updated types/masses.

    Notes
    -----
    If the target atom has no neighbors, the proton is placed along the Z-axis.
    The function automatically updates `data.atom_types` and `data.masses` 
    if a new element is introduced.
    """
    # Extract target atom coordinates using iloc
    target_atom = data.atoms.iloc[atom_index]
    pos_target = target_atom[['x', 'y', 'z']].values.astype(float)
    
    # Determine orientation using MDAnalysis
    u = data.to_mda()
    # MDAnalysis uses 1-based indexing for selection strings
    neighbors = u.select_atoms(f"around 2.2 index {atom_index + 1}")
    
    if len(neighbors) > 0:
        # Calculate direction opposite to the neighbors' center of mass
        pos_neighbors = neighbors.center_of_mass()
        direction = pos_target - pos_neighbors
    else:
        # Default direction if the atom is isolated
        direction = np.array([0, 0, 1.0])
    
    # Normalize the direction vector
    norm = np.linalg.norm(direction)
    if norm < 1e-5: 
        direction = np.array([0, 0, 1.0])
    else:
        direction = direction / norm
    
    # Calculate new position
    new_pos = pos_target + (direction * bond_length)
    
    # USE THE NEW add_atom METHOD
    # This replaces all the manual dictionary/concat/mass logic
    p_mass = MASSES_DICT.get('H', 1.008)
    
    new_id = data.add_atom(
        atype='H',
        position= new_pos,
        charge=+1,   # Or any default charge you prefer
        mass=p_mass
    )

    return data
