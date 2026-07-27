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

import os
import re
import warnings
import tempfile
from typing import Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from pymatgen.core import Composition, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from .._paths import STRUCTURES_DIR
from .._constants import MASSES_DICT, AVOGADRO
from .._utils import (
    lattice2lammps, 
    lattice2vectors, 
    vectors2lattice,
    require_program
)
from ._packmol import (
    add_packmol_structure, 
    get_structure_path,
    run_packmol
)

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    import MDAnalysis as mda
    from ..core.atomic_system import AtomicSystem

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

def _validate_for_surface(system: AtomicSystem) -> None:
    """Validate that a pymatgen Structure can be used to generate surfaces."""

    try:
        if hasattr(system, '_pmg_struct'):
            struct = system._pmg_struct
        else:
            struct = system.to_pmg()
    except (TypeError, ValueError) as e:
        print(f"Cannot generate surfaces: {e}")
        return None
    
    if not isinstance(struct, Structure):
        raise TypeError(
            "Surface generation requires a periodic 3D structure. "
            "Isolated molecules are not supported."
        )
    
    try:
        analyzer = SpacegroupAnalyzer(struct, symprec=0.1)
        spacegroup = analyzer.get_space_group_symbol()
    except Exception:
        raise ValueError(
            "Could not determine the space group of the structure. "
            "Make sure the structure is a well-defined crystal."
        )
    
    # 3. Vérifier que la maille n'est pas trop distordue
    a, b, c = struct.lattice.abc
    if min(a, b, c) < 1.0:
        raise ValueError(
            f"Lattice parameter too small ({min(a,b,c):.2f} Å). "
            "The structure may not be a valid crystal."
        )

    return struct

def rebuild_silicates(u, si_o_cutoff=1.9, oh_dist=0.96, si_o_dist=1.6):
    """Rebuild the coordination of Si atoms with coordination < 4 by adding silanol (OH) groups.

    Uses MDAnalysis selection and distance calculation tools.

    Parameters
    ----------
    u : MDAnalysis.Universe
        The universe containing the system.
    si_o_cutoff : float, optional
        Cutoff radius to search for neighboring oxygens.
    oh_dist : float, optional
        Bond length of the added O-H group.
    si_o_dist : float, optional
        Bond length of the added Si-O group.

    Returns
    -------
    list[dict]
        A list of dictionaries containing the type and position of the new atoms.
    """
    new_atoms_data = []
    
    # 1. Select all silicon
    silicons = u.select_atoms("type Si")
    
    for si in silicons:
        # 2. Count neighboring oxygens in the cutoff radius (using PBC)
        # 'around' is MDAnalysis's native neighborhood tool
        neighbors = u.select_atoms(f"type O and around {si_o_cutoff} index {si.index}")
        
        n_bonds = len(neighbors)
        
        if n_bonds < 4:
            missing = 4 - n_bonds
            
            for i in range(missing):
                # Calculation of a simple direction (towards the outside of the tetrahedron)
                # Note: In a real case, we would calculate the vector opposite the barycenter
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                
                # Position of the new Oxygen
                o_pos = si.position + (direction * si_o_dist)
                # Position of the new Hydrogen
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
      -add one H on the O side (-> existing O becomes a silanol)
      -add one O+H on the Si/Al side (-> new silanol replacing the lost O)
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

        # ---Cap the surviving bridging O with an H ---
        # place the H roughly along the (now vacant) Si->O direction,
        # extended past O
        direction = o_atom.position - si_atom.position
        direction /= np.linalg.norm(direction)
        h_on_o = o_atom.position + direction * oh_bond_length
        new_pos.append(h_on_o)
        new_types.append("H")  # bonded to the existing O -> becomes Osih

        # ---Cap the under-coordinated Si with a new O-H ---
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


def build_surfaces(data: AtomicSystem, 
                 miller_indices: Sequence[int], 
                 min_slab_size: float=25.0, 
                 min_vacuum_size: float=15.0
                 ) -> tuple[list[AtomicSystem],
                            list[float],
                            list[float],
                            int]:
    """Generate crystalline surfaces from a given structure.

    Parameters
    ----------
    data : AtomicSystem
        The initial structure.
    miller_indices : Sequence[int]
        Miller indices for the surface cut.
    min_slab_size : float, optional
        Minimum thickness of the slab.
    min_vacuum_size : float, optional
        Minimum size of the added vacuum.

    Returns
    -------
    slabs_list_as : list[AtomicSystem]
        List of generated surfaces as AtomicSystem objects.
    shift_list : list[float]
        Shifts used for surface generation.
    dipole_list : list[float]
        Calculated dipoles for each surface.
    actual_broken : int
        Number of bonds broken during generation.
    """

    from ..core.atomic_system import AtomicSystem

    struct = _validate_for_surface(data)
        
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
        slabs_list_as.append(AtomicSystem.from_pmg(slab))
    
    return slabs_list_as, shift_list, dipole_list, actual_broken

def explore_surfaces(data: AtomicSystem) -> AtomicSystem | None:
    """Interactive surface explorer."""

    _ = _validate_for_surface(data)
    
    # Miller indices input
    miller_str = questionary.text(
        "Enter Miller indices (e.g. 1 0 0):",
        default="1 0 0"
    ).ask()
    if not miller_str: return None
    miller = [int(x) for x in miller_str.split()]

    min_slab = questionary.text("Min slab size (Å):", default="25.0").ask()
    min_vac  = questionary.text("Min vacuum size (Å):", default="15.0").ask()

    print(f"Generating ({miller[0]}{miller[1]}{miller[2]}) surfaces...")
    slabs, shifts, dipoles, n_broken = build_surfaces(
        data, miller,
        min_slab_size=float(min_slab),
        min_vacuum_size=float(min_vac)
    )

    if not slabs:
        print("No surfaces generated.")
        return None

    # Format display
    HEADER_FORMAT = "{cursor} {idx:>4} | {shift:>10} | {dipole:>12} | {natoms:>8}"
    ROW_FORMAT = "{cursor} {idx:>4} | {shift:>10.4f} | {dipole:>12.4f} | {natoms:>8}"
    index = 0

    def render():
        header = HEADER_FORMAT.format(
            cursor=" ", idx="#", shift="Shift", 
            dipole="Dipole (D)", natoms="N atoms"
        )
        lines = [
            f"Generated {len(slabs)} surfaces — {n_broken} broken bonds",
            "↑↓ Move   [Enter] Select   [q] Quit",
            "",
            header, "-" * len(header)
        ]
        for i, (slab, shift, dipole) in enumerate(zip(slabs, shifts, dipoles)):
            cursor = "➜" if i == index else " "
            lines.append(ROW_FORMAT.format(
                cursor=cursor,
                idx=i + 1,
                shift=shift,
                dipole=dipole,
                natoms=slab.num_atoms
            ))
        return "\n".join(lines)

    # Même pattern que explore_pubchem
    selected_idx = [0]
    kb = KeyBindings()

    @kb.add("up")
    def _(e):
        nonlocal index
        if index > 0: index -= 1
        e.app.invalidate()

    @kb.add("down")
    def _(e):
        nonlocal index
        if index < len(slabs) - 1: index += 1
        e.app.invalidate()

    @kb.add("enter")
    def _(e):
        selected_idx[0] = index
        e.app.exit()

    @kb.add("q")
    @kb.add("escape")
    def _(e): e.app.exit()

    window = Window(
        content=FormattedTextControl(render),
        always_hide_cursor=True,
    )
    Application(layout=Layout(HSplit([window])), key_bindings=kb).run()

    return slabs[selected_idx[0]]

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

    require_program('packmol')

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
    mass_ions_g = (mass_solutes_total_uma / AVOGADRO)
    
    # Deduce the remaining mass of water
    mass_water_g = mass_total_target_g - mass_ions_g
    
    if mass_water_g <= 0:
        raise ValueError("The density or volume is too low to hold all these ions!")

    # Number of water molecules
    water_molarmass = 2 * MASSES_DICT['H'] + MASSES_DICT['O']
    num_water = round((mass_water_g * AVOGADRO) / water_molarmass)

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

    if solutes_dict:
        for key, num in solutes_dict.items():
            print(f"{num} {key}")

    if not isinstance(box, list):
        box = box.tolist()

    data.set_box(box + [90, 90, 90])

    return data

def build_glass(box: Sequence[float] | np.ndarray, 
               density: float, 
               stoichiometry_dic: dict[str, float]) -> AtomicSystem:
    """Create an atomic system of glass from a composition of oxides or molecules.

    Parameters
    ----------
    box : Sequence[float] or np.ndarray
        Box dimensions.
    density : float
        Target density in g/cm³.
    stoichiometry_dic : dict[str, float]
        Dictionary mapping formula strings to their coefficients.

    Returns
    -------
    AtomicSystem
        The generated glass structure.
    """

    require_program('packmol')
    
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

                    atomic_composition[element] = atomic_composition.get(element, 0) + (amount * coeff)
                
                total_pattern_mass += m_part * coeff
            except Exception:
                raise ValueError(f"Component '{formula}' not found (no PDB or valid formula)")

    # Calculation of the number of units for the target density
    # n = (rho *V *Na) /M
    num_units = int(np.round((density * vol * 1e-24 * AVOGADRO) / total_pattern_mass))
    if num_units == 0: num_units = 1

    rho_real = (num_units * total_pattern_mass) / (AVOGADRO * vol * 1e-24)
    err = (rho_real - density) / density

    # Calling Packmol via temporary directory
    with tempfile.TemporaryDirectory(dir='.') as tmp:

        # Packmol config
        structures = []

        for comp, coeff_in_motif in atomic_composition.items():
            count = int(np.round(coeff_in_motif * num_units))
            if count <= 0: continue
            
            path2structure = get_structure_path(comp, tmp)
            
            structures.append(add_packmol_structure(
                path2structure, count,
                "center",
                f"inside box 1 1 1 {boxa-1:.4f} {boxb-1:.4f} {boxc-1:.4f}\n"
            ))

        data = run_packmol(structures)
        
    # Finalization
    data.set_box(box + [90, 90, 90])

    return data

def add_droplet(solid_system: AtomicSystem, 
                radius: float, 
                density: float = 1.0,
                distance: float = 2.0,
                solutes_dict: dict[str, int] = None, 
                structures_dict: dict[str, AtomicSystem] = None,
                vacuum: float = 10.0) -> AtomicSystem:
    """Add a hemispherical liquid droplet onto a solid surface.

    Parameters
    ----------
    solid_system : AtomicSystem
        The substrate system.
    radius : float
        Radius of the hemispherical droplet.
    density : float, optional
        Target liquid density.
    distance : float, optional
        Distance between the droplet and the surface.
    solutes_dict : dict, optional
        Dictionary of solutes in the droplet.
    structures_dict : dict, optional
        Dictionary of custom structures for solutes.
    vacuum : float, optional
        Vacuum space added to the system.

    Returns
    -------
    AtomicSystem
        The system with the added droplet.
    """

    require_program('packmol')

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
    mass_solutes_g = (mass_solutes_total_uma / AVOGADRO)
    mass_water_g = mass_total_target_g - mass_solutes_g
    
    if mass_water_g <= 0:
        raise ValueError("Density or volume too low for these ions/solutes!")

    water_molarmass = 2 * MASSES_DICT['H'] + MASSES_DICT['O']
    num_water = round((mass_water_g * AVOGADRO) / water_molarmass)

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
    
    return data

def add_structure(solid_system: AtomicSystem, 
                  structure_to_add: AtomicSystem, 
                  distance: float = 2.0, 
                  axis: str = 'z',
                  vacuum: float = 10.0) -> AtomicSystem:
    """Add a structure to a solid surface at a given distance.

    Aligns Centers of Mass (COM) on the transverse axes.

    Parameters
    ----------
    solid_system : AtomicSystem
        The base solid surface.
    structure_to_add : AtomicSystem
        The structure to add (e.g., droplet, liquid).
    distance : float, optional
        Distance between the surface and the new structure.
    axis : str, optional
        Axis along which the addition is performed ('x', 'y', 'z').
    vacuum : float, optional
        Empty space added after the new structure.

    Returns
    -------
    AtomicSystem
        The final combined system.
    """

    from ..core.atomic_system import AtomicSystem

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
    
    return data

def add_liquid(solid_system: AtomicSystem, 
               thickness: float, 
               density: float = 1.0,
               distance: float = 2.0,
               solutes_dict: dict[str, int] = None,
               structures_dict: dict[str, AtomicSystem] = None,
               vacuum: float = 0.0, 
               axis: str = 'z') -> AtomicSystem:
    """Create a solid/liquid interface by generating a liquid block.

    Parameters
    ----------
    solid_system : AtomicSystem
        The solid system.
    thickness : float
        Thickness of the liquid layer.
    density : float, optional
        Liquid density.
    distance : float, optional
        Gap between liquid and solid.
    solutes_dict : dict, optional
        Solutes in the liquid.
    structures_dict : dict, optional
        Custom solute structures.
    vacuum : float, optional
        Added vacuum space.
    axis : str, optional
        Interface axis.

    Returns
    -------
    AtomicSystem
        The solid/liquid system.
    """

    from ..core.atomic_system import AtomicSystem

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


def split(solid_system: AtomicSystem, 
                 axis: int = 2,
                 coordinate: float = None,
                 gap_size: float = 20.0,
                 add_solution: bool = False, 
                 density: float = 1.0,
                 solutes_dict: dict[str, int] = None,
                 structures_dict: dict[str, AtomicSystem] = None) -> AtomicSystem:
    """Split an AtomicSystem into two parts along a specified axis.

    If no position is provided, the system is split at the center of the 
    simulation box along the chosen axis.

    Parameters
    ----------
    data : AtomicSystem
        The system to be split.
    axis : str, optional
        The axis along which to perform the split ('x', 'y', or 'z'). 
        Defaults to 'z'.
    position : float, optional
        The coordinate value along the axis at which to perform the split. 
        If None, uses the geometric center of the box.

    Returns
    -------
    part_a : AtomicSystem
        The first portion of the system (atoms with coordinate < position).
    part_b : AtomicSystem
        The second portion of the system (atoms with coordinate >= position).
    """
    # PREPARATION
    vec_list = list(lattice2vectors(solid_system.box))
    target_vec = vec_list[axis]
    unit_norm = target_vec / np.linalg.norm(target_vec)
    
    if coordinate is None:
        coordinate = np.linalg.norm(target_vec) / 2.0
    
    # work on positions directly via the system
    # (assuming solid_system allows access to positions)
    universe = solid_system.to_mda()
    pos = universe.atoms.positions
    
    # MOVING FRAGMENTS
    # identify the atoms located after the cutting coordinate
    mask = np.dot(pos, unit_norm) >= coordinate
    shift_vector = unit_norm * gap_size
    pos[mask] += shift_vector
    
    # Updating positions in the system
    universe.atoms.positions = pos
    
    # BOX UPDATE
    vec_list[axis] = target_vec + shift_vector
    solid_system.set_box(vectors2lattice(tuple(vec_list)))
    
    if add_solution:
        # Simplified logic for inserting the liquid into the center of the gap
        liquid_data = build_solution([vec_list[0][0], vec_list[1][1], gap_size], density, solutes_dict, structures_dict)
        # (Add your centering logic specific to your tool here)
        solid_system = merge(solid_system, liquid_data)
        
    return solid_system


def merge(lmp_data_a: AtomicSystem, 
          lmp_data_b: AtomicSystem, 
          box:np.ndarray | Sequence[float]=None
          ) -> AtomicSystem:
    """Merge two AtomicSystem objects and handle topology updates.

    Parameters
    ----------
    lmp_data_a : AtomicSystem
        First system.
    lmp_data_b : AtomicSystem
        Second system.
    box : np.ndarray or Sequence[float], optional
        New box parameters for the merged system.

    Returns
    -------
    AtomicSystem
        The merged AtomicSystem.
    """

    from ..core.atomic_system import AtomicSystem

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

