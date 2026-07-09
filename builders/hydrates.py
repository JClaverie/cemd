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

import subprocess
import tempfile
import shutil
import os
import re
from typing import Sequence, Callable

import numpy as np

from .. import AtomicSystem
from .._config import STRUCTURES_DIR, PYCSH_DIR
from ._packmol import _add_packmol_structure, _get_structure_path, _run_packmol

from ._silicate_helpers import (
    remove_bridging_silicates, 
    substitute_si_by_al, 
    calculate_csh_modifiers,
    neutralize_csh_charge
)

from ._interlayer_helpers import (
    distribute_species_in_layers, 
    fill_csh_interlayers
)

def pycsh(cs_ratio: float, 
          ws_ratio: float, 
          supercell: Sequence=[3,5,2], 
          nsamples: int=1, 
          name=None, 
          seed=None, 
          progress_callback: Callable[[int, str], None]=None
          ) -> list[AtomicSystem]:
    """Create a C-S-H model or a list of C-S-H models as LAMMPSData using the pyCSH code (see https://doi.org/10.1016/j.cemconres.2024.107593).

    Parameters
    ----------
        cs_ratio
            The calcium/silicate ratio of the C-S-H model to be created
        ws_ratio
            The water/silicate ratio of the C-S-H model to be created
        supercell
            A list with the number of replicates of the unit cell in each direction (a, b, c)
        nsamples
            The number of models to create
        name
            Prefix for the LAMMPSData files
        seed
            Seed for the pyCSH generator
        progress_callback: 
            Progress callback for GUI

    """
    output_path = os.path.join(PYCSH_DIR,"output")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    if seed is None: 
        seed = int.from_bytes(os.urandom(4), 'big')

    os.chdir(PYCSH_DIR)
    with open(os.path.join(PYCSH_DIR, "parameters.py"), 'w') as f:
        f.write(f"seed = {seed}\n")
        f.write(f"shape = {supercell}\n")
        f.write(f"Ca_Si_ratio = {cs_ratio}\n")
        f.write(f"W_Si_ratio = {ws_ratio}\n")
        f.write(f"N_samples = {nsamples}\n")
        f.write(f"create = True\n")
        f.write(f"write_lammps = True\n")
        f.write(f"write_lammps_erica = False\n")
        f.write(f"write_vasp = False\n")
        f.write(f"write_siesta = False\n")
        if name is not None:
            f.write(f"prefix = {name}")
        else:
            name = f"cs{cs_ratio}_ws{ws_ratio}"
            f.write(f'prefix = "{name}"')

    try:
        pattern = re.compile(r"Structure\s+(\d+)\s+converged")
        
        process = subprocess.Popen(
            ['python', '-u', "main_brick.py"],
            cwd=PYCSH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            match = pattern.search(line)
            if match and progress_callback:
                current_id = int(match.group(1)) + 1
                
                msg_status = f"pyCSH is running, it could take some time..."
                progress_callback(current_id, msg_status)

        process.wait()
  
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing pyCSH:\n{e.stderr}"
        raise RuntimeError(error_msg)

    for i in ["distributions", "MCL", "water", "XOX_X"]:
        graph_file = f"{i}.pdf"
        src = os.path.join(PYCSH_DIR, graph_file)
        if os.path.exists(src):
            shutil.move(src, output_path)
        else:
            print(f"Warning: Graph {graph_file} was not generated.")

    log_src = os.path.join(PYCSH_DIR, "created_samples.log")
    if os.path.exists(log_src):
        shutil.move(log_src, output_path)

    atomic_systems_list = []
    
    for i in range(1, nsamples+1):
        reax_name = f"{name}_reax{i}.data"
        lmp_path = os.path.join(output_path, reax_name)
        
        if not os.path.exists(lmp_path):
            raise FileNotFoundError(
f"Critical error: pyCSH did not generate file {reax_name}."
                "Check the cs_ratio and ws_ratio parameters."
            )
        try:
            data = AtomicSystem.from_file(lmp_path)
            data.reset_types()
            atomic_systems_list.append(data)
        except Exception as e:
            raise RuntimeError(f"Error reading generated file {reax_name}: {str(e)}")

    # Secure cleaning
    for i in range(1, nsamples+1):
        for suffix in [f"_{i}.data", f"_reax{i}.data"]:
            cleanup_path = os.path.join(output_path, f"{name}{suffix}")
            if os.path.exists(cleanup_path):
                os.remove(cleanup_path)
    
    return atomic_systems_list[0] if len(atomic_systems_list) == 1 else atomic_systems_list


def make_csh(cs_ratio: float, 
             ws_ratio: float,
             supercell: Sequence[int]=None, 
             model: str='tob11a_hamid.cif', 
             min_mcl=3,
             symmetry: bool=True,
             progress_callback: Callable[[int, str], None]=None
             ) -> AtomicSystem:
    """Create a C-S-H supercell. To reach
    the wanted C/S ratio: first removes bridging SiO2 groups until the minimum
    meaning chain length (MCL) is reached, then add calcium ions. A fully
    dimeric silicate structure was not found experimentally and the minimum
    MCL is close to 3 (see https://doi.org/10.1107/S2052520614021982).
    Subsequently, the packmol code is called to insert water molecules in the
    interlayer ro reach the given H2O/Si ratio. The relation between
    C/S and H/S found in experiments
    (see https://doi.org/10.1107S2052520614021982) is:
    :math:`H/S = \\frac{19}{17} C/S - \\frac{7}{17}`.

    .. todo:: prevent crossing with Merlino unit cell
    .. todo:: improve detection of silicate layers

    Parameters
    ----------
        cs_ratio
            The calcium/silicate ratio of the C-S-H model to be created
        ws_ratio
            The water/silicate ratio of the C-S-H model to be created
        supercell
            A list with the number of replicates of the unit cell in each direction (a, b, c)
        model
            Name of the tobermorite model to use.

            Currently supported:
            -"tob11a_hamid.cif"
            -"tob11a_merlino.cif"
        min_mcl
            Minimum mean chain length to keep in the calculation, whatever the C/S ratio.
        symmetry
            Remove brindging silicates symmetrically (True) of randomly (False)
        progress_callback: 
            Progress callback for GUI

    """

    system = AtomicSystem.from_file(os.path.join(STRUCTURES_DIR, model))
    if model == "tob11a_hamid.cif":
        if supercell is None:
            supercell = [3,5,1]
            system.replicate(supercell)
            system.unskew()
    elif model == "tob11a_merlino.cif":
        system.orthogonalize()
        if supercell is None:
            supercell = [4,1,1]
            system.replicate([4,1,1])
    
    univ = system.to_mda()
    box = univ.dimensions

    # Calculate modifications
    nsi = len(univ.select_atoms("type Si"))
    nca = len(univ.select_atoms("type Ca"))
    nsi_to_remove, nca_to_add, vacancy_fraction = calculate_csh_modifiers(nsi, nca, cs_ratio, min_mcl)

    if progress_callback: progress_callback(10, "Randomly removing brindging silicates...")

    univ = remove_bridging_silicates(univ, nsi_to_remove, supercell[2], symmetry)

    # Read the new pdb file and calculate the number of water molecules to add
    # univ = mda.Universe("temp.pdb")
    si_sel = univ.select_atoms("type Si")
    nsi = len(si_sel)

    # add more water to balance the removal of H+ to charge balance the addition of Ca2+ and get the targe H/S ratio
    num_water  = round(nsi * ws_ratio) + nca_to_add

    if num_water == 0:
        final_system = AtomicSystem.from_mda(univ)
        final_system.set_box(box)
        return final_system

    si_layers_pos, nw_layers, nca_layers, nlayers = distribute_species_in_layers(
        num_water, nca_to_add, si_sel.positions[:, 2], supercell[2], box[2]
    )

    with tempfile.TemporaryDirectory(dir='.') as tmp:
        final_pdb = fill_csh_interlayers(
            univ, box, si_layers_pos, nw_layers, nca_layers,
            nca_to_add, nlayers, tmp, progress_callback
        )

        if nca_to_add != 0:
            neutralize_csh_charge(final_pdb)

        final_system = AtomicSystem.from_file(final_pdb)

    # 4. Finalisation
    final_system.set_box(box)
    final_system.set_topo(style='cshff')
    # Final system.wrap()

    nsi = final_system.get_count("Si")
    nca = final_system.get_count("Ca") + final_system.get_count("Cw")
    num_water = (final_system.get_count("Hw") + final_system.get_count("Hh") + final_system.get_count("H")) / 2

    errcs = (nca / nsi - cs_ratio) / cs_ratio
    errhs = (num_water / nsi - ws_ratio) / ws_ratio if ws_ratio != 0 else 0

    print("C-S-H model created")
    print(f"Real C/S ratio: {nca/nsi:.2f} (difference: {errcs*100:.1f}%)")
    print(f"Real H/S ratio: {num_water/nsi:.2f} (difference: {errhs*100:.1f}%)")
    print(f"Mean Chain Length: {(1-vacancy_fraction)/vacancy_fraction:.2f}")

    if progress_callback:
        progress_callback(100, "Structure complete")

    return final_system


# def make_cash(atomic_system: AtomicSystem, 
#               as_ratio: float) -> tuple[AtomicSystem, float]:
#     """Find the bridging silicates in a C-S-H structure and randomly transform
#     some of them into aluminates.

#     Parameters
#     ----------
#         atomic_system
#             Input AtomicSystem
#         as_ratio
#             The aluminium/silica ratio of the C-A-S-H model to be created

#     """

#     # Get information with MDAnalysis
#     univ = atomic_system.to_mda()
#     si_sel = univ.select_atoms("type Si")
#     if len(si_sel) == 0:
#         raise ValueError("No silicon atoms found in the system.")

#     # Get averaged coordinates of silicates layers, including averaged coordinates of bridging silicates
#     _, list_indices, list_num = grouped_average(si_sel.positions[:,2], univ.dimensions[2])

#     min_atoms_in_group = min(list_num)
#     bridging_groups = [group for group in list_indices if len(group) == min_atoms_in_group]
#     if not bridging_groups:
#         raise ValueError("Impossible d'identifier les silicates pontants.")
    
#     flat_bridging_idx = np.concatenate(bridging_groups).flatten()
#     bridging_ids = si_sel.ids[flat_bridging_idx].tolist()

#     nsi = len(si_sel)
#     nal_target = round(as_ratio *nsi)
#     if nal_target > len(bridging_ids):
#         raise ValueError("Their is not enough bridging Si to reach the given Al/Si ratio.\nPlease provide a smaller value.")

#     sampled_id = random.sample(bridging_ids, nal_target)
#     atomic_system.set_type2atoms(sampled_id, "Al")
#     atomic_system.keep_connection_types(
#         bond_types=[('Hsi', 'Osih'), ('Ha', 'Oah'), ('Hh', 'Oh'), ('Hw', 'Ow')], angle_types=[('Hw', 'Ow', 'Hw')])

#     err = (nal_target/nsi -as_ratio) /as_ratio

#     print("\n############ C-A-S-H model created with success! ############")
#     print(f"Real Al/Si ratio: {nal_target /nsi:.2f} (difference: {err *100:.1f}%)")
    
#     return atomic_system, nal_target/nsi


def csh_to_cash(atomic_system: AtomicSystem, 
              as_ratio: float,
              supercell_factor: int=1) -> tuple[AtomicSystem, float]:
    """Find the bridging silicates in a C-S-H structure and randomly transform
    some of them into aluminates in a symmetric manner.
    """
    universe = atomic_system.to_mda()
    si_sel = universe.select_atoms("type Si")
    if len(si_sel) == 0:
        raise ValueError("No silicon atoms found in the system.")

    nsi = len(si_sel)
    nal_target = round(as_ratio * nsi)

    # 1. Clean use of the dedicated helper function with symmetry!
    # (We assume sc_z=1 or calculated from the dimensions of the box)
    universe, substituted_ids = substitute_si_by_al(universe, 
                                                nal_target, 
                                                supercell_factor)

    # 2. Re-application of types and topological connections on AtomicSystem
    atomic_system.set_type2atoms(substituted_ids, "Al")
    atomic_system.keep_connection_types(
        bond_types=[('Hsi', 'Osih'), ('Ha', 'Oah'), ('Hh', 'Oh'), ('Hw', 'Ow')], 
        angle_types=[('Hw', 'Ow', 'Hw')]
    )

    err = (nal_target / nsi - as_ratio) / as_ratio if as_ratio != 0 else 0

    print("\n############ C-A-S-H model created with success! ############")
    print(f"Real Al/Si ratio: {nal_target / nsi:.2f} (difference: {err * 100:.1f}%)")
    
    return atomic_system, nal_target / nsi

def make_af(ws_ratio: float, supercell: Sequence[int]=None) -> AtomicSystem:
    """Call packmol to insert water molecules in AFt or AFm model according to
    the H2O/S ratio. Only support orthorhombic boxes.

    Parameters
    ----------
        ws_ratio
            Water/sulfate ratio
        supercell
            A list with the number of replicates of the unit cell in each direction (a, b, c)

    """

    input_atomic_system = AtomicSystem.from_file(
        os.path.join(STRUCTURES_DIR, "aft_moore.cif")
    )
    if supercell is not None:
        input_atomic_system.replicate(supercell)
    input_atomic_system.orthogonalize()

    # Get information with MDAnalysis
    univ = input_atomic_system.to_mda()
    box = univ.dimensions
    s_sel = univ.select_atoms("type S")
    num_s = len(s_sel)
    num_water = round(num_s * ws_ratio)

    # Get the index of the atom with the lowest coordinate as a reference for packmol
    sel = univ.select_atoms("all")
    idmin = np.argmin(sel.positions[:, 2]) + 1

    with tempfile.TemporaryDirectory(dir='.') as tmp:
        tempfile_ipdb = os.path.join(tmp, 'itmp.pdb')
        sel.write(tempfile_ipdb)

        # Utilisation des helpers de _packmol
        h2o_pdb = _get_structure_path('h2o', tmp)

        structures = [
            _add_packmol_structure(
                tempfile_ipdb, 
                1,
                f"inside box 0 0 0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}",
                f"atoms {idmin}",
                "fixed 0 0 0 0 0 0",
                "end atoms"
            ),
            _add_packmol_structure(
                h2o_pdb, 
                num_water,
                f"inside box 0 0 0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}"
            )
        ]

        output_atomic_system = _run_packmol(structures)

    # Set the box parameters and write topologic features
    output_atomic_system.set_box(box)
    output_atomic_system.set_topo()

    return output_atomic_system