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

import os

import numpy as np
import MDAnalysis as mda

from ._packmol import get_structure_path, add_packmol_structure, run_packmol
from .._utils import grouped_average, lattice2vectors

def _plane(vecp: np.ndarray, vecq: np.ndarray, vecr: np.ndarray) -> str:
    """Given three 3D-vectors, return the parameters a, b, c, d of the equation of the plane of coordinates (x, y, z) that goes through them, where:
    .. math:: ax + by + cz = d
    """
    
    vecu = vecq - vecp
    vecv = vecr - vecp
    vecw = np.cross(vecu, vecv)

    return f"{vecw[0]} {vecw[1]} {vecw[2]} {np.dot(vecw, vecp)}"

def _get_packmol_bounding_planes(box: np.ndarray) -> tuple:
    """Computes the 4 bounding planes for Packmol from box vectors."""
    vec0 = np.zeros(3)
    vec1, vec2, vec3 = lattice2vectors(box)
    vec1, vec2, vec3 = vec1 * 0.95, vec2 * 0.95, vec3 * 0.95

    return (
        _plane(vec0, vec2, vec3),           # Left
        _plane(vec1, vec1+vec2, vec1+vec3), # Right
        _plane(vec0, vec3, vec1),           # Bottom
        _plane(vec2, vec2+vec3, vec1+vec2)  # Top
    )

def distribute_species_in_layers(num_water: int, 
                                  nca_to_add: int, 
                                  si_posz: np.ndarray, 
                                  supercell_factor: int,
                                  box_z: float):# -> tuple[list, list, list, int]:
    """
    Calculates layer positions and distributes H2O and Ca per layer.
    """
    # Find layers and keep only the main silicate planes (max atoms)
    si_layers_pos, _, list_num = grouped_average(si_posz, 8 * supercell_factor)
    si_layers_pos = si_layers_pos[np.array(list_num) == max(list_num)]

    nplanes = len(si_layers_pos)
    interlayers_bounds = []

    for i in range(nplanes):
        z_start = si_layers_pos[i]
        z_end = si_layers_pos[(i + 1) % nplanes]
        
        # Management of periodicity at the edges of the box (PBC)
        dist = (z_end - z_start) % box_z
        
        # An intersheet space is typically > 5.0 Å (while an intrasheet space is ~3.5 Å)
        if dist > 5.0:
            interlayers_bounds.append((z_start, z_end, dist))

    nlayers = len(interlayers_bounds)
    
    # Helper to distribute a total number into n parts as evenly as possible
    def _distribute_evenly(total, n):
        if n == 0: return []
        base, extra = divmod(total, n)
        return [base + (1 if i < extra else 0) for i in range(n)]

    nw_layers = _distribute_evenly(num_water, nlayers)
    nca_layers = _distribute_evenly(nca_to_add, nlayers)
    
    return interlayers_bounds, nw_layers, nca_layers, nlayers

def fill_csh_interlayers(univ: mda.Universe,
                           box: np.ndarray,
                           interlayers_bounds: list[tuple[float,float,float]],
                           nw_layers: list,
                           nca_layers: list,
                           nca_to_add: int,
                           nlayers: int,
                           tmp: str,
                           progress_callback=None) -> str:
    """Runs Packmol iteratively to fill each CSH interlayer with water and Ca."""

    left, right, bottom, top = _get_packmol_bounding_planes(box)
    h2o_pdb = get_structure_path('h2o', tmp)
    ca_pdb = get_structure_path('ca', tmp) if nca_to_add != 0 else None

    current_pdb = os.path.join(tmp, 'tmp0.pdb')
    univ.write(current_pdb)
    idmin = np.argmin(univ.select_atoms("all").positions[:, 2]) + 1

    for i in range(nlayers):
        z_start, z_end, dist = interlayers_bounds[i]

        z_upper = z_start + dist if z_end < z_start else z_end

        layer_instructions = [
            f"over plane 0 0 1 {z_start + 1.5:.4f}",
            f"below plane 0 0 1 {z_upper - 1.5:.4f}",
            f"over plane {bottom}", f"below plane {top}",
            f"over plane {left}",  f"below plane {right}",
        ]

        structures = [
            add_packmol_structure(current_pdb, 1,
                f"inside box 0 0 0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}",
                f"atoms {idmin}", "fixed 0 0 0 0 0 0"
            ),
            add_packmol_structure(h2o_pdb, int(nw_layers[i]), *layer_instructions),
        ]
        if nca_to_add != 0:
            structures.append(add_packmol_structure(ca_pdb, int(nca_layers[i]), *layer_instructions))

        next_pdb = os.path.join(tmp, f'tmp{i+1}.pdb')
        run_packmol(structures, next_pdb)
        current_pdb = next_pdb

        msg = f"Adding {nw_layers[i]} H2O" + (f" and {nca_layers[i]} Ca2+" if nca_to_add != 0 else "")
        print(f"{msg} in layer {i+1}")
        if progress_callback:
            p = 20 + int((i / nlayers) * 70)
            progress_callback(p, f"{msg} in interlayer {i+1}/{nlayers}...")

    return current_pdb