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

import numpy as np
import random
import MDAnalysis as mda

from .._utils import grouped_average

def _get_bridging_silicates(si_sel: mda.AtomGroup, 
                            supercell_factor: int) -> np.ndarray:
    """Returns indices of bridging silicates (groups with the minimum atom count)."""
    si_indices = si_sel.indices
    si_posz = si_sel.positions[:, 2]

    _, list_indices, list_num = grouped_average(si_posz, 8 * supercell_factor)
    min_len = min(list_num)
    bridging_groups = [group for group in list_indices if len(group) == min_len]
    
    return si_indices[np.concatenate(bridging_groups).flatten()]

def _find_symmetric_bridging_pairs(si_sel: mda.AtomGroup, 
                                   supercell_factor: int
                                   ) -> list[tuple[int, int | None]]:
    """
    Identifies bridging silicates and associates them in symmetrical pairs 
    on either side of the pores/sheets.
    """

    bridging_atoms = _get_bridging_silicates(si_sel, supercell_factor)
    coords = bridging_atoms.positions

    pairs = []
    used = set()

    # Global Center for Symmetry
    center = np.mean(si_sel.positions, axis=0)

    for i, idx1 in enumerate(bridging_atoms.indices):
        if idx1 in used:
            continue
        
        pos1 = coords[i]
        # Inversion relative to the center (or XY mirror relative to Z)
        target_pos = 2 * center - pos1 

        # Find the atom closest to the theoretical symmetrical position
        best_match = None
        min_dist = float('inf')

        for j, idx2 in enumerate(bridging_atoms.indices):
            if idx2 in used or idx1 == idx2:
                continue
            dist = np.linalg.norm(coords[j] - target_pos)
            if dist < min_dist:
                min_dist = dist
                best_match = idx2

        # If the symmetrical partner is found within a reasonable tolerance (e.g.: < 3.0 Å)
        if best_match is not None and min_dist < 3.0:
            pairs.append((idx1, best_match))
            used.add(idx1)
            used.add(best_match)
        else:
            # Symmetry orphan atom
            pairs.append((idx1, None))
            used.add(idx1)

    return pairs

def calculate_csh_modifiers(
        nsi: int, nca: int, target_cs_ratio: float, min_mcl: float
        ) -> tuple[int, int, float]:
    """Calculate the vacancy fraction and check that it is higher than 1 /(1 + MCL) to keep a minimum of bridging silicates. Calculate the number of bridging SiO2 and Ca2+ to add to reach the target C/S."""
    nsi_to_remove = nsi - round(nca / target_cs_ratio)
    vacancy_fraction = nsi_to_remove / nsi
    min_vacancy_fraction = 1 / (1 + min_mcl)

    nca_to_add = 0
    if vacancy_fraction > min_vacancy_fraction:
        while vacancy_fraction > min_vacancy_fraction:
            nsi_to_remove -= 1
            vacancy_fraction = nsi_to_remove / nsi

        nca_to_add = max(0, np.floor(target_cs_ratio * (nsi - nsi_to_remove)) - nca)

    return nsi_to_remove, nca_to_add, vacancy_fraction

def remove_bridging_silicates(univ: mda.Universe, 
                               nsi_to_remove: int, 
                               supercell_factor: int, 
                               symmetry: bool = True) -> mda.Universe:
    """
    Identifies bridging silicates and removes them (symmetrically or randomly).
    """
    if nsi_to_remove <= 0:
        return univ

    si_sel = univ.select_atoms("type Si")

    if symmetry:
        pairs = _find_symmetric_bridging_pairs(si_sel, supercell_factor)
        random.shuffle(pairs)

        to_remove = []
        count = 0
        for idx1, idx2 in pairs:
            if count >= nsi_to_remove:
                break
            to_remove.append(idx1)
            count += 1

            if idx2 is not None and count < nsi_to_remove:
                to_remove.append(idx2)
                count += 1
    else:
        bridging_indices = _get_bridging_silicates(si_sel, supercell_factor)
        to_remove = np.random.choice(bridging_indices, nsi_to_remove, replace=False)

    # Universe update
    univ = univ.select_atoms(f"not index {' '.join(map(str, to_remove))}")
    univ = univ.select_atoms("not (name O and not around 2.1 type Si)")
    
    return univ

def substitute_si_by_al(univ: mda.Universe, 
                         nal: int, 
                         supercell_factor: int, 
                         symmetry: bool = True
                         ) -> tuple[mda.Universe, list[int]]:
    """
    Substitutes bridging Si with Al according to a ratio.
    
    Parameters
    ----------
    univ : mda.Universe
        The MDAnalysis Universe containing the structure.
    nal : int
        Number of Al atoms to substitute.
    supercell_factor : int
        Supercell factor along the Z direction.
    symmetry : bool, optional
        If True, substitutes Si by Al in symmetric pairs across interlayers.
        If False, substitutes randomly. Default is True.
    """
    if nal <= 0:
        return univ, []

    si_sel = univ.select_atoms("type Si")
    
    if symmetry:
        # Symmetric pairwise selection
        pairs = _find_symmetric_bridging_pairs(si_sel, supercell_factor)
        random.shuffle(pairs)

        to_sub = []
        count = 0
        for idx1, idx2 in pairs:
            if count >= nal:
                break
            to_sub.append(idx1)
            count += 1

            if idx2 is not None and count < nal:
                to_sub.append(idx2)
                count += 1
    else:
        bridging_indices = _get_bridging_silicates(si_sel, supercell_factor)
        to_sub = list(np.random.choice(bridging_indices, nal, replace=False))

    # Changing atom types
    for idx in to_sub:
        atom = univ.atoms[idx]
        atom.type = 'Al'
        atom.name = 'Al'

    substituted_ids = univ.atoms[to_sub].ids.tolist()
            
    return univ, substituted_ids

def neutralize_csh_charge(pdb_path:str) -> float:
    """Remove hydrogens to neutralize the C-S-H system"""

    univ = mda.Universe(pdb_path)

    si_sel = univ.select_atoms("name Si SI")
    ca_sel = univ.select_atoms("name Ca CA")
    o_sel = univ.select_atoms("name O")
    h_sel = univ.select_atoms("name H")
    total_charge = 2 * len(ca_sel) + len(h_sel) + 4 * len(si_sel) - 2 * len(o_sel)
    h_candidates = list(h_sel.indices[::2])
    if len(h_candidates) < total_charge:
        total_charge = len(h_candidates)
    to_remove = random.sample(h_candidates, total_charge)

    # Filter universe and overwrite PDB
    remaining_indices = np.setdiff1d(univ.atoms.indices, to_remove)
    univ.atoms[remaining_indices].write(pdb_path)
    
    return total_charge / 2
