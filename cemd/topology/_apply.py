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
# from __future__ import annotations

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

from .rules import DihedralRule, TopologyRule

if TYPE_CHECKING:
    import MDAnalysis as mda


# Updating the MDAnalysis topology
def _update_topo(univ, attr, new_entries):
    if not new_entries:
        return
    existing = list(getattr(univ, attr).indices) if hasattr(univ, attr) else []
    combined = list({tuple(e) for e in existing} | {tuple(e) for e in new_entries})
    if hasattr(univ, attr):
        univ.del_TopologyAttr(attr)
    univ.add_TopologyAttr(attr, combined)


def apply_single_rule_to_universe(
    universe: mda.Universe, rule: TopologyRule
) -> mda.Universe:
    """
    Applies a single geometry-based connectivity rule in-place on a MDAnalysis Universe.

    Parameters
    ----------
    universe : mda.Universe
        The MDAnalysis universe to modify.
    rule : TopologyRule
        The topology rule to apply.

    Returns
    -------
    mda.Universe
        The modified universe.
    """
    center_atoms = universe.select_atoms(rule.center)

    new_bonds = []
    new_angles = []
    new_impropers = []

    for c in center_atoms:
        matched_neighbors = []
        neighbor_type_tasks = []
        is_valid = True

        for n_criterion in rule.neighbors:
            found = universe.select_atoms(
                f"({n_criterion.selection}) and "
                f"(around {n_criterion.cutoff} index {c.index}) and "
                f"not index {c.index}"
            )

            if n_criterion.exact_match:
                if len(found) != n_criterion.count:
                    is_valid = False
                    break
            else:
                if len(found) < n_criterion.count:
                    is_valid = False
                    break

            matched_neighbors.extend(list(found))

            if n_criterion.new_type is not None:
                neighbor_type_tasks.append((found, n_criterion.new_type))

        if not is_valid:
            continue

        # Apply new type to center
        if rule.new_type is not None:
            c.type = rule.new_type

        # Apply new types to neighbors
        for atoms, new_type in neighbor_type_tasks:
            atoms.types = new_type

        # Create bonds
        if rule.bonds:
            for n in matched_neighbors:
                new_bonds.append(tuple(sorted((c.index, n.index))))

        # Create angles
        if rule.angles and len(matched_neighbors) >= 2:
            for n1, n2 in combinations(matched_neighbors, 2):
                new_angles.append((n1.index, c.index, n2.index))

        # Create impropers
        if rule.impropers and len(matched_neighbors) >= 3:
            for n1, n2, n3 in combinations(matched_neighbors, 3):
                new_impropers.append((c.index, n1.index, n2.index, n3.index))

    _update_topo(universe, "bonds", new_bonds)
    _update_topo(universe, "angles", new_angles)
    _update_topo(universe, "impropers", new_impropers)

    return universe


def apply_single_dihedral_rule_to_universe(
    universe: mda.Universe,
    dihedral_rules: list[DihedralRule],
    default_cutoff: float = 2.0,
) -> Sequence[int]:
    """Generate proper dihedrals using temporary bonds when needed.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis universe containing the system.
    dihedral_rules : List[DihedralRule]
        List of DihedralRule objects defining dihedral patterns.
    default_cutoff : float
        Default cutoff if not specified in rule.

    Returns
    -------
    Sequence[int]
        Sequence of dihedral indices (i, j, k, l).
    """

    from MDAnalysis.lib.distances import capped_distance

    dihedrals = []
    seen_dihedrals = set()

    for rule in dihedral_rules:
        sel_i = universe.select_atoms(rule.i)
        sel_j = universe.select_atoms(rule.j)
        sel_k = universe.select_atoms(rule.k)
        sel_l = universe.select_atoms(rule.l_)

        cutoffs = rule.cutoffs
        if len(cutoffs) < 3:
            cutoffs = list(cutoffs) + [default_cutoff] * (3 - len(cutoffs))

        if len(sel_i) == 0 or len(sel_j) == 0 or len(sel_k) == 0 or len(sel_l) == 0:
            continue

        pairs_ij, _ = capped_distance(
            sel_i.positions,
            sel_j.positions,
            max_cutoff=cutoffs[0],
            box=universe.dimensions,
        )

        pairs_jk, _ = capped_distance(
            sel_j.positions,
            sel_k.positions,
            max_cutoff=cutoffs[1],
            box=universe.dimensions,
        )

        pairs_kl, _ = capped_distance(
            sel_k.positions,
            sel_l.positions,
            max_cutoff=cutoffs[2],
            box=universe.dimensions,
        )

        idx_ij = {(sel_i.indices[i], sel_j.indices[j]) for i, j in pairs_ij}
        idx_jk = {(sel_j.indices[j], sel_k.indices[k]) for j, k in pairs_jk}
        idx_kl = {(sel_k.indices[k], sel_l.indices[l_]) for k, l_ in pairs_kl}

        for i, j in idx_ij:
            for j2, k in idx_jk:
                if j != j2:
                    continue
                for k2, l_ in idx_kl:
                    if k != k2:
                        continue
                    if i == l_:
                        continue

                    dih = (i, j, k, l_)

                    dih_rev = (l_, k, j, i)
                    if dih not in seen_dihedrals and dih_rev not in seen_dihedrals:
                        seen_dihedrals.add(dih)
                        dihedrals.append(dih)

        _update_topo(universe, "dihedrals", dihedrals)

    return universe


def _generate_dihedrals_from_bonds(universe: mda.Universe) -> list[tuple]:
    """Generate all proper dihedrals from existing bonds.

    A dihedral i-j-k-l is defined by a central bond j-k,
    where i is a neighbor of j and l is a neighbor of k.
    """
    if not hasattr(universe, "bonds") or len(universe.bonds) == 0:
        return []

    neighbors = {i: set() for i in range(len(universe.atoms))}
    for bond in universe.bonds.indices:
        i, j = bond
        neighbors[i].add(j)
        neighbors[j].add(i)

    dihedrals = set()

    for j, k in universe.bonds.indices:
        for i in neighbors[j] - {k}:
            for l_ in neighbors[k] - {j}:
                if i == l_:
                    continue
                dih = (i, j, k, l_)
                dih_rev = (l_, k, j, i)
                if dih_rev not in dihedrals:
                    dihedrals.add(dih)

    return list(dihedrals)


# def apply_clayff_rules(universe: mda.Universe) -> tuple[mda.Universe, dict]:
#     """Applies ClayFF to the universe."""
#     for rule in CLAYFF_RULES:
#         universe = apply_single_rule_to_universe(universe, rule)
#     return universe, {}


# def apply_cshff_rules(universe: mda.Universe) -> tuple[mda.Universe, dict]:
#     """Applies CSHFF to the universe and returns the calcium indices to modify."""
#     from ..build.cement_hydrates._silicate_helpers import get_interlayer_ca_indices

#     universe, _ = apply_clayff_rules(universe)
#     list_ids_cw = get_interlayer_ca_indices(universe)
#     actions = {"rename_atoms": (list_ids_cw, "Cw")}
#     return universe, actions
