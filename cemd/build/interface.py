# cemd/builders/interface.py
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

import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._constants import AVOGADRO, MASSES_DICT
from ..core._format import (
    lattice2vectors,
    vectors2lattice,
)
from ..core.atomic_system import AtomicSystem
from .base import require_program
from .solution import SolutionBuilder

if TYPE_CHECKING:
    pass


def _calculate_liquid_box(
    system: AtomicSystem, thickness: float, axis: str
) -> list[float]:
    """Calculate liquid box dimensions from the existing system."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map[axis.lower()]
    transverse_indices = [i for i in [0, 1, 2] if i != idx]

    boxv = lattice2vectors(system.box)

    dims = [0, 0, 0]
    dims[transverse_indices[0]] = boxv[transverse_indices[0]][transverse_indices[0]]
    dims[transverse_indices[1]] = boxv[transverse_indices[1]][transverse_indices[1]]
    dims[idx] = thickness

    return dims


def _build_droplet_from_blueprint(
    blueprint: SolutionBuilder, radius: float, axis: str = "z"
) -> AtomicSystem:
    """Build a hemispherical droplet from a blueprint."""
    from ._packmol import add_packmol_structure, get_structure_path, run_packmol

    require_program("packmol")

    # Hemisphere volume
    volume = 2 * np.pi * radius**3 / 3

    # Get accounts
    solute_counts = blueprint.to_counts(volume)

    # Calculate the number of water molecules
    water_molarmass = 2 * MASSES_DICT["H"] + MASSES_DICT["O"]
    mass_solutes_uma = blueprint.get_solute_mass(volume)
    mass_solutes_g = mass_solutes_uma / AVOGADRO
    mass_total_target_g = blueprint.density * volume * 1e-24
    mass_water_g = mass_total_target_g - mass_solutes_g

    if mass_water_g <= 0:
        raise ValueError("Target density or droplet volume is too low.")

    num_water = int(round((mass_water_g * AVOGADRO) / water_molarmass))

    # Plane pour Packmol
    if axis == "z":
        plane = "over plane 0 0 1 0"
    elif axis == "y":
        plane = "over plane 0 1 0 0"
    elif axis == "x":
        plane = "over plane 1 0 0 0"
    else:
        raise ValueError(f"Invalid axis: {axis}")

    # Build with Packmol
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        tmp_path = Path(tmp)
        structures = []

        # Eau
        h2o_path = get_structure_path("H2O", tmp_path)
        structures.append(
            add_packmol_structure(
                h2o_path, num_water, f"inside sphere 0 0 0 {radius:.4f}", plane
            )
        )

        # Solutes
        for species, count in solute_counts.items():
            if count <= 0:
                continue

            if species in blueprint.structures:
                # Remplacement de os.path.join par l'opérateur /
                struct_path = tmp_path / f"custom_{species}.pdb"
                # On convertit en str si la méthode write attend une chaîne de caractères
                blueprint.structures[species].write(str(struct_path))
            else:
                struct_path = get_structure_path(species, tmp_path)

            structures.append(
                add_packmol_structure(
                    struct_path, count, f"inside sphere 0 0 0 {radius:.4f}", plane
                )
            )

        data = run_packmol(structures)

    # Box
    box_size = radius * 2.5
    data.set_box([box_size, box_size, radius * 1.5, 90, 90, 90])

    return data


def _merge_data(
    system_a: AtomicSystem, system_b: AtomicSystem, box: np.ndarray
) -> AtomicSystem:
    """
    Merge two AtomicSystem objects.
    """

    output_topology = {}

    # Box
    output_topology["box"] = box

    # Merge atoms
    indices_a = system_a.atoms.index.to_numpy()
    indices_b = system_b.atoms.index.to_numpy() + system_a.num_atoms
    merged_indices = np.concatenate([indices_a, indices_b])
    merged_atoms = pd.concat([system_a.atoms, system_b.atoms], ignore_index=True)
    merged_atoms.index = merged_indices
    output_topology["atoms"] = merged_atoms

    # Merge velocities
    if system_a.velocities is not None and system_b.velocities is not None:
        output_topology["velocities"] = np.concatenate(
            [system_a.velocities, system_b.velocities]
        )
    else:
        output_topology["velocities"] = None

    # Merge masses
    masses_dict_a = system_a.masses
    masses_dict_b = system_b.masses

    masses_dict_a.update(masses_dict_b)
    output_topology["masses"] = dict(sorted(masses_dict_a.items()))

    # Merge charges
    charges_dict_a = system_a.charges
    charges_dict_b = system_b.charges

    charges_dict_a.update(charges_dict_b)
    output_topology["charges"] = dict(sorted(charges_dict_a.items()))

    # Get topology info
    connectivity_keys = ["bonds", "angles", "dihedrals", "impropers"]
    a_connectivity = {
        "bonds": system_a.bonds,
        "angles": system_a.angles,
        "dihedrals": system_a.dihedrals,
        "impropers": system_a.impropers,
    }
    b_connectivity = {
        "bonds": system_b.bonds,
        "angles": system_b.angles,
        "dihedrals": system_b.dihedrals,
        "impropers": system_b.impropers,
    }

    for key in connectivity_keys:
        if b_connectivity[key] is not None:
            b_connectivity[key] = b_connectivity[key].copy()
            b_connectivity[key].iloc[:, 1:] = (
                b_connectivity[key].iloc[:, 1:].astype(int) + system_a.num_atoms
            )
        else:
            b_connectivity[key] = pd.DataFrame()

        if a_connectivity[key] is None:
            a_connectivity[key] = pd.DataFrame()

    param_attrs = [
        "pair_params",
        "bond_params",
        "angle_params",
        "dihedral_params",
        "improper_params",
        "bondbond_params",
        "bondangle_params",
    ]

    for attr in param_attrs:
        dict_a = getattr(system_a, attr, {}) or {}
        dict_b = getattr(system_b, attr, {}) or {}
        output_topology[attr] = merge_param_dicts(dict_a, dict_b, param_name=attr)

    # Set new topology
    for key in connectivity_keys:
        df_merged = pd.concat(
            [a_connectivity[key], b_connectivity[key]], ignore_index=True
        )

        if len(df_merged) > 0:
            df_merged.index = range(1, len(df_merged) + 1)
            output_topology[key] = df_merged
        else:
            output_topology[key] = None

    # Create the merged system
    merged_system = AtomicSystem(output_topology)
    merged_system.wrap()

    # Preserve metadata
    if hasattr(system_a, "_pmg_struct"):
        merged_system._pmg_struct = system_a._pmg_struct

    return merged_system


def _merge_structure(
    base_system: AtomicSystem,
    structure_to_add: AtomicSystem,
    distance: float,
    axis: str = "z",
    vacuum: float = 0.0,
) -> AtomicSystem:
    """
    Merge a structure onto the base system.

    This is the core merge function that handles:
    - Axis mapping
    - Center of mass alignment
    - Box update
    - Topology merging

    Parameters
    ----------
    base_system : AtomicSystem
        The base system (e.g., surface).
    structure_to_add : AtomicSystem
        The structure to add (e.g., droplet, liquid).
    distance : float
        Distance between the surface and the new structure.
    axis : str, default='z'
        Axis along which the addition is performed ('x', 'y', 'z').
    vacuum : float, default=0.0
        Empty space added after the new structure.

    Returns
    -------
    AtomicSystem
        The final combined system.

    Examples
    --------
    >>> result = merge_structure(surface, droplet, distance=2.0)
    """
    # Copy to avoid modifying the original
    structure_to_add = structure_to_add.copy()

    # Axis mapping
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_names = ["x", "y", "z"]
    idx = axis_map[axis.lower()]
    trans_indices = [i for i in [0, 1, 2] if i != idx]

    # Align centers of mass in transverse directions
    com_base = base_system.get_center_of_mass()
    com_struct = structure_to_add.get_center_of_mass()

    for i in trans_indices:
        t_name = axis_names[i]
        shift_trans = com_base[i] - com_struct[i]
        structure_to_add.atoms[t_name] += shift_trans

    # Offset along the principal axis
    base_surface = base_system.atoms[axis.lower()].max()
    struct_base = structure_to_add.atoms[axis.lower()].min()

    shift_main = (base_surface - struct_base) + distance
    structure_to_add.atoms[axis.lower()] += shift_main

    # Update box
    new_box = base_system.box.copy()
    base_min = base_system.atoms[axis.lower()].min()
    struct_top = structure_to_add.atoms[axis.lower()].max()

    box_vecs = lattice2vectors(new_box)
    box_vecs[idx][idx] = (struct_top - base_min) + vacuum + distance
    new_box = vectors2lattice(box_vecs)

    # Merge the two systems
    return _merge_data(base_system, structure_to_add, new_box)


def add_liquid_layer(
    system: AtomicSystem,
    blueprint: SolutionBuilder,
    thickness: float,
    distance: float = 2.0,
    vacuum: float = 0.0,
    axis: str = "z",
) -> AtomicSystem:
    """
    Add a liquid layer to a system.

    Parameters
    ----------
    system : AtomicSystem
        The base system (e.g., surface).
    blueprint : SolutionBuilder
        Solution blueprint.
    thickness : float
        Thickness of the liquid layer in Å.
    distance : float, default=2.0
        Distance between the surface and the liquid.
    vacuum : float, default=0.0
        Vacuum space above the liquid.
    axis : str, default='z'
        Axis along which to add the layer.

    Returns
    -------
    AtomicSystem
        The combined system.

    Examples
    --------
    >>> blueprint = SolutionBuilder.from_molarities(...)
    >>> result = add_liquid_layer(surface, blueprint, thickness=30.0)
    """
    liquid_box = _calculate_liquid_box(system, thickness, axis)
    liquid = blueprint.build(liquid_box)
    return _merge_structure(system, liquid, distance, axis, vacuum)


def add_droplet(
    system: AtomicSystem,
    blueprint: SolutionBuilder,
    radius: float,
    distance: float = 2.0,
    vacuum: float = 10.0,
    axis: str = "z",
) -> AtomicSystem:
    """
    Add a droplet to a system.

    Parameters
    ----------
    system : AtomicSystem
        The base system (e.g., surface).
    blueprint : SolutionBuilder
        Solution blueprint.
    radius : float
        Radius of the hemispherical droplet in Å.
    distance : float, default=2.0
        Distance between the surface and the droplet.
    vacuum : float, default=10.0
        Vacuum space above the droplet.
    axis : str, default='z'
        Axis along which the droplet sits.

    Returns
    -------
    AtomicSystem
        The combined system.

    Examples
    --------
    >>> blueprint = SolutionBuilder.from_molarities(...)
    >>> result = add_droplet(surface, blueprint, radius=15.0)
    """
    droplet = _build_droplet_from_blueprint(blueprint, radius, axis)
    return _merge_structure(system, droplet, distance, axis, vacuum)


def add_vacuum(system: AtomicSystem, thickness: float, axis: str = "z") -> AtomicSystem:
    """
    Add vacuum to a system.

    Parameters
    ----------
    system : AtomicSystem
        The base system.
    thickness : float
        Thickness of vacuum to add in Å.
    axis : str, default='z'
        Axis along which to add vacuum.

    Returns
    -------
    AtomicSystem
        The system with vacuum added.

    Examples
    --------
    >>> result = add_vacuum(system, thickness=10.0)
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map[axis.lower()]

    # Extend the box
    box = system.box.copy()
    box[idx] += thickness
    system.set_box(box)

    return system


def add_structure(
    solid_system: AtomicSystem,
    structure_to_add: AtomicSystem,
    distance: float = 2.0,
    axis: str = "z",
    vacuum: float = 10.0,
) -> AtomicSystem:
    """
    Add a structure to a solid surface at a given distance.

    Aligns Centers of Mass (COM) on the transverse axes.

    Parameters
    ----------
    solid_system : AtomicSystem
        The base solid surface.
    structure_to_add : AtomicSystem
        The structure to add (e.g., droplet, liquid).
    distance : float, default=2.0
        Distance between the surface and the new structure.
    axis : str, default='z'
        Axis along which the addition is performed ('x', 'y', 'z').
    vacuum : float, default=10.0
        Empty space added after the new structure.

    Returns
    -------
    AtomicSystem
        The final combined system.

    Examples
    --------
    >>> result = add_structure(surface, droplet, distance=2.0, vacuum=10.0)
    """
    from ..core.atomic_system import AtomicSystem

    if isinstance(solid_system, str):
        solid_system = AtomicSystem.from_file(solid_system)
    if isinstance(structure_to_add, str):
        structure_to_add = AtomicSystem.from_file(structure_to_add)
    else:
        structure_to_add = structure_to_add.copy()

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_names = ["x", "y", "z"]
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

    result = _merge_data(solid_system, structure_to_add, new_box)
    result.wrap()

    return result


def merge_param_dicts(
    dict_a: dict[Any, Any], dict_b: dict[Any, Any], param_name: str
) -> dict[Any, Any]:
    """
    Fusionne deux dictionnaires de paramètres avec détection des conflits.
    """
    merged = dict_a.copy()
    for key, val_b in dict_b.items():
        if key in merged:
            val_a = merged[key]
            # Si les paramètres sont différents pour la même clé -> Warning
            if val_a != val_b:
                warnings.warn(
                    f"Conflict detected in '{param_name}' for key '{key}':\n"
                    f"  System A: {val_a}\n"
                    f"  System B: {val_b}\n"
                    f"System B parameters will overwrite System A.",
                    category=UserWarning,
                    stacklevel=3,
                )
        merged[key] = val_b
    return merged
