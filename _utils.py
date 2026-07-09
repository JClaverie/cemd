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

from functools import singledispatch
from typing import Sequence

import numpy as np
import scipy.constants as cst


def lammps2lattice(box: tuple[
        tuple[float, float], # [xlo, xhi]
        tuple[float, float], # [ylo, yhi]
        tuple[float, float], # [zlo, zhi]
        tuple[float, float, float]]  # [xy, xz, yz]
        ) -> np.ndarray:
    """Return the lattice parameters corresponding to the input LAMMPS box parameters.

    Parameters
    ----------
        box
            Box parameters in the form: ((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz)

    """

    length_x = box[0][1] - box[0][0]
    length_y = box[1][1] - box[1][0]
    length_z = box[2][1] - box[2][0]
    tilt_xy, tilt_xz, tilt_yz = box[3]

    boxa = length_x
    boxb = ( length_y**2 + tilt_xy**2 ) ** 0.5
    boxc = ( length_z**2 + tilt_xz**2 + tilt_yz**2 ) ** 0.5

    alpha = np.acos( (tilt_xy * tilt_xz + length_y * tilt_yz) / boxb / boxc )
    beta = np.acos( tilt_xz / boxc )
    gamma = np.acos( tilt_xy / boxb)

    return np.array([boxa, boxb, boxc, np.degrees(alpha),
    np.degrees(beta), np.degrees(gamma)])

def lattice2vectors(box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the vectors corresponding to box parameters.

    Parameters
    ----------
        box
            Box parameters in the form: [a, b, c, alpha, beta, gamma]
    
    """

    boxa, boxb, boxc, alpha, beta, gamma = box

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    ax = boxa
    bx = boxb * np.cos(gamma)
    by = boxb * np.sin(gamma)
    cx = boxc * np.cos(beta)
    cy = ( boxb * boxc * np.cos(alpha) - bx * cx) / by
    cz = ( boxc**2 - cx**2 - cy**2 ) ** 0.5

    vec_a = np.array([ax, 0.0, 0.0])
    vec_b = np.array([bx, by, 0.0])
    vec_c = np.array([cx, cy, cz])

    return vec_a, vec_b, vec_c

def vectors2lattice(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return the lattice parameters corresponding to box vectors.

    Parameters
    ----------
        vectors
            Sequence of vectors (vec_a, vec_b, vec_c)

    Returns
    -------
    box
        Box parameters in the form: [a, b, c, alpha, beta, gamma]
    
    """

    vec_a, vec_b, vec_c = vectors

    boxa = ( np.sum(vec_a**2) ) ** (1/2)
    boxb = ( np.sum(vec_b**2) ) ** (1/2)
    boxc = ( np.sum(vec_c**2) ) ** (1/2)
    alpha = np.acos( np.dot(vec_b, vec_c) / (boxb * boxc) )
    beta = np.acos( np.dot(vec_a, vec_c) / (boxa * boxc) )
    gamma = np.acos( np.dot(vec_a, vec_b)  / (boxa * boxb) )

    alpha = np.degrees(alpha)
    beta = np.degrees(beta)
    gamma = np.degrees(gamma)

    return np.array([boxa, boxb, boxc, alpha, beta, gamma])

def lattice2lammps(box: np.ndarray | Sequence[float]
                   ) -> tuple[
        tuple[float, float], # [xlo, xhi]
        tuple[float, float], # [ylo, yhi]
        tuple[float, float], # [zlo, zhi]
        tuple[float, float, float]  # [xy, xz, yz]
    ]:
    """Return the LAMMPS box parameters corresponding to the input lattice parameters.

    Parameters
    ----------
        box: list of float
            Box parameters in the form: ((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))

    """

    vec_a, vec_b, vec_c = lattice2vectors(box)

    lx = vec_a[0]
    ly = vec_b[1]
    lz = vec_c[2]
    xy = vec_b[0]
    xz = vec_c[0]
    yz = vec_c[1]
    
    return (0, lx), (0, ly), (0, lz), (xy, xz, yz)


@singledispatch
def concentration2count(arg, box_or_volume) -> None:
    """Generic function to calculate particle counts from molarity/concentration.

    Supports either a scalar molarity (float/int) or a dictionary of 
    concentrations {species: molarity}.
    """
    raise TypeError(
        f"Unsupported argument type: {type(arg)}. "
        "Expected float/int (molarity) or dict {species: molarity}."
    )


# 1. Implementation for a single scalar molarity
@concentration2count.register(float)
@concentration2count.register(int)
def _(
    molarity: float | int, 
    volume: float
) -> tuple[int, float]:
    """Calculates the integer particle count and relative error for a single molarity.

    Parameters
    ----------
    molarity : float | int
        Target concentration in mol/L (M).
    volume : float
        Volume of the simulation box in cubic Angstroms (Å³).

    Returns
    -------
    tuple[int, float]
        - Particle count (integer).
        - Relative discretization error percentage (float).
    """
    if molarity == 0:
        return 0, 0.0

    theoretical_n = molarity * cst.Avogadro * volume * 1e-27
    final_count = int(round(theoretical_n))

    if theoretical_n > 0:
        error_pct = abs(final_count - theoretical_n) / theoretical_n
    else:
        error_pct = 0.0

    return final_count, error_pct


# 2. Implementation for a dictionary of concentrations
@concentration2count.register(dict)
def _(
    concentrations_dict: dict[str, float], 
    box: Sequence[float] | np.ndarray | float
) -> tuple[dict[str, int], dict[str, float]]:
    """Converts a dictionary of solute concentrations into integer particle counts.

    Parameters
    ----------
    concentrations_dict : dict[str, float]
        Dictionary mapping species names to target molarities in mol/L (M).
    box : Sequence[float] | np.ndarray | float
        Box dimensions [a, b, c] in Angstroms, or directly the volume in Å³.

    Returns
    -------
    tuple[dict[str, int], dict[str, float]]
        - Dictionary mapping species names to integer particle counts (`solutes_dict`).
        - Dictionary mapping species names to relative error percentages.
    """
    if isinstance(box, (list, tuple, np.ndarray)):
        boxa, boxb, boxc = box[:3]
        volume = float(boxa * boxb * boxc)
    else:
        volume = float(box)

    solutes_dict = {}
    errors_dict = {}

    for key, molarity in concentrations_dict.items():
        count, error_pct = concentration2count(molarity, volume)
        solutes_dict[key] = count
        errors_dict[key] = error_pct

    return solutes_dict, errors_dict


def grouped_average(array: np.ndarray, expected_size: int) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """
    Groups coordinates that are close to each other and returns their averages.
    Optimized to avoid large memory allocation and fix boolean ambiguity errors.
    """
    if array.size == 0:
        return np.array([]), [], []

    # Sort the array to make grouping much faster (O(N log N))
    sorted_arr = np.sort(array)
    tol = 0.2
    
    def get_groups(data, tolerance):
        # Find gaps between points larger than tolerance
        gaps = np.diff(data) > tolerance
        # Create group IDs (0, 0, 1, 1, 1, 2...)
        group_ids = np.concatenate(([0], np.cumsum(gaps)))
        return group_ids

    # Iterate tolerance until we reach the expected number of layers
    group_ids = get_groups(sorted_arr, tol)
    num_groups = group_ids[-1] + 1
    
    while num_groups > expected_size:
        tol += 0.2
        group_ids = get_groups(sorted_arr, tol)
        num_groups = group_ids[-1] + 1

    # Calculate averages and collect indices
    averaged_values = []
    indices_list = []
    counts = []
    
    for g_id in range(num_groups):
        # Get indices where the sorted array belongs to this group
        mask = (group_ids == g_id)
        group_data = sorted_arr[mask]
        
        averaged_values.append(np.mean(group_data))
        counts.append(len(group_data))
        
        # To get indices relative to the ORIGINAL (unsorted) array:
        # We find where original values match the values in this group
        original_indices = np.where(np.isin(array, group_data))[0]
        indices_list.append(original_indices)

    return np.array(averaged_values), indices_list, counts


