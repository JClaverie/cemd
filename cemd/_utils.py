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

import shutil
from collections.abc import Sequence
from functools import lru_cache

import numpy as np


def lammps2lattice(
    box: tuple[
        tuple[float, float],  # [xlo, xhi]
        tuple[float, float],  # [ylo, yhi]
        tuple[float, float],  # [zlo, zhi]
        tuple[float, float, float],
    ],  # [xy, xz, yz]
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
    boxb = (length_y**2 + tilt_xy**2) ** 0.5
    boxc = (length_z**2 + tilt_xz**2 + tilt_yz**2) ** 0.5

    alpha = np.acos((tilt_xy * tilt_xz + length_y * tilt_yz) / boxb / boxc)
    beta = np.acos(tilt_xz / boxc)
    gamma = np.acos(tilt_xy / boxb)

    return np.array(
        [boxa, boxb, boxc, np.degrees(alpha), np.degrees(beta), np.degrees(gamma)]
    )


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
    cy = (boxb * boxc * np.cos(alpha) - bx * cx) / by
    cz = (boxc**2 - cx**2 - cy**2) ** 0.5

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

    boxa = (np.sum(vec_a**2)) ** (1 / 2)
    boxb = (np.sum(vec_b**2)) ** (1 / 2)
    boxc = (np.sum(vec_c**2)) ** (1 / 2)
    alpha = np.acos(np.dot(vec_b, vec_c) / (boxb * boxc))
    beta = np.acos(np.dot(vec_a, vec_c) / (boxa * boxc))
    gamma = np.acos(np.dot(vec_a, vec_b) / (boxa * boxb))

    alpha = np.degrees(alpha)
    beta = np.degrees(beta)
    gamma = np.degrees(gamma)

    return np.array([boxa, boxb, boxc, alpha, beta, gamma])


def lattice2lammps(
    box: np.ndarray | Sequence[float],
) -> tuple[
    tuple[float, float],  # [xlo, xhi]
    tuple[float, float],  # [ylo, yhi]
    tuple[float, float],  # [zlo, zhi]
    tuple[float, float, float],  # [xy, xz, yz]
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


@lru_cache
def require_program(name) -> str:
    """Return the path of an external executable.

    Raises
    ------
    RuntimeError
        If the executable is not found in PATH.
    """
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found")
    return path
