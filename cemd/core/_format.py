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

# Cemd/core/formats.py

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


def normalize_dataframe(
    df: pd.DataFrame | None,
    columns: tuple[str, ...] | list[str] | dict[str, type],
    name: str,
) -> pd.DataFrame | None:
    """Validate and normalize a topology DataFrame.

    Ensures required columns exist and reorders them according to
    the defined standard.
    """
    if df is None:
        return None

    col_names = list(columns.keys()) if isinstance(columns, dict) else list(columns)

    missing = [col for col in col_names if col not in df.columns]

    # Si la colonne 'ff_key' est absente, on l'ajoute automatiquement avec du vide
    if "ff_key" in missing:
        df["ff_key"] = ""
        missing.remove("ff_key")

    if missing:
        raise ValueError(
            f"Invalid {name} DataFrame. "
            f"Missing columns: {missing}. "
            f"Expected: {col_names}."
        )

    return df.loc[:, col_names]


def normalize_property_to_dict(
    prop: list | tuple | np.ndarray | dict | None, atom_types: list
) -> dict:
    """
    Convert a sequence of atomic properties (masses, charges) into a dictionary
    mapping atom types to their respective values. If the property is already
    a dictionary, it is returned as is.

    Parameters
    ----------
    prop : list, tuple, np.ndarray, dict, or None
        The property to normalize.
    atom_types : list
        The list of atom types to use as keys if `prop` is a sequence.

    Returns
    -------
    dict
        A dictionary mapping atom types to property values.

    Raises
    ------
    ValueError
        If `prop` is a sequence but its length does not match `atom_types`.
    TypeError
        If `prop` is of an unsupported type.
    """
    if prop is None:
        return {}

    if isinstance(prop, dict):
        return prop

    if isinstance(prop, (list, tuple, np.ndarray)):
        if len(prop) != len(atom_types):
            raise ValueError(
                f"Length mismatch for property: expected {len(atom_types)} values "
                f"(to match atom_types), but got {len(prop)}."
            )
        # Zip lie chaque type d'atome à sa valeur correspondante dans la liste
        return dict(zip(atom_types, prop))

    raise TypeError(f"Expected dict or sequence, got {type(prop).__name__}")


# ----------------------------------------------------------------
# DataFrame formats — already implemented
# ----------------------------------------------------------------

ATOMS_COLUMNS = {
    "type": str | int,
    "ff_key": str,
    "charge": float,
    "x": float,
    "y": float,
    "z": float,
}

BONDS_COLUMNS = {
    "type": str | int,
    "ff_key": str,
    "atom_1": int,
    "atom_2": int,
}

ANGLES_COLUMNS = {
    "type": str | int,
    "ff_key": str,
    "atom_1": int,
    "atom_2": int,
    "atom_3": int,
}

DIHEDRALS_COLUMNS = {
    "type": str | int,
    "ff_key": str,
    "atom_1": int,
    "atom_2": int,
    "atom_3": int,
    "atom_4": int,
}

IMPROPERS_COLUMNS = DIHEDRALS_COLUMNS.copy()

VELOCITIES_COLUMNS = {
    "vx": float,
    "vy": float,
    "vz": float,
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

LatticeBox: TypeAlias = Sequence[float]

VectorsBox: TypeAlias = Sequence[Sequence[float]]

LammpsBox: TypeAlias = tuple[
    tuple[float, float],  # (xlo, xhi)
    tuple[float, float],  # (ylo, yhi)
    tuple[float, float],  # (zlo, zhi)
    tuple[float, float, float],  # (xy, xz, yz)
]


class BoxFormat(Enum):
    """Supported simulation box formats."""

    LAMMPS = "lammps"
    LATTICE = "lattice"
    VECTORS = "vectors"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_box_format(box) -> BoxFormat:
    """Automatically detect the format of a simulation box.

    Supported formats
    ------------------
    LATTICE
        ``[a, b, c, alpha, beta, gamma]``

    VECTORS
        ``[[ax, ay, az], [bx, by, bz], [cx, cy, cz]]``

    LAMMPS
        ``((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))``

    Parameters
    ----------
    box
        Box representation to inspect.

    Returns
    -------
    BoxFormat
        Detected box format.

    Raises
    ------
    ValueError
        If the format cannot be detected.
    """

    # NumPy arrays and array-like objects
    try:
        arr = np.asarray(box, dtype=float)

        if arr.ndim == 2 and arr.shape == (3, 3):
            return BoxFormat.VECTORS

        if arr.ndim == 1 and arr.size == 6:
            return BoxFormat.LATTICE

    except (ValueError, TypeError):
        pass

    # LAMMPS format:
    # ((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))
    if (
        isinstance(box, (tuple, list))
        and len(box) == 4
        and all(isinstance(item, (tuple, list)) for item in box)
        and all(len(item) == 2 for item in box[:3])
        and len(box[3]) == 3
    ):
        return BoxFormat.LAMMPS

    raise ValueError(
        f"Cannot detect box format. Got: {type(box)}, value: {box}\n"
        "Supported formats:\n"
        "  LATTICE : [a, b, c, alpha, beta, gamma]\n"
        "  LAMMPS  : ((xlo,xhi),(ylo,yhi),(zlo,zhi),(xy,xz,yz))\n"
        "  VECTORS : [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]"
    )


# ---------------------------------------------------------------------------
# LAMMPS <-> Lattice
# ---------------------------------------------------------------------------


def lammps2lattice(box: LammpsBox) -> np.ndarray:
    """Convert LAMMPS box parameters to lattice parameters.

    Parameters
    ----------
    box
        LAMMPS box parameters:

        ``((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))``

    Returns
    -------
    numpy.ndarray
        Lattice parameters:

        ``[a, b, c, alpha, beta, gamma]``
    """

    lx = box[0][1] - box[0][0]
    ly = box[1][1] - box[1][0]
    lz = box[2][1] - box[2][0]

    xy, xz, yz = box[3]

    a = lx
    b = np.sqrt(ly**2 + xy**2)
    c = np.sqrt(lz**2 + xz**2 + yz**2)

    alpha_cos = (xy * xz + ly * yz) / (b * c)
    beta_cos = xz / c
    gamma_cos = xy / b

    alpha = np.degrees(np.arccos(np.clip(alpha_cos, -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(beta_cos, -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(gamma_cos, -1.0, 1.0)))

    return np.array([a, b, c, alpha, beta, gamma])


def lattice2lammps(box: LatticeBox) -> LammpsBox:
    """Convert lattice parameters to LAMMPS box parameters.

    Parameters
    ----------
    box
        Lattice parameters:

        ``[a, b, c, alpha, beta, gamma]``

        Angles are given in degrees.

    Returns
    -------
    LammpsBox
        LAMMPS box parameters.
    """

    vectors = lattice2vectors(box)

    vec_a, vec_b, vec_c = vectors

    lx = vec_a[0]
    ly = vec_b[1]
    lz = vec_c[2]

    xy = vec_b[0]
    xz = vec_c[0]
    yz = vec_c[1]

    return (
        (0.0, float(lx)),
        (0.0, float(ly)),
        (0.0, float(lz)),
        (float(xy), float(xz), float(yz)),
    )


# ---------------------------------------------------------------------------
# Lattice <-> Vectors
# ---------------------------------------------------------------------------


def lattice2vectors(box: LatticeBox) -> np.ndarray:
    """Convert lattice parameters to box vectors.

    Parameters
    ----------
    box
        Lattice parameters:

        ``[a, b, c, alpha, beta, gamma]``

        Angles are given in degrees.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` array containing the box vectors as rows.
    """

    a, b, c, alpha, beta, gamma = np.asarray(box, dtype=float)

    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * np.cos(gamma)
    by = b * np.sin(gamma)
    bz = 0.0

    cx = c * np.cos(beta)

    cy = (b * c * np.cos(alpha) - bx * cx) / by

    cz_squared = c**2 - cx**2 - cy**2
    cz = np.sqrt(max(cz_squared, 0.0))

    return np.array(
        [
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz],
        ]
    )


def vectors2lattice(vectors: VectorsBox) -> np.ndarray:
    """Convert box vectors to lattice parameters.

    Parameters
    ----------
    vectors
        Three box vectors given as rows of a ``(3, 3)`` array.

    Returns
    -------
    numpy.ndarray
        Lattice parameters:

        ``[a, b, c, alpha, beta, gamma]``

        Angles are returned in degrees.
    """

    vectors = np.asarray(vectors, dtype=float)

    if vectors.shape != (3, 3):
        raise ValueError(
            f"Expected box vectors with shape (3, 3), got {vectors.shape}."
        )

    vec_a, vec_b, vec_c = vectors

    a = np.linalg.norm(vec_a)
    b = np.linalg.norm(vec_b)
    c = np.linalg.norm(vec_c)

    alpha_cos = np.dot(vec_b, vec_c) / (b * c)
    beta_cos = np.dot(vec_a, vec_c) / (a * c)
    gamma_cos = np.dot(vec_a, vec_b) / (a * b)

    alpha = np.degrees(np.arccos(np.clip(alpha_cos, -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(beta_cos, -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(gamma_cos, -1.0, 1.0)))

    return np.array([a, b, c, alpha, beta, gamma])


# ---------------------------------------------------------------------------
# LAMMPS <-> Vectors
# ---------------------------------------------------------------------------


def lammps2vectors(box: LammpsBox) -> np.ndarray:
    """Convert LAMMPS box parameters to box vectors.

    Parameters
    ----------
    box
        LAMMPS box parameters:

        ``((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))``

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` array containing the box vectors as rows.
    """

    lx = box[0][1] - box[0][0]
    ly = box[1][1] - box[1][0]
    lz = box[2][1] - box[2][0]

    xy, xz, yz = box[3]

    return np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ]
    )


def vectors2lammps(vectors: VectorsBox) -> LammpsBox:
    """Convert box vectors to LAMMPS box parameters.

    Parameters
    ----------
    vectors
        Three box vectors given as rows of a ``(3, 3)`` array.

    Returns
    -------
    LammpsBox
        LAMMPS box parameters.

    Notes
    -----
    The vectors are assumed to use the conventional LAMMPS
    triclinic representation:

    ``a = [lx, 0, 0]``

    ``b = [xy, ly, 0]``

    ``c = [xz, yz, lz]``
    """

    vectors = np.asarray(vectors, dtype=float)

    if vectors.shape != (3, 3):
        raise ValueError(
            f"Expected box vectors with shape (3, 3), got {vectors.shape}."
        )

    vec_a, vec_b, vec_c = vectors

    lx = vec_a[0]
    ly = vec_b[1]
    lz = vec_c[2]

    xy = vec_b[0]
    xz = vec_c[0]
    yz = vec_c[1]

    # Check that the vectors are compatible with the LAMMPS
    # restricted triclinic representation.
    if not np.allclose(
        vec_a[1:],
        0.0,
    ):
        raise ValueError(
            "Box vector a is not compatible with the "
            "LAMMPS restricted triclinic representation."
        )

    if not np.isclose(vec_b[2], 0.0):
        raise ValueError(
            "Box vector b is not compatible with the "
            "LAMMPS restricted triclinic representation."
        )

    return (
        (0.0, float(lx)),
        (0.0, float(ly)),
        (0.0, float(lz)),
        (float(xy), float(xz), float(yz)),
    )


# ---------------------------------------------------------------------------
# Generic normalization
# ---------------------------------------------------------------------------


def normalize_box(
    box,
    target: BoxFormat = BoxFormat.LAMMPS,
):
    """Convert a box to the requested format.

    Parameters
    ----------
    box
        Box in any supported format.

    target
        Desired output format. Defaults to ``BoxFormat.LAMMPS``.

    Returns
    -------
    numpy.ndarray or LammpsBox
        Box converted to the requested format.

    Examples
    --------
    >>> normalize_box([10, 20, 30, 90, 90, 90])
    ((0.0, 10.0), (0.0, 20.0), (0.0, 30.0), (0.0, 0.0, 0.0))

    >>> normalize_box(
    ...     np.eye(3) * 10,
    ...     target=BoxFormat.LATTICE,
    ... )
    array([10., 10., 10., 90., 90., 90.])
    """

    fmt = detect_box_format(box)

    if fmt == target:
        if target == BoxFormat.LAMMPS:
            return box

        return np.asarray(box, dtype=float)

    # ------------------------------------------------------------------
    # Convert to target format directly
    # ------------------------------------------------------------------

    if fmt == BoxFormat.LATTICE:
        if target == BoxFormat.VECTORS:
            return lattice2vectors(box)

        if target == BoxFormat.LAMMPS:
            return lattice2lammps(box)

    elif fmt == BoxFormat.VECTORS:
        if target == BoxFormat.LATTICE:
            return vectors2lattice(box)

        if target == BoxFormat.LAMMPS:
            return vectors2lammps(box)

    elif fmt == BoxFormat.LAMMPS:
        if target == BoxFormat.LATTICE:
            return lammps2lattice(box)

        if target == BoxFormat.VECTORS:
            return lammps2vectors(box)

    raise ValueError(f"Unsupported box conversion: {fmt.value} -> {target.value}")


# ----------------------------------------------------------------
# system_dict format
# ----------------------------------------------------------------

SYSTEM_DICT_FORMAT: dict[str, dict] = {
    # Mandatory keys
    "required": {
        "atoms": {"type": pd.DataFrame, "columns": ATOMS_COLUMNS},
        "box": {"type": (list, tuple, np.ndarray)},
        "atom_types": {"type": list},
        "masses": {"type": dict},
    },
    # Optional keys
    "optional": {
        "bonds": {
            "type": (pd.DataFrame, type(None)),
            "columns": BONDS_COLUMNS,
        },
        "angles": {
            "type": (pd.DataFrame, type(None)),
            "columns": ANGLES_COLUMNS,
        },
        "dihedrals": {
            "type": (pd.DataFrame, type(None)),
            "columns": DIHEDRALS_COLUMNS,
        },
        "impropers": {
            "type": (pd.DataFrame, type(None)),
            "columns": IMPROPERS_COLUMNS,
        },
        "velocities": {
            "type": (pd.DataFrame, type(None)),
            "columns": VELOCITIES_COLUMNS,
        },
        "charges": {"type": dict},
        "atom_style": {"type": str},
        "pair_params": {"type": dict},
        "bond_params": {"type": dict},
        "angle_params": {"type": dict},
        "dihedral_params": {"type": dict},
        "improper_params": {"type": dict},
    },
}


def validate_system_dict(d: dict[str, Any]) -> None:
    """Validate a system_dict against SYSTEM_DICT_FORMAT.

    Parameters
    ----------
    d : dict
        Dictionary to validate.

    Raises
    ------
    KeyError
        If a required key is missing.
    TypeError
        If a value has the wrong type.
    ValueError
        If a DataFrame has missing columns.
    """
    fmt = SYSTEM_DICT_FORMAT

    # Check mandatory keys
    for key, spec in fmt["required"].items():
        if key not in d:
            raise KeyError(f"system_dict missing required key: '{key}'")

        val = d[key]
        if not isinstance(val, spec["type"]):
            raise TypeError(
                f"system_dict['{key}'] must be {spec['type']}, got {type(val)}"
            )

        # Check columns of DataFrames
        if isinstance(val, pd.DataFrame) and "columns" in spec:
            missing = set(spec["columns"]) - set(val.columns)
            if missing:
                raise ValueError(f"system_dict['{key}'] missing columns: {missing}")

    # Check optional keys if present
    for key, spec in fmt["optional"].items():
        if key not in d:
            continue
        val = d[key]
        if val is not None and not isinstance(val, spec["type"]):
            raise TypeError(
                f"system_dict['{key}'] must be {spec['type']}, got {type(val)}"
            )
