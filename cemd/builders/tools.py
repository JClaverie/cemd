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

from functools import singledispatch
from typing import Sequence

import numpy as np
import scipy.constants as cst

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



