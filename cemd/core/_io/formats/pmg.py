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

"""
Pymatgen reader.
"""

import numpy as np
import pandas as pd

from .base import BaseReader
from ...._constants import MASSES_DICT, CHARGES_DICT
from ...._utils import lattice2lammps

class PmgReader(BaseReader):
    """Read from Pymatgen Structure."""

    @classmethod
    def read(cls, structure) -> dict:
        """Read from Pymatgen Structure."""
        abc = structure.lattice.abc
        angles = structure.lattice.angles

        positions = structure.cart_coords
        types = [site.species.elements[0].name for site in structure]

        ids = np.arange(1, len(positions) + 1)
        unique_types = sorted(set(types))

        masses = {t: MASSES_DICT.get(t, 1.0) for t in unique_types}
        charges = {t: CHARGES_DICT.get(t, 0.0) for t in unique_types}
        charges_arr = np.array([CHARGES_DICT.get(t, 0.0) for t in types])

        df_atoms = pd.DataFrame({
            'id': ids,
            'type': types,
            'charge': charges_arr,
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
        }).set_index('id')

        return {
            'lmp_box': lattice2lammps(abc + angles),
            'masses': masses,
            'charges': charges,
            'atom_types': unique_types,
            'atoms': df_atoms,
            'bonds': None,
            'angles': None,
            'dihedrals': None,
            'impropers': None,
            'velocities': None,
            'atom_style': 'full',
        }