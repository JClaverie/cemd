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

import os

import pandas as pd

from ...._constants import MASSES_DICT
from .base import BaseReader


class SDFReader(BaseReader):
    """Read SDF files."""

    @classmethod
    def read(cls, source: str) -> dict:
        """Read SDF file or string."""
        if "\n" in source.strip() or not os.path.exists(source):
            lines = source.splitlines()
        else:
            with open(source, encoding="utf-8") as f:
                lines = f.read().splitlines()

        if len(lines) < 4:
            raise ValueError("Invalid or empty SDF file.")

        n_atoms = int(lines[3][0:3].strip())

        atom_data = []
        atom_types = []

        for i in range(4, 4 + n_atoms):
            parts = lines[i].split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            symbol = parts[3]
            atom_id = i - 3
            atom_data.append([atom_id, symbol, 0.0, x, y, z])
            if symbol not in atom_types:
                atom_types.append(symbol)

        df_atoms = pd.DataFrame(
            atom_data, columns=["id", "type", "charge", "x", "y", "z"]
        ).set_index("id")

        masses = {t: MASSES_DICT.get(t, 12.011) for t in atom_types}
        charges = {t: 0.0 for t in atom_types}

        coords = df_atoms[["x", "y", "z"]].values
        mins, maxs = coords.min(axis=0) - 5, coords.max(axis=0) + 5
        lmp_box = (
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2]),
            (0, 0, 0),
        )

        return {
            "lmp_box": lmp_box,
            "masses": masses,
            "charges": charges,
            "atom_types": atom_types,
            "atoms": df_atoms,
            "bonds": None,
            "angles": None,
            "dihedrals": None,
            "impropers": None,
            "velocities": None,
        }
