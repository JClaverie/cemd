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

        # Counts line (V2000):
        # columns 1-3: number of atoms
        # columns 4-6: number of bonds
        n_atoms = int(lines[3][0:3].strip())
        n_bonds = int(lines[3][3:6].strip())

        # ------------------------------------------------------------------
        # Atoms
        # ------------------------------------------------------------------
        atom_data = []
        atom_types = []

        atom_start = 4
        atom_end = atom_start + n_atoms

        for i in range(atom_start, atom_end):
            parts = lines[i].split()

            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            symbol = parts[3]

            atom_id = i - atom_start + 1

            atom_data.append([atom_id, symbol, 0.0, x, y, z])

            if symbol not in atom_types:
                atom_types.append(symbol)

        df_atoms = pd.DataFrame(
            atom_data,
            columns=["id", "type", "charge", "x", "y", "z"],
        ).set_index("id")

        # ------------------------------------------------------------------
        # Bonds
        # ------------------------------------------------------------------
        bond_data = []

        bond_start = atom_end
        bond_end = bond_start + n_bonds

        for i in range(bond_start, bond_end):
            parts = lines[i].split()

            atom1 = int(parts[0])
            atom2 = int(parts[1])

            bond_id = i - bond_start + 1

            type1 = df_atoms.loc[atom1, "type"]
            type2 = df_atoms.loc[atom2, "type"]

            # Canonical order
            if str(type1) > str(type2):
                type1, type2 = type2, type1

            connection_type = f"{type1}-{type2}"

            bond_data.append([bond_id, connection_type, atom1, atom2])

        df_bonds = pd.DataFrame(
            bond_data,
            columns=["id", "type", "atom_1", "atom_2"],
        ).set_index("id")

        # ------------------------------------------------------------------
        # Masses and charges
        # ------------------------------------------------------------------
        masses = {t: MASSES_DICT.get(t, 12.011) for t in atom_types}
        charges = {t: 0.0 for t in atom_types}

        # ------------------------------------------------------------------
        # Box
        # ------------------------------------------------------------------
        coords = df_atoms[["x", "y", "z"]].values
        mins, maxs = coords.min(axis=0) - 5, coords.max(axis=0) + 5

        lmp_box = (
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2]),
            (0, 0, 0),
        )

        return {
            "box": lmp_box,
            "masses": masses,
            "charges": charges,
            "atoms": df_atoms,
            "bonds": df_bonds,
            "angles": None,
            "dihedrals": None,
            "impropers": None,
            "velocities": None,
        }
