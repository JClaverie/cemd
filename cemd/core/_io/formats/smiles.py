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

import pandas as pd

from .base import BaseReader


class SmilesReader(BaseReader):
    """Read from SMILES string using RDKit."""

    @classmethod
    def read(cls, smiles: str) -> dict:
        """Read from SMILES string."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        if status == -1:
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)

        if status != -1:
            AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()

        # Atoms
        atom_data = []
        atom_types = []
        masses = {}
        charges = {}

        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            charge = float(atom.GetFormalCharge())

            if symbol not in atom_types:
                atom_types.append(symbol)
                masses[symbol] = atom.GetMass()
                charges[symbol] = charge

            atom_data.append(
                {
                    "id": i + 1,
                    "type": symbol,
                    "charge": charge,
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z,
                }
            )

        df_atoms = pd.DataFrame(atom_data).set_index("id")

        # Bonds
        bond_data = []
        for i, bond in enumerate(mol.GetBonds()):
            bond_data.append(
                {
                    "id": i + 1,
                    "type": f"{bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}",
                    "atom_1": bond.GetBeginAtomIdx() + 1,
                    "atom_2": bond.GetEndAtomIdx() + 1,
                }
            )
        df_bonds = pd.DataFrame(bond_data).set_index("id") if bond_data else None

        # Box
        coords = df_atoms[["x", "y", "z"]].values
        mins = coords.min(axis=0) - 10
        maxs = coords.max(axis=0) + 10
        lmp_box = (
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2]),
            (0.0, 0.0, 0.0),
        )

        return {
            "lmp_box": lmp_box,
            "masses": masses,
            "charges": charges,
            "atom_types": atom_types,
            "atoms": df_atoms,
            "bonds": df_bonds,
            "angles": None,
            "dihedrals": None,
            "impropers": None,
            "velocities": None,
            "atom_style": "full",
        }
