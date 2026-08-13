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

import re
from pathlib import Path

import pandas as pd

from ...._constants import MASSES_DICT
from .base import BaseReader


class LTReader(BaseReader):
    """Read LAMMPS template (LT) files from moltemplate."""

    @classmethod
    def read(cls, path: str, ff_dir: str | None = None) -> dict:
        """
        Read a moltemplate LT file and return a topology dictionary.
        """
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")

        # Parse molecule name and inherited FF
        mol_name, ff_name = cls._parse_inheritance(content)

        # Parse atoms - obtenir aussi le mapping nom -> id
        atoms_data, atom_ff_map, atom_name_to_id = cls._parse_atoms(content, mol_name)

        # Parse bonds, angles, dihedrals, impropers
        bonds_data, bond_ff_map = cls._parse_connectivity(
            content, "Bonds", atom_name_to_id
        )
        angles_data, angle_ff_map = cls._parse_connectivity(
            content, "Angles", atom_name_to_id
        )
        dihedrals_data, dihedral_ff_map = cls._parse_connectivity(
            content, "Dihedrals", atom_name_to_id
        )
        impropers_data, improper_ff_map = cls._parse_connectivity(
            content, "Impropers", atom_name_to_id
        )

        df_atoms = cls._create_atoms_df(atoms_data)  # ← Sans ff_name
        coords = df_atoms[["x", "y", "z"]].values
        mins = coords.min(axis=0) - 10
        maxs = coords.max(axis=0) + 10
        lmp_box = (
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2]),
            (0.0, 0.0, 0.0),
        )

        # Create topology dictionary
        topology = {
            "atoms": df_atoms,
            "bonds": cls._create_connectivity_df(bonds_data, 2),
            "angles": cls._create_connectivity_df(angles_data, 3),
            "dihedrals": cls._create_connectivity_df(dihedrals_data, 4),
            "impropers": cls._create_connectivity_df(impropers_data, 4),
            "velocities": None,
            "box": lmp_box,
            "masses": cls._create_masses_dict(atoms_data),
            "charges": cls._create_charges_dict(atoms_data),
            "atom_style": "full",
            "ff_name": ff_name,  # Nom du force field (pour référence)
            "pair_params": {},
            "bond_params": {},
            "angle_params": {},
            "dihedral_params": {},
            "improper_params": {},
            "atom_name_to_id": atom_name_to_id,
        }

        # Extraire les types et charges des atomes
        charges = {}
        for atom in atoms_data:
            charges[atom["local_name"]] = atom["charge"]

        topology["charges"] = charges

        return topology

    @classmethod
    def _parse_inheritance(cls, content: str) -> tuple[str, str | None]:
        """Parse the molecule name and inherited force field."""
        pattern = r"^(\w+)\s+inherits\s+(\w+)\s*\{"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1), match.group(2)
        return "unknown", None

    @classmethod
    def _parse_atoms(cls, content: str, mol_name: str) -> tuple[list[dict], dict, dict]:
        """
        Parse atom definitions from the Data Atoms section.

        Returns
        -------
        tuple[list[dict], dict, dict]
            (atoms_data, atom_ff_map, atom_name_to_id)
        """
        atoms = []
        atom_ff_map = {}
        atom_name_to_id = {}
        section = cls._extract_section(content, "Data Atoms")

        if not section:
            return atoms, atom_ff_map, atom_name_to_id

        pattern = r"\$atom:(\w+)\s+\$mol:\.\.\.\s+@atom:(\w+)\s+([-\d.]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"

        for line in section.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.search(pattern, line)
            if match:
                local_name = match.group(1)
                global_type = match.group(2)
                charge = float(match.group(3))
                x = float(match.group(4))
                y = float(match.group(5))
                z = float(match.group(6))

                atom_id = len(atoms) + 1
                atoms.append(
                    {
                        "id": atom_id,
                        "local_name": local_name,
                        "type": local_name,
                        "global_type": global_type,
                        "charge": charge,
                        "mass": MASSES_DICT[cls._guess_element(local_name)],
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

                atom_ff_map[local_name] = global_type
                atom_name_to_id[local_name] = atom_id

        return atoms, atom_ff_map, atom_name_to_id

    @classmethod
    def _parse_connectivity(
        cls, content: str, section_name: str, atom_name_to_id: dict
    ) -> tuple[list[dict], dict]:
        """
        Parse connectivity from LT file.

        Returns
        -------
        tuple[list[dict], dict]
            (items_data, ff_map)
        """
        items = []
        ff_map = {}
        section = cls._extract_section(content, f"Data {section_name}")

        if not section:
            return items, ff_map

        if section_name == "Bonds":
            pattern = r"\$bond:(\w+)\s+@bond:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Angles":
            pattern = r"\$angle:(\w+)\s+@angle:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Dihedrals":
            pattern = r"\$dihedral:(\w+)\s+@dihedral:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Impropers":
            pattern = r"\$improper:(\w+)\s+@improper:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        else:
            return items, ff_map

        for line in section.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                local_name = groups[0]
                global_type = groups[1]
                atom_names = list(groups[2:])

                # Convertir les noms d'atomes en IDs
                atom_ids = []
                for atom_name in atom_names:
                    atom_id = atom_name_to_id.get(atom_name)
                    if atom_id is None:
                        raise ValueError(f"Atom '{atom_name}' not found in atom list")
                    atom_ids.append(atom_id)

                # Créer le type descriptif
                desc_type = "-".join(atom_names)

                items.append(
                    {
                        "id": len(items) + 1,
                        "local_name": local_name,
                        "type": desc_type,
                        "global_type": global_type,
                        "atom_names": atom_names,
                        "atoms": atom_ids,
                    }
                )

                ff_map[desc_type] = global_type

        return items, ff_map

    @classmethod
    def _extract_section(cls, content: str, section_name: str) -> str:
        """Extract a specific section from the LT file content."""
        pattern = rf'write\("{section_name}"\)\s*{{(.*?)}}'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    @classmethod
    def _create_atoms_df(cls, atoms_data: list[dict]) -> pd.DataFrame:
        """Create atoms DataFrame from parsed atom data."""
        if not atoms_data:
            return pd.DataFrame(
                columns=["id", "type", "ff_key", "charge", "x", "y", "z"]
            )

        df = pd.DataFrame(atoms_data)
        df["ff_key"] = "gromos" + df["global_type"]
        df = df[["id", "type", "ff_key", "charge", "x", "y", "z"]]
        df.set_index("id", inplace=True)
        return df

    @classmethod
    def _create_masses_dict(cls, atoms_data: list[dict]) -> dict:
        """Create masses dictionary from parsed atom data."""
        masses = {}
        for atom in atoms_data:
            masses[atom["type"]] = atom["mass"]
        return masses

    @classmethod
    def _create_charges_dict(cls, atoms_data: list[dict]) -> dict:
        """Create charges dictionary from parsed atom data."""
        charges = {}
        for atom in atoms_data:
            charges[atom["type"]] = atom["charge"]
        return charges

    @classmethod
    def _create_connectivity_df(
        cls, items: list[dict], n_atoms: int
    ) -> pd.DataFrame | None:
        """Create connectivity DataFrame from parsed items."""
        if not items:
            return None

        data = []
        for item in items:
            # Créer ff_key avec le préfixe du force field en minuscule
            ff_key = f"gromos.{item['global_type']}"
            row = [item["id"], item["type"], ff_key] + item["atoms"]
            data.append(row)

        columns = ["id", "type", "ff_key"] + [
            f"atom_{i}" for i in range(1, n_atoms + 1)
        ]
        df = pd.DataFrame(data, columns=columns)
        df.set_index("id", inplace=True)
        return df

    @staticmethod
    def _guess_element(atom_type: str) -> str:
        """Guess the chemical element from an atom type name."""
        if not atom_type:
            raise ValueError("Empty atom type.")

        if len(atom_type) >= 2 and atom_type[1].isalpha():
            return atom_type[0].upper() + atom_type[1].lower()

        return atom_type[0].upper()
