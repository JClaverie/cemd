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
import re

import pandas as pd

from ...._constants import MASSES_DICT
from ..._forcefield.models import (
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicDihedralParams,
    HarmonicImproperParams,
    LJParams,
)
from .base import BaseReader


class LTReader(BaseReader):
    """Read LAMMPS template (LT) files from moltemplate."""

    @classmethod
    def read(cls, path: str, ff_dir: str | None = None) -> dict:
        """
        Read a moltemplate LT file and return a topology dictionary.

        Parameters
        ----------
        path : str
            Path to the LT file.
        ff_dir : str, optional
            Directory containing force field LT files.

        Returns
        -------
        dict
            Topology dictionary compatible with AtomicSystem.
        """
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Parse molecule name and inherited FF
        mol_name, ff_name = cls._parse_inheritance(content)

        # Parse atoms - obtenir aussi le mapping nom -> id
        atoms_data, atom_type_map, atom_ff_map, atom_name_to_id = cls._parse_atoms(
            content, mol_name
        )

        # Parse bonds, angles, dihedrals, impropers
        bonds_data, bond_type_map, bond_ff_map = cls._parse_connectivity(
            content, "Bonds", mol_name, atom_name_to_id
        )
        angles_data, angle_type_map, angle_ff_map = cls._parse_connectivity(
            content, "Angles", mol_name, atom_name_to_id
        )
        dihedrals_data, dihedral_type_map, dihedral_ff_map = cls._parse_connectivity(
            content, "Dihedrals", mol_name, atom_name_to_id
        )
        impropers_data, improper_type_map, improper_ff_map = cls._parse_connectivity(
            content, "Impropers", mol_name, atom_name_to_id
        )

        df_atoms = cls._create_atoms_df(atoms_data)
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
            "atoms": cls._create_atoms_df(atoms_data),
            "bonds": cls._create_connectivity_df(bonds_data, 2),
            "angles": cls._create_connectivity_df(angles_data, 3),
            "dihedrals": cls._create_connectivity_df(dihedrals_data, 4),
            "impropers": cls._create_connectivity_df(impropers_data, 4),
            "velocities": None,
            "box": lmp_box,
            "masses": cls._create_masses_dict(atoms_data),
            "charges": cls._create_charges_dict(atoms_data),
            "atom_style": "full",
            "pair_params": {},
            "bond_params": {},
            "angle_params": {},
            "dihedral_params": {},
            "improper_params": {},
            "atom_ff_mapping": atom_ff_map,
            "bond_ff_mapping": {},
            "angle_ff_mapping": {},
            "dihedral_ff_mapping": {},
            "improper_ff_mapping": {},
            "atom_name_to_id": atom_name_to_id,
        }

        # Extraire les types et charges des atomes
        types = set()
        charges = {}
        for atom in atoms_data:
            types.add(atom["local_name"])
            charges[atom["local_name"]] = atom["charge"]

        topology["charges"] = charges

        # Construire les mappings pour les bonds, angles, dihedrals, impropers
        for data, ff_mapping, _ in [
            (bonds_data, "bond_ff_mapping", "bond"),
            (angles_data, "angle_ff_mapping", "angle"),
            (dihedrals_data, "dihedral_ff_mapping", "dihedral"),
            (impropers_data, "improper_ff_mapping", "improper"),
        ]:
            for item in data:
                # Créer le type descriptif: "H6-O1" pour les bonds, "H6-O1-C1" pour les angles, etc.
                desc_type = "-".join(item["atom_names"])
                topology[ff_mapping][desc_type] = item["global_type"]

        # Try to load force field parameters if ff_dir is provided
        if ff_dir and ff_name:
            ff_params = cls._load_forcefield(ff_dir, ff_name)
            if ff_params:
                converted_params = cls._convert_ff_params(
                    ff_params,
                    atom_ff_map,
                    topology["bond_ff_mapping"],
                    topology["angle_ff_mapping"],
                    topology["dihedral_ff_mapping"],
                    topology["improper_ff_mapping"],
                )
                topology.update(converted_params)

        return topology

    @classmethod
    def _convert_ff_params(
        cls,
        ff_params: dict,
        atom_ff_map: dict,
        bond_ff_mapping: dict,
        angle_ff_mapping: dict,
        dihedral_ff_mapping: dict,
        improper_ff_mapping: dict,
    ) -> dict:
        """Convert FF parameters from global types to descriptive types."""
        converted = {
            "masses": {},
            "charges": {},
            "pair_params": {},
            "bond_params": {},
            "angle_params": {},
            "dihedral_params": {},
            "improper_params": {},
        }

        # Convert masses: global_type -> descriptive name
        for global_type, mass in ff_params.get("masses", {}).items():
            # Trouver le nom descriptif à partir du mapping inverse
            desc_name = None
            for desc, gtype in atom_ff_map.items():
                if gtype == global_type:
                    desc_name = desc
                    break
            if desc_name:
                converted["masses"][desc_name] = mass

        # Convert pair_params
        for key, params in ff_params.get("pair_params", {}).items():
            if isinstance(key, tuple):
                # Trouver les noms descriptifs
                desc1 = None
                desc2 = None
                for desc, gtype in atom_ff_map.items():
                    if gtype == key[0]:
                        desc1 = desc
                    if gtype == key[1]:
                        desc2 = desc
                if desc1 and desc2:
                    converted["pair_params"][(desc1, desc2)] = params
            else:
                for desc, gtype in atom_ff_map.items():
                    if gtype == key:
                        converted["pair_params"][desc] = params
                        break

        # Convert bond_params: global_type -> descriptive type
        for global_type, params in ff_params.get("bond_params", {}).items():
            for desc_type, gtype in bond_ff_mapping.items():
                if gtype == global_type:
                    converted["bond_params"][desc_type] = params
                    break

        # Convert angle_params
        for global_type, params in ff_params.get("angle_params", {}).items():
            for desc_type, gtype in angle_ff_mapping.items():
                if gtype == global_type:
                    converted["angle_params"][desc_type] = params
                    break

        # Convert dihedral_params
        for global_type, params in ff_params.get("dihedral_params", {}).items():
            for desc_type, gtype in dihedral_ff_mapping.items():
                if gtype == global_type:
                    converted["dihedral_params"][desc_type] = params
                    break

        # Convert improper_params
        for global_type, params in ff_params.get("improper_params", {}).items():
            for desc_type, gtype in improper_ff_mapping.items():
                if gtype == global_type:
                    converted["improper_params"][desc_type] = params
                    break

        return converted

    @classmethod
    def _load_forcefield(cls, ff_dir: str, ff_name: str) -> dict:
        """Load force field parameters from LT file."""
        possible_names = [
            f"{ff_name}.lt",
            f"{ff_name}.lammps.lt",
            f"ff_{ff_name}.lt",
            f"{ff_name}_ff.lt",
        ]

        for name in possible_names:
            ff_path = os.path.join(ff_dir, name)
            if os.path.exists(ff_path):
                return LTForceFieldReader.read(ff_path)

        return {}

    @classmethod
    def _parse_inheritance(cls, content: str) -> tuple[str, str | None]:
        """Parse the molecule name and inherited force field."""
        pattern = r"^(\w+)\s+inherits\s+(\w+)\s*\{"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1), match.group(2)
        return "unknown", None

    @classmethod
    def _parse_atoms(
        cls, content: str, mol_name: str
    ) -> tuple[list[dict], dict, dict, dict]:
        """
        Parse atom definitions from the Data Atoms section.

        Returns
        -------
        tuple[list[dict], dict, dict, dict]
            (atoms_data, atom_type_map, atom_ff_map, atom_name_to_id)
        """
        atoms = []
        atom_type_map = {}
        atom_ff_map = {}
        atom_name_to_id = {}
        section = cls._extract_section(content, "Data Atoms")

        if not section:
            return atoms, atom_type_map, atom_ff_map, atom_name_to_id

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

                atom_type_map[local_name] = global_type
                atom_ff_map[local_name] = global_type
                atom_name_to_id[local_name] = atom_id

        return atoms, atom_type_map, atom_ff_map, atom_name_to_id

    @classmethod
    def _parse_connectivity(
        cls, content: str, section_name: str, mol_name: str, atom_name_to_id: dict
    ) -> tuple[list[dict], dict, dict]:
        """
        Parse connectivity from LT file.

        Returns
        -------
        tuple[list[dict], dict, dict]
            (items_data, type_map, ff_map)
            items_data contient les IDs des atomes (pas les noms)
        """
        items = []
        type_map = {}
        ff_map = {}
        section = cls._extract_section(content, f"Data {section_name}")

        if not section:
            return items, type_map, ff_map

        if section_name == "Bonds":
            pattern = r"\$bond:(\w+)\s+@bond:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Angles":
            pattern = r"\$angle:(\w+)\s+@angle:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Dihedrals":
            pattern = r"\$dihedral:(\w+)\s+@dihedral:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        elif section_name == "Impropers":
            pattern = r"\$improper:(\w+)\s+@improper:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)\s+\$atom:(\w+)"
        else:
            return items, type_map, ff_map

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
                        "atom_names": atom_names,  # Pour les mappings
                        "atoms": atom_ids,  # IDs pour le DataFrame
                    }
                )

                # Mapping: type_descriptif -> local_name
                type_map[desc_type] = local_name
                # Mapping: type_descriptif -> global_type
                ff_map[desc_type] = global_type

        return items, type_map, ff_map

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
            return pd.DataFrame(columns=["id", "type", "charge", "x", "y", "z"])

        df = pd.DataFrame(atoms_data)
        df = df[["id", "type", "charge", "x", "y", "z"]]
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
            row = [item["id"], item["type"]] + item["atoms"]
            data.append(row)

        columns = ["id", "type"] + [f"atom_{i}" for i in range(1, n_atoms + 1)]
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


class LTForceFieldReader(BaseReader):
    """Read force field parameters from LT files (GROMOS style)."""

    @classmethod
    def read(cls, path: str) -> dict:
        """Read force field parameters from a GROMOS-style LT file."""
        with open(path, encoding="utf-8") as f:
            content = f.read()

        params = {
            "masses": {},
            "bond_params": {},
            "angle_params": {},
            "dihedral_params": {},
            "improper_params": {},
            "pair_params": {},
        }

        # Parse masses
        masses = cls._parse_masses(content)
        params["masses"].update(masses)

        # Parse pair coefficients (LJ)
        pair_params = cls._parse_pair_params(content)
        for pair_type, (eps, sigma) in pair_params.items():
            params["pair_params"][pair_type] = LJParams(epsilon=eps, sigma=sigma)

        # Parse bond coefficients
        bond_params = cls._parse_bond_params(content)
        for bond_type, (k, r0) in bond_params.items():
            params["bond_params"][bond_type] = HarmonicBondParams(k=k, r0=r0)

        # Parse angle coefficients
        angle_params = cls._parse_angle_params(content)
        for angle_type, (k, theta0) in angle_params.items():
            params["angle_params"][angle_type] = HarmonicAngleParams(k=k, theta0=theta0)

        # Parse dihedral coefficients
        dihedral_params = cls._parse_dihedral_params(content)
        for dihedral_type, (k, n, phi0) in dihedral_params.items():
            params["dihedral_params"][dihedral_type] = HarmonicDihedralParams(
                k=k, d=int(phi0), n=int(n)
            )

        # Parse improper coefficients
        improper_params = cls._parse_improper_params(content)
        for improper_type, (k, xi0) in improper_params.items():
            params["improper_params"][improper_type] = HarmonicImproperParams(
                k=k, chi0=xi0
            )

        return params

    @classmethod
    def _parse_masses(cls, content: str) -> dict:
        """Parse mass definitions from LT file."""
        masses = {}
        pattern = r"mass\s+@atom:(\w+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            atom_type = match.group(1)
            mass = float(match.group(2))
            masses[atom_type] = mass
        return masses

    @classmethod
    def _parse_pair_params(cls, content: str) -> dict:
        """Parse pair (LJ) parameters from LT file."""
        params = {}
        pattern = r"pair_coeff\s+@atom:(\w+)\s+@atom:(\w+)\s+([\d.]+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            type1 = match.group(1)
            type2 = match.group(2)
            eps = float(match.group(3))
            sigma = float(match.group(4))
            if type1 == type2:
                params[type1] = (eps, sigma)
            else:
                params[(type1, type2)] = (eps, sigma)
        return params

    @classmethod
    def _parse_bond_params(cls, content: str) -> dict:
        """Parse bond parameters from LT file."""
        params = {}
        pattern = r"bond_coeff\s+@bond:(\w+)\s+([\d.]+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            bond_type = match.group(1)
            k = float(match.group(2))
            r0 = float(match.group(3))
            params[bond_type] = (k, r0)
        return params

    @classmethod
    def _parse_angle_params(cls, content: str) -> dict:
        """Parse angle parameters from LT file."""
        params = {}
        pattern = r"angle_coeff\s+@angle:(\w+)\s+([\d.]+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            angle_type = match.group(1)
            k = float(match.group(2))
            theta0 = float(match.group(3))
            params[angle_type] = (k, theta0)
        return params

    @classmethod
    def _parse_dihedral_params(cls, content: str) -> dict:
        """Parse dihedral parameters from LT file."""
        params = {}
        pattern = r"dihedral_coeff\s+@dihedral:(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            dihedral_type = match.group(1)
            k = float(match.group(2))
            n = float(match.group(3))
            phi0 = float(match.group(4))
            params[dihedral_type] = (k, n, phi0)
        return params

    @classmethod
    def _parse_improper_params(cls, content: str) -> dict:
        """Parse improper parameters from LT file."""
        params = {}
        pattern = r"improper_coeff\s+@improper:(\w+)\s+([\d.]+)\s+([\d.]+)"
        for match in re.finditer(pattern, content):
            improper_type = match.group(1)
            k = float(match.group(2))
            xi0 = float(match.group(3))
            params[improper_type] = (k, xi0)
        return params


def read_lt(path: str, ff_dir: str | None = None) -> dict:
    """
    Read an LT file and optionally merge with force field parameters.

    Parameters
    ----------
    path : str
        Path to the molecule LT file.
    ff_dir : str, optional
        Directory containing force field LT files.

    Returns
    -------
    dict
        Topology dictionary compatible with AtomicSystem.
    """
    return LTReader.read(path, ff_dir)
