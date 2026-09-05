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

import pandas as pd

from ...._constants import MASSES_DICT


class PDBReader:
    """Column offset tolerant PDB file reader."""

    @classmethod
    def read(cls, path: str) -> dict:
        atoms_data = []
        bonds_data = []
        box_params = None

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line_type = line[:6].strip()

            # Analysis of the box (CRYST1)
            if line_type == "CRYST1":
                parts = line.split()
                if len(parts) >= 7:
                    box_params = tuple(float(p) for p in parts[1:7])

            # Parsing atoms (ATOM / HETATM)
            elif line_type in ("ATOM", "HETATM"):
                parts = line.split()

                atom_id = int(parts[1])
                local_name = parts[2]

                # Dynamic extraction of the 3 coordinates (X, Y, Z) via Regex
                # Search for 3 consecutive floating numbers (ex: -4.280 -4.341 1.315)
                coords_match = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                # The 3 coordinates are the 3 decimal numbers that appear after the atom identifier
                floats = [float(val) for val in coords_match if "." in val]

                if len(floats) >= 3:
                    x, y, z = floats[0], floats[1], floats[2]
                else:
                    # Fallback safety if Regex does not find decimal places
                    x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])

                # The standard PDB format has no charge field: columns
                # after x/y/z are occupancy and temperature factor (both
                # written as fixed placeholders by `PDBWriter`, and by
                # Packmol's own output). Reading `floats[3]` back as
                # "charge" was actually reading occupancy (always 1.00),
                # silently overwriting every atom's real charge with 1.0
                # on any PDB round-trip.
                charge = 0.0

                # The element column is written upper-cased ("CA", "SI") by
                # the PDB standard, so it has to be normalized before being
                # looked up in the mass table. An entry that still doesn't
                # match a known element (a truncated symbol from an older
                # file, a non-standard label) falls back to the atom name.
                element = line[76:78].strip() if len(line) >= 78 else ""
                element = element.capitalize()
                if element not in MASSES_DICT:
                    element = cls._guess_element(local_name)

                atoms_data.append(
                    {
                        "id": atom_id,
                        "type": local_name,
                        "charge": charge,
                        "mass": cls._guess_mass(element),
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )

            # Connectivity analysis (CONECT)
            elif line_type == "CONECT":
                parts = line.split()
                source_id = int(parts[1])
                for target_id_str in parts[2:]:
                    target_id = int(target_id_str)
                    if source_id < target_id:
                        bonds_data.append((source_id, target_id))

        # Creation of DataFrames
        df_atoms = pd.DataFrame(atoms_data)
        if not df_atoms.empty:
            df_atoms.set_index("id", inplace=True)

        df_bonds = None
        if bonds_data:
            bond_rows = []
            for b_id, (a1, a2) in enumerate(bonds_data, start=1):
                t1 = df_atoms.loc[a1, "type"] if a1 in df_atoms.index else "X"
                t2 = df_atoms.loc[a2, "type"] if a2 in df_atoms.index else "X"
                bond_rows.append(
                    {
                        "id": b_id,
                        "type": f"{t1}-{t2}",
                        "atom_1": a1,
                        "atom_2": a2,
                    }
                )
            df_bonds = pd.DataFrame(bond_rows).set_index("id")

        # Configuration of the simulation box
        if box_params:
            a, b, c, _, _, _ = box_params
            lmp_box = ((0.0, a), (0.0, b), (0.0, c), (0.0, 0.0, 0.0))
        elif not df_atoms.empty:
            coords = df_atoms[["x", "y", "z"]].values
            mins = coords.min(axis=0) - 10.0
            maxs = coords.max(axis=0) + 10.0
            lmp_box = (
                (mins[0], maxs[0]),
                (mins[1], maxs[1]),
                (mins[2], maxs[2]),
                (0.0, 0.0, 0.0),
            )
        else:
            lmp_box = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 0.0, 0.0))

        charges_dict = {row["type"]: row["charge"] for row in atoms_data}
        masses_dict = {row["type"]: row["mass"] for row in atoms_data}

        return {
            "atoms": df_atoms,
            "bonds": df_bonds,
            "angles": None,
            "dihedrals": None,
            "impropers": None,
            "velocities": None,
            "box": lmp_box,
            "masses": masses_dict,
            "charges": charges_dict,
            "atom_style": "full",
        }

    @staticmethod
    def _guess_element(atom_type: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z]", "", atom_type)
        if not cleaned:
            return "C"
        if len(cleaned) >= 2 and cleaned[1].islower():
            return cleaned[:2].capitalize()
        return cleaned[0].upper()

    @classmethod
    def _guess_mass(cls, element: str) -> float:
        return MASSES_DICT.get(element, 12.011)


class PDBWriter:
    """Writing an AtomicSystem object in strict PDB format (Packmol compatible)."""

    @classmethod
    def write(cls, system, path: str) -> None:
        """
        Exports an AtomicSystem to a .pdb file conforming to the PDB standard.

        Settings
        ----------
        system: AtomicSystem
            Instance of the atomic system to export.
        path : str
            Path of the output file.
        """
        lines = []

        # Box header (CRYST1)
        if hasattr(system, "box") and system.box is not None:
            box = system.box
            a, b, c = box[0], box[1], box[2]
            alpha = box[3] if len(box) > 3 else 90.0
            beta = box[4] if len(box) > 4 else 90.0
            gamma = box[5] if len(box) > 5 else 90.0

            cryst_line = (
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
            )
            lines.append(cryst_line)

        # Atoms section (HETATM)
        atoms_df = system.atoms

        # Elements are inferred from the masses carried by the system, which
        # is the only reliable source here: the type names are force-field
        # labels ("Ow", "Hw", "Oc"...) and truncating them to their first
        # letter turned every two-letter element into another one -- "Ca"
        # was written as carbon, "Si" as sulfur -- so a PDB round-trip (i.e.
        # every Packmol build) silently replaced their masses.
        elements = dict(system.elements)

        for atom_id, row in atoms_df.iterrows():
            atom_type = str(row["type"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])

            # Atom name format on 4 characters
            if len(atom_type) < 4:
                formatted_atom_name = f" {atom_type:<3s}"
            else:
                formatted_atom_name = f"{atom_type[:4]:<4s}"

            # Element symbol (ex: C, Ca, Si), upper-cased as the PDB standard
            # requires, right-justified in columns 77-78.
            element = elements.get(row["type"], "")
            if not element:
                element = PDBReader._guess_element(atom_type)
            element = element.upper()[:2]

            # Strict Fortran/PDB column layout required by Packmol:
            # COLS 1-6 : "HETATM"
            # COLS 7-11 : Atom serial number (%5d)
            # COLS 13-16 : Atom name (%4s)
            # COLS 18-20 : Residue name (%3s)
            # COL 22 : Chain identifier (%1s)
            # COLS 23-26 : Residue sequence number (%4d)
            # COLS 31-38, 39-46, 47-54 : X, Y, Z (%8.3f)
            # COLS 55-60 : Occupancy (%6.2f)
            # COLS 61-66 : Temp factor (%6.2f)
            # COLS 77-78 : Element symbol (%2s)
            line = (
                f"HETATM{int(atom_id):5d} {formatted_atom_name} "
                f"UNK A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          "
                f"{element:>2s}"
            )
            lines.append(line)

        # Connections section (CONECT) - optional
        if (
            hasattr(system, "bonds")
            and system.bonds is not None
            and not system.bonds.empty
        ):
            adj = {}
            for _, row in system.bonds.iterrows():
                a1, a2 = int(row["atom_1"]), int(row["atom_2"])
                adj.setdefault(a1, []).append(a2)
                adj.setdefault(a2, []).append(a1)

            for source_id in sorted(adj.keys()):
                targets = sorted(adj[source_id])
                for i in range(0, len(targets), 4):
                    chunk = targets[i : i + 4]
                    targets_str = "".join(f"{t:5d}" for t in chunk)
                    lines.append(f"CONECT{source_id:5d}{targets_str}")

        lines.append("END")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
