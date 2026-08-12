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

import re

import numpy as np
import pandas as pd

from ...._constants import MASSES_DICT
from .base import BaseReader


class MDAReader(BaseReader):
    """Read an MDAnalysis Universe or AtomGroup."""

    # ------------------------------------------------------------------
    # Atom information
    # ------------------------------------------------------------------

    @staticmethod
    def _get_elements(atoms) -> np.ndarray:
        """Get atom elements from MDAnalysis.

        The search order is:
        1. ``atoms.elements``
        2. ``atoms.names``
        3. ``atoms.types``

        Parameters
        ----------
        atoms
            MDAnalysis AtomGroup or Universe.atoms.

        Returns
        -------
        numpy.ndarray
            Chemical element for each atom.
        """

        def _clean_symbol(raw_name: str) -> str:
            """Extrait le symbole chimique (ex: C16 -> C, Fe2 -> Fe)."""
            match = re.match(r"^[A-Za-z]+", str(raw_name).strip())
            if match:
                elem = match.group()
                return elem[0].upper() + elem[1:].lower()
            return str(raw_name).strip()

        try:
            if hasattr(atoms, "elements"):
                elements = np.asarray(atoms.elements, dtype=str)
                if len(elements) == len(atoms) and np.all(
                    np.char.strip(elements) != ""
                ):
                    return np.asarray(
                        [str(element).strip() for element in elements],
                        dtype=str,
                    )
        except (AttributeError, ValueError, TypeError):
            pass

        try:
            if hasattr(atoms, "names"):
                names = np.asarray(atoms.names, dtype=str)
                if len(names) == len(atoms) and np.all(np.char.strip(names) != ""):
                    return np.asarray(
                        [_clean_symbol(name) for name in names],
                        dtype=str,
                    )
        except (AttributeError, ValueError, TypeError):
            pass

        try:
            if hasattr(atoms, "types"):
                types = np.asarray(atoms.types, dtype=str)
                if len(types) == len(atoms) and np.all(np.char.strip(types) != ""):
                    return np.asarray(
                        [_clean_symbol(atom_type) for atom_type in types],
                        dtype=str,
                    )
        except (AttributeError, ValueError, TypeError):
            pass

        raise ValueError(
            "Unable to determine atom elements from MDAnalysis. "
            "Neither 'elements', 'names' nor 'types' are available."
        )

    @staticmethod
    def _get_masses(
        atoms,
        elements: np.ndarray,
    ) -> np.ndarray:
        """Get atomic masses.

        Explicit masses stored in MDAnalysis are preferred if valid.
        Otherwise, masses are obtained from ``MASSES_DICT`` using elements.
        """
        try:
            masses = np.asarray(atoms.masses, dtype=float)

            # Si toutes les masses valent exactement 1.0, cela signifie probablement
            # que MDAnalysis a mis une valeur par défaut. On force le recalcul via MASSES_DICT.
            if (
                len(masses) == len(atoms)
                and np.all(np.isfinite(masses))
                and np.all(masses > 0)
                and not np.all(masses == 1.0)
            ):
                return masses

        except (AttributeError, ValueError, TypeError):
            pass

        # Recalcul des masses réelles avec MASSES_DICT basé sur les vrais éléments (C, H, O...)
        return np.asarray(
            [MASSES_DICT.get(str(element), 1.0) for element in elements],
            dtype=float,
        )

    @staticmethod
    def _get_atom_types(atoms) -> np.ndarray:
        """Get atom types from MDAnalysis.

        Atom types are taken from ``atoms.types``. If the topology does
        not provide atom types, atom names are used as a fallback.

        Parameters
        ----------
        atoms
            MDAnalysis AtomGroup.

        Returns
        -------
        numpy.ndarray
            Atom types as strings.
        """

        try:
            types = np.asarray(atoms.types, dtype=str)

            if len(types) == len(atoms) and np.all(np.char.strip(types) != ""):
                return np.asarray(
                    [str(atom_type).strip() for atom_type in types],
                    dtype=str,
                )

        except (AttributeError, ValueError, TypeError):
            pass

        # Fallback to atom names.
        try:
            names = np.asarray(atoms.names, dtype=str)

            if len(names) == len(atoms) and np.all(np.char.strip(names) != ""):
                return np.asarray(
                    [str(name).strip() for name in names],
                    dtype=str,
                )

        except (AttributeError, ValueError, TypeError):
            pass

        raise ValueError(
            "Unable to determine atom types from MDAnalysis. "
            "Neither 'types' nor 'names' are available."
        )

    @staticmethod
    def _get_atom_ids(atoms) -> np.ndarray:
        """Get atom IDs, generating sequential IDs if unavailable."""

        try:
            ids = np.asarray(atoms.ids, dtype=int)

            if len(ids) == len(atoms):
                return ids

        except (AttributeError, ValueError, TypeError):
            pass

        # Utilisation de getattr pour gérer proprement les Universes ou AtomGroups
        n_atoms = getattr(atoms, "n_atoms", len(atoms))
        return np.arange(1, n_atoms + 1, dtype=int)

    @staticmethod
    def _get_charges(atoms) -> np.ndarray:
        """Get atomic charges, defaulting to zero."""

        try:
            charges = np.asarray(atoms.charges, dtype=float)

            if len(charges) == len(atoms):
                return charges

        except (AttributeError, ValueError, TypeError):
            pass

        return np.zeros(len(atoms), dtype=float)

    @staticmethod
    def _get_velocities(atoms) -> np.ndarray | None:
        """Get atomic velocities if available."""

        try:
            velocities = np.asarray(atoms.velocities, dtype=float)

            if velocities.shape == (len(atoms), 3) and np.all(np.isfinite(velocities)):
                return velocities

        except (AttributeError, ValueError, TypeError):
            pass

        return None

    # ------------------------------------------------------------------
    # Connection information
    # ------------------------------------------------------------------

    @staticmethod
    def _remap2numerical(
        types: np.ndarray,
    ) -> np.ndarray:
        """Remap connection types to numerical type IDs."""

        unique_types = np.unique(types)

        remap_dict = {
            type_name: index for index, type_name in enumerate(unique_types, start=1)
        }

        return np.asarray(
            [remap_dict[type_name] for type_name in types],
            dtype=int,
        )

    @staticmethod
    def _get_connection_types(
        conn,
        atom_types: np.ndarray,
        name: str,
        numerical_types: bool = False,
    ) -> np.ndarray:
        """Get connection types from an MDAnalysis connection."""

        n_atoms_map = {
            "bonds": 2,
            "angles": 3,
            "dihedrals": 4,
            "impropers": 4,
        }

        try:
            n_atoms = n_atoms_map[name]
        except KeyError as exc:
            raise ValueError(f"Unknown connection type: {name!r}") from exc

        types = []

        for atom_indices in conn.indices:
            if len(atom_indices) != n_atoms:
                continue

            connection_types = [str(atom_types[int(index)]) for index in atom_indices]

            # Normalize the type.
            #
            # H-O instead of O-H
            # C-C-H instead of H-C-C
            type_str = "-".join(sorted(connection_types))

            types.append(type_str)

        types_array = np.asarray(types, dtype=str)

        if numerical_types and len(types_array) > 0:
            types_array = MDAReader._remap2numerical(types_array)

        return types_array

    # ------------------------------------------------------------------
    # Box
    # ------------------------------------------------------------------

    @staticmethod
    def _get_box(universe, coordinates: np.ndarray):
        """Get the simulation box from MDAnalysis.

        MDAnalysis stores box dimensions as:

        ``[lx, ly, lz, alpha, beta, gamma]``.

        If no valid box is available, a non-periodic LAMMPS box is
        generated around the atomic coordinates.
        """

        try:
            dimensions = np.asarray(
                universe.dimensions,
                dtype=float,
            )

            if (
                dimensions.shape == (6,)
                and np.all(np.isfinite(dimensions))
                and np.all(dimensions[:3] > 0)
            ):
                return dimensions

        except (AttributeError, TypeError, ValueError):
            pass

        # No valid simulation box.
        mins = coordinates.min(axis=0) - 10.0
        maxs = coordinates.max(axis=0) + 10.0

        return (
            (float(mins[0]), float(maxs[0])),
            (float(mins[1]), float(maxs[1])),
            (float(mins[2]), float(maxs[2])),
            (0.0, 0.0, 0.0),
        )

    # ------------------------------------------------------------------
    # Main reader
    # ------------------------------------------------------------------

    @classmethod
    def read(cls, obj) -> dict:
        """Read an MDAnalysis Universe or AtomGroup."""

        universe = obj
        atoms = universe.atoms

        # --------------------------------------------------------------
        # Atom information
        # --------------------------------------------------------------

        ids = cls._get_atom_ids(atoms)

        elements = cls._get_elements(atoms)

        atom_types = cls._get_atom_types(atoms)

        positions = np.asarray(
            atoms.positions,
            dtype=float,
        )

        masses = cls._get_masses(
            atoms,
            elements,
        )

        charges = cls._get_charges(atoms)

        # --------------------------------------------------------------
        # Dictionaries
        # --------------------------------------------------------------

        univ_masses_dic = dict(
            sorted(
                {
                    str(atom_types): float(mass)
                    for atom_types, mass in zip(atom_types, masses)
                }.items()
            )
        )

        univ_charges_dic = dict(
            sorted(
                {
                    str(atom_type): float(charge)
                    for atom_type, charge in zip(
                        atom_types,
                        charges,
                    )
                }.items()
            )
        )

        # Connection types are numerical only when atom types are
        # numerical.
        numerical_types = all(str(atom_type).isdigit() for atom_type in atom_types)

        # --------------------------------------------------------------
        # Atoms
        # --------------------------------------------------------------

        stacked_arrays = np.column_stack(
            (
                ids,
                atom_types,
                charges,
                positions,
            )
        )

        df_atoms = pd.DataFrame(
            stacked_arrays,
            columns=[
                "id",
                "type",
                "charge",
                "x",
                "y",
                "z",
            ],
        )

        df_atoms["id"] = df_atoms["id"].astype(int)

        df_atoms[["charge", "x", "y", "z"]] = df_atoms[
            ["charge", "x", "y", "z"]
        ].astype(float)

        df_atoms.set_index(
            "id",
            inplace=True,
        )

        # --------------------------------------------------------------
        # Velocities
        # --------------------------------------------------------------

        velocities_array = cls._get_velocities(atoms)

        velocities = None

        if velocities_array is not None:
            df_velocities = pd.DataFrame(
                np.column_stack(
                    (
                        ids,
                        velocities_array,
                    )
                ),
                columns=[
                    "id",
                    "vx",
                    "vy",
                    "vz",
                ],
            )

            df_velocities["id"] = df_velocities["id"].astype(int)

            df_velocities[["vx", "vy", "vz"]] = df_velocities[
                ["vx", "vy", "vz"]
            ].astype(float)

            df_velocities.set_index(
                "id",
                inplace=True,
            )

            velocities = df_velocities

        # --------------------------------------------------------------
        # Connections
        # --------------------------------------------------------------

        def read_connections(
            name: str,
            n_atoms: int,
        ):
            """Read one MDAnalysis connection table."""

            conn = getattr(universe, name, None)

            if conn is None or len(conn) == 0:
                return None

            connection_indices = np.asarray(
                conn.indices,
                dtype=int,
            )

            connection_types = cls._get_connection_types(
                conn,
                atom_types,
                name,
                numerical_types,
            )

            # ----------------------------------------------------------
            # Universe
            # ----------------------------------------------------------

            if atoms is universe.atoms:
                connection_indices = connection_indices + 1

            # ----------------------------------------------------------
            # AtomGroup
            # ----------------------------------------------------------

            else:
                atom_index_to_id = {
                    int(atom_index): int(atom_id)
                    for atom_index, atom_id in zip(
                        atoms.indices,
                        ids,
                    )
                }

                valid_connections = []
                valid_types = []

                for conn_indices, type_name in zip(
                    connection_indices,
                    connection_types,
                ):
                    try:
                        mapped = [
                            atom_index_to_id[int(index)] for index in conn_indices
                        ]
                    except KeyError:
                        # The connection contains an atom outside
                        # the selected AtomGroup.
                        continue

                    if len(mapped) != n_atoms:
                        continue

                    valid_connections.append(mapped)
                    valid_types.append(type_name)

                connection_indices = np.asarray(
                    valid_connections,
                    dtype=int,
                )

                connection_types = np.asarray(
                    valid_types,
                )

            if len(connection_indices) == 0:
                return None

            columns = [
                "id",
                "type",
                *[f"atom_{i}" for i in range(1, n_atoms + 1)],
            ]

            connection_ids = np.arange(
                1,
                len(connection_indices) + 1,
                dtype=int,
            )

            stacked_arrays = np.column_stack(
                (
                    connection_ids,
                    connection_types,
                    connection_indices,
                )
            )

            df = pd.DataFrame(
                stacked_arrays,
                columns=columns,
            )

            df["id"] = df["id"].astype(int)

            atom_columns = [f"atom_{i}" for i in range(1, n_atoms + 1)]

            df[atom_columns] = df[atom_columns].astype(int)

            df.set_index(
                "id",
                inplace=True,
            )

            return df

        df_bonds = read_connections(
            "bonds",
            2,
        )

        df_angles = read_connections(
            "angles",
            3,
        )

        df_dihedrals = read_connections(
            "dihedrals",
            4,
        )

        df_impropers = read_connections(
            "impropers",
            4,
        )

        # --------------------------------------------------------------
        # Box
        # --------------------------------------------------------------

        box = cls._get_box(
            universe,
            positions,
        )

        # --------------------------------------------------------------
        # Topology
        # --------------------------------------------------------------

        return {
            "box": box,
            "masses": univ_masses_dic,
            "charges": univ_charges_dic,
            "atoms": df_atoms,
            "bonds": df_bonds,
            "angles": df_angles,
            "dihedrals": df_dihedrals,
            "impropers": df_impropers,
            "velocities": velocities,
            "atom_style": "full",
        }
