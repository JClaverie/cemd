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
LAMMPS data file reader and writer.
"""

import datetime

import numpy as np
import pandas as pd

from .base import BaseReader, BaseWriter
from ..._forcefield.models import LJParams, BuckinghamParams, BondParams, AngleParams

class LammpsReader(BaseReader):
    """Read LAMMPS data files."""

    HEADERS = {
        'Atom Type Labels', 'Bond Type Labels', 'Angle Type Labels',
        'Dihedral Type Labels', 'Improper Type Labels', 'Pair Coeffs',
        'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
        'Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals',
        'Impropers', 'CS-Info'
    }

    @classmethod
    def read(cls, path: str) -> dict:
        """Read LAMMPS data file."""
        lines = cls._read_lines(path)
        sections = cls._parse_sections(lines)
        box = cls._parse_box(path)

        topology = {
            'lmp_box': box,
            'atoms': None,
            'bonds': None,
            'angles': None,
            'dihedrals': None,
            'impropers': None,
            'velocities': None,
            'masses': {},
            'charges': {},
            'atom_types': [],
            'bond_types': [],
            'angle_types': [],
            'dihedral_types': [],
            'improper_types': [],
            'atom_style': 'full',
            'pair_params': {},
            'bond_params': {},
            'angle_params': {},
            'dihedral_params': {},
            'improper_params': {},
        }

        cls._parse_section_data(sections, topology)
        return topology

    @staticmethod
    def _read_lines(path: str) -> list[str]:
        """Read and clean lines from file."""
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#"):
                    line = line.partition('#')[-1].strip()
                else:
                    line = line.partition('#')[0].strip()
                if line:
                    lines.append(line)
        return lines

    @staticmethod
    def _parse_sections(lines: list[str]) -> dict[str, list[str]]:
        """Parse sections from lines."""
        starts = [i for i, line in enumerate(lines) if line in LammpsReader.HEADERS]
        starts.append(None)
        return {lines[l]: lines[l+1:starts[i+1]] for i, l in enumerate(starts[:-1])}

    @staticmethod
    def _parse_box(path: str) -> tuple:
        """Parse box parameters from file."""
        xlo = xhi = ylo = yhi = zlo = zhi = 0.0
        xy = xz = yz = 0.0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                parts = stripped.split()
                if stripped.endswith('xlo xhi'):
                    xlo, xhi = float(parts[0]), float(parts[1])
                elif stripped.endswith('ylo yhi'):
                    ylo, yhi = float(parts[0]), float(parts[1])
                elif stripped.endswith('zlo zhi'):
                    zlo, zhi = float(parts[0]), float(parts[1])
                elif stripped.endswith('xy xz yz'):
                    xy, xz, yz = float(parts[0]), float(parts[1]), float(parts[2])

        return ((xlo, xhi), (ylo, yhi), (zlo, zhi), (xy, xz, yz))

    @classmethod
    def _parse_section_data(cls, sections: dict, topology: dict) -> None:
        """Parse each section."""
        # Les sections 'Atoms', 'Bonds', etc. sont parsées ici
        for key, values in sections.items():
            array = np.array([line.split() for line in values])

            if key == 'Atoms':
                cls._parse_atoms(array, topology)
            elif key == 'Bonds':
                cls._parse_connectivity(array, topology, 'bonds', 2)
            elif key == 'Angles':
                cls._parse_connectivity(array, topology, 'angles', 3)
            elif key == 'Dihedrals':
                cls._parse_connectivity(array, topology, 'dihedrals', 4)
            elif key == 'Impropers':
                cls._parse_connectivity(array, topology, 'impropers', 4)
            elif key == 'Masses':
                cls._parse_masses(array, topology)
            elif key == 'Atom Type Labels':
                topology['atom_types'] = list(array[:, 1])
            elif key == 'Pair Coeffs':
                cls._parse_coeffs(array, topology, 'pair_params')

    @staticmethod
    def _parse_atoms(array: np.ndarray, topology: dict) -> None:
        """Parse Atoms section."""
        n_cols = array.shape[1]
        if n_cols == 6:
            # atom_style 'charge'
            columns = ['id', 'type', 'charge', 'x', 'y', 'z']
        elif n_cols >= 7:
            # atom_style 'full' ou avec flags
            columns = ['id', 'type', 'charge', 'x', 'y', 'z']
            if n_cols == 10:
                array = array[:, [0, 2, 3, 4, 5, 6]]
            else:
                array = array[:, [0, 2, 3, 4, 5, 6]] if n_cols == 7 else array

        df = pd.DataFrame(array, columns=columns)
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df[['charge', 'x', 'y', 'z']] = df[['charge', 'x', 'y', 'z']].astype(float)
        df.set_index('id', inplace=True)
        topology['atoms'] = df
        topology['charges'] = dict(zip(df['type'], df['charge']))

    @staticmethod
    def _parse_connectivity(array: np.ndarray, topology: dict,
                            name: str, n_atoms: int) -> None:
        """Parse Bonds, Angles, etc."""
        if array.size == 0:
            topology[name] = None
            return

        columns = ['id', 'type'] + [f'atom_{i}' for i in range(1, n_atoms + 1)]
        df = pd.DataFrame(array, columns=columns)
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        for col in columns[2:]:
            df[col] = df[col].astype(int)
        df.set_index('id', inplace=True)
        topology[name] = df

    @staticmethod
    def _parse_masses(array: np.ndarray, topology: dict) -> None:
        """Parse Masses section."""
        masses = list(map(float, array[:, 1]))
        topology['masses'] = {str(i + 1): m for i, m in enumerate(masses)}

    @staticmethod
    def _parse_coeffs(array: np.ndarray, topology: dict, key: str, 
                    atom_types: list[str] = None,
                    bond_types: list[str] = None,
                    angle_types: list[str] = None) -> None:
        """
        Parse Pair Coeffs, Bond Coeffs, etc. and store as dataclasses.
        """
        params = {}
        
        if key == 'pair_params' and atom_types is not None:
            # Pair Coeffs
            for row in array:
                type_id = int(row[0])
                coeffs = [float(x) for x in row[1:]]
                label = str(atom_types[type_id - 1])
                
                if len(coeffs) == 2:
                    params[label] = LJParams(epsilon=coeffs[0], sigma=coeffs[1])
                elif len(coeffs) == 3:
                    params[label] = BuckinghamParams(A=coeffs[0], rho=coeffs[1], C=coeffs[2])
                else:
                    params[label] = coeffs
                    
        elif key == 'bond_params' and bond_types is not None:
            # Bond Coeffs
            for row in array:
                type_id = int(row[0])
                coeffs = [float(x) for x in row[1:]]
                label = str(bond_types[type_id - 1])
                
                if len(coeffs) >= 2:
                    params[label] = BondParams(k=coeffs[0], r0=coeffs[1])
                else:
                    params[label] = coeffs
                    
        elif key == 'angle_params' and angle_types is not None:
            # Angle Coeffs
            for row in array:
                type_id = int(row[0])
                coeffs = [float(x) for x in row[1:]]
                label = str(angle_types[type_id - 1])
                
                if len(coeffs) >= 2:
                    params[label] = AngleParams(k=coeffs[0], theta0=coeffs[1])
                else:
                    params[label] = coeffs
                    
        else:
            for row in array:
                type_id = int(row[0])
                coeffs = [float(x) for x in row[1:]]
                params[type_id] = coeffs
        
        topology[key] = params

class LammpsWriter(BaseWriter):
    """Write LAMMPS data files."""

    @classmethod
    def write(cls, system, path: str, atom_style: str = 'full',
              oldstyle: bool = False) -> None:
        """Write system to LAMMPS data file.
        
        Parameters
        ----------
        system : AtomicSystem
            System to write.
        path : str
            Output file path.
        atom_style : str, optional
            LAMMPS atom style ('full' or 'charge').
        oldstyle : bool, optional
            If True, write in old style compatible with VMD/topotools.
            This uses numeric IDs instead of text labels and comments.
        """
        with open(path, 'w', encoding='utf-8') as f:
            cls._write_header(f, system)
            cls._write_box(f, system)
            cls._write_labels(f, system, oldstyle)
            cls._write_coeffs(f, system, oldstyle)
            cls._write_masses(f, system, oldstyle)
            cls._write_atoms(f, system, atom_style)
            cls._write_velocities(f, system)
            cls._write_connectivity(f, system, 'bonds')
            cls._write_connectivity(f, system, 'angles')
            cls._write_connectivity(f, system, 'dihedrals')
            cls._write_connectivity(f, system, 'impropers')

    @staticmethod
    def _write_labels(f, system, oldstyle: bool) -> None:
        """Write type labels (only if not oldstyle)."""
        if oldstyle or not isinstance(system.atom_types[0], str):
            return

        f.write("\n Atom Type Labels\n\n")
        for i, t in enumerate(system.atom_types):
            f.write(f"{i+1} {t}\n")

        for name in ['bond', 'angle', 'dihedral', 'improper']:
            types = getattr(system, f'{name}_types')
            if types:
                f.write(f"\n {name.capitalize()} Type Labels\n\n")
                for i, t in enumerate(types):
                    f.write(f"{i+1} {t}\n")

    @staticmethod
    def _write_masses(f, system, oldstyle: bool) -> None:
        """Write masses."""
        f.write("\n Masses\n\n")
        if oldstyle or not isinstance(system.atom_types[0], str):
            # Utiliser des IDs numériques
            for i, (t, mass) in enumerate(zip(system.atom_types, system.masses), 1):
                f.write(f"{i} {mass}\n")
        else:
            # Utiliser les labels textuels
            for t, mass in zip(system.atom_types, system.masses):
                f.write(f"{t} {mass}\n")

    @staticmethod
    def _write_coeffs(f, system, oldstyle: bool) -> None:
        """Write Pair Coeffs, Bond Coeffs, etc."""
        
        def format_pair_coeff_line(id_i, id_j, params, label_i, label_j, is_ij_format=False):
            """Formate une ligne de Pair Coeffs ou PairIJ Coeffs."""
            if isinstance(params, BuckinghamParams):
                A, rho, C = params.A, params.rho, params.C
                prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                return f"{prefix} {A:>12.6e} {rho:>10.5f} {C:>12.6e}   # {label_i}-{label_j}\n"
            elif isinstance(params, LJParams):
                eps, sigma = params.epsilon, params.sigma
                prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                return f"{prefix} {eps:>12.4e} {sigma:>10.4f}   # {label_i}-{label_j}\n"
            else:
                # Fallback: liste de floats
                if len(params) == 3:
                    A, rho, C = params
                    prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                    return f"{prefix} {A:>12.6e} {rho:>10.5f} {C:>12.6e}   # {label_i}-{label_j}\n"
                else:
                    eps, sigma = params
                    prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                    return f"{prefix} {eps:>12.4e} {sigma:>10.4f}   # {label_i}-{label_j}\n"
        
        # ============================================================
        # Pair Coeffs / PairIJ Coeffs
        # ============================================================
        if system.pair_params:
            n = len(system.atom_types)
            actual_entries = len(system.pair_params)
            
            if oldstyle:
                f.write("\n# Pair Coeffs\n#\n")
                for i, t in enumerate(system.atom_types, 1):
                    f.write(f"# {i} {t}\n")
                f.write("\n")
                return
            
            # Déterminer si on utilise Pair Coeffs ou PairIJ Coeffs
            all_self = all(k[0] == k[1] for k in system.pair_params.keys())
            
            if all_self and actual_entries == n:
                # --- MODE PAIR COEFFS ---
                f.write("\n Pair Coeffs\n\n")
                for i in range(1, n + 1):
                    label_i = system.atom_types[i - 1]
                    params = system.pair_params.get((label_i, label_i))
                    if params is not None:
                        f.write(format_pair_coeff_line(i, i, params, label_i, label_i, is_ij_format=False))
            else:
                # --- MODE PAIRIJ COEFFS ---
                f.write("\n PairIJ Coeffs\n\n")
                for i in range(1, n + 1):
                    for j in range(i, n + 1):
                        label_i = system.atom_types[i - 1]
                        label_j = system.atom_types[j - 1]
                        
                        params = system.pair_params.get((label_i, label_j))
                        if params is None:
                            params = system.pair_params.get((label_j, label_i))
                        
                        if params is not None:
                            f.write(format_pair_coeff_line(i, j, params, label_i, label_j, is_ij_format=True))
            f.write("\n")
        
        # ============================================================
        # Bond Coeffs
        # ============================================================
        if system.bond_params:
            if oldstyle:
                f.write("\n# Bond Coeffs\n#\n")
                for i, t in enumerate(system.bond_types, 1):
                    f.write(f"# {i} {t}\n")
                f.write("\n")
            else:
                f.write("\n Bond Coeffs\n\n")
                for i, bond_label in enumerate(system.bond_types):
                    bid = i + 1
                    params = system.bond_params.get(bond_label)
                    if params is not None:
                        if isinstance(params, BondParams):
                            f.write(f"{bid:<3} {params.k:>12.4f} {params.r0:>10.4f}   # {bond_label}\n")
                        else:
                            f.write(f"{bid:<3} " + " ".join(f"{v:>12.4f}" for v in params) + f"   # {bond_label}\n")
                f.write("\n")
        
        # ============================================================
        # Angle Coeffs
        # ============================================================
        if system.angle_params:
            if oldstyle:
                f.write("\n# Angle Coeffs\n#\n")
                for i, t in enumerate(system.angle_types, 1):
                    f.write(f"# {i} {t}\n")
                f.write("\n")
            else:
                f.write("\n Angle Coeffs\n\n")
                for i, angle_label in enumerate(system.angle_types):
                    aid = i + 1
                    params = system.angle_params.get(angle_label)
                    if params is not None:
                        if isinstance(params, AngleParams):
                            f.write(f"{aid:<3} {params.k:>12.4f} {params.theta0:>10.4f}   # {angle_label}\n")
                        else:
                            f.write(f"{aid:<3} " + " ".join(f"{v:>12.4f}" for v in params) + f"   # {angle_label}\n")
                f.write("\n")
        
        # ============================================================
        # Dihedral Coeffs
        # ============================================================
        if system.dihedral_params:
            if oldstyle:
                f.write("\n# Dihedral Coeffs\n#\n")
                for i, t in enumerate(system.dihedral_types, 1):
                    f.write(f"# {i} {t}\n")
                f.write("\n")
            else:
                f.write("\n Dihedral Coeffs\n\n")
                for i, dihedral_label in enumerate(system.dihedral_types):
                    did = i + 1
                    params = system.dihedral_params.get(dihedral_label)
                    if params is not None:
                        if isinstance(params, (list, tuple)):
                            params_str = ' '.join(f"{v:>12.4f}" if isinstance(v, (int, float)) else str(v) for v in params)
                            f.write(f"{did:<3} {params_str}   # {dihedral_label}\n")
                        else:
                            f.write(f"{did:<3} {params}   # {dihedral_label}\n")
                f.write("\n")
        
        # ============================================================
        # Improper Coeffs
        # ============================================================
        if system.improper_params:
            if oldstyle:
                f.write("\n# Improper Coeffs\n#\n")
                for i, t in enumerate(system.improper_types, 1):
                    f.write(f"# {i} {t}\n")
                f.write("\n")
            else:
                f.write("\n Improper Coeffs\n\n")
                for i, improper_label in enumerate(system.improper_types):
                    impid = i + 1
                    params = system.improper_params.get(improper_label)
                    if params is not None:
                        if isinstance(params, AngleParams):
                            f.write(f"{impid:<3} {params.k:>12.4f} {params.theta0:>10.4f}   # {improper_label}\n")
                        elif isinstance(params, (list, tuple)):
                            params_str = ' '.join(f"{v:>12.4f}" if isinstance(v, (int, float)) else str(v) for v in params)
                            f.write(f"{impid:<3} {params_str}   # {improper_label}\n")
                        else:
                            f.write(f"{impid:<3} {params}   # {improper_label}\n")
                f.write("\n")

    @staticmethod
    def _write_header(f, system) -> None:
        """Write file header."""
        now = datetime.datetime.now()
        f.write(f"LAMMPS data file generated by cemd on {now.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f" {system.num_atoms} atoms\n")
        if system.num_bonds:
            f.write(f" {system.num_bonds} bonds\n")
        if system.num_angles:
            f.write(f" {system.num_angles} angles\n")
        if system.num_dihedrals:
            f.write(f" {system.num_dihedrals} dihedrals\n")
        if system.num_impropers:
            f.write(f" {system.num_impropers} impropers\n")

        f.write(f" {system.num_atom_types} atom types\n")
        if system.num_bond_types:
            f.write(f" {system.num_bond_types} bond types\n")
        if system.num_angle_types:
            f.write(f" {system.num_angle_types} angle types\n")
        if system.num_dihedral_types:
            f.write(f" {system.num_dihedral_types} dihedral types\n")
        if system.num_improper_types:
            f.write(f" {system.num_improper_types} improper types\n")

    @staticmethod
    def _write_box(f, system) -> None:
        """Write box parameters."""
        xlo, xhi = system._lmp_box[0]
        ylo, yhi = system._lmp_box[1]
        zlo, zhi = system._lmp_box[2]
        f.write(f"   {xlo:>14.6f} {xhi:>14.6f} xlo xhi\n")
        f.write(f"   {ylo:>14.6f} {yhi:>14.6f} ylo yhi\n")
        f.write(f"   {zlo:>14.6f} {zhi:>14.6f} zlo zhi\n")

        xy, xz, yz = system._lmp_box[3]
        if any(abs(v) > 1e-8 for v in (xy, xz, yz)):
            f.write(f"   {xy:>14.6f} {xz:>14.6f} {yz:>14.6f} xy xz yz\n")

    @staticmethod
    def _write_atoms(f, system, atom_style: str) -> None:
        """Write atoms."""
        f.write(f"\n Atoms # {atom_style}\n\n")
        df = system.atoms.copy()
        if atom_style == 'full':
            df.insert(0, 'molecule', np.ones(len(df), dtype=int))
        f.write(df.to_string(header=False, index_names=False))
        f.write("\n")

    @staticmethod
    def _write_velocities(f, system) -> None:
        """Write velocities."""
        if system.velocities is None:
            return
        velocities_sorted = system.velocities.sort_index()
        f.write("\n Velocities\n\n")
        f.write(velocities_sorted.to_string(header=False, index_names=False))
        f.write("\n")

    @staticmethod
    def _write_connectivity(f, system, name: str) -> None:
        """Write bonds, angles, etc."""
        df = getattr(system, name)
        if df is None or df.empty:
            return
        df_sorted = df.sort_index().copy()
        f.write(f"\n {name.capitalize()}\n\n")
        f.write(df_sorted.to_string(header=False, index_names=False))
        f.write("\n")