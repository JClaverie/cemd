#
# This file is part of the CEMD distribution
# Copyright (c) 2024-2026 Jérôme Claverie.
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

from __future__ import annotations

import os
import datetime
import warnings
import MDAnalysis as mda
import pandas as pd
import numpy as np

from pymatgen.core import Structure, Lattice
from typing import TYPE_CHECKING
from rdkit import Chem
from rdkit.Chem import AllChem

from .._utils import lattice2lammps
from .._constants import MASSES_DICT, CHARGES_DICT

if TYPE_CHECKING:
    from atomic_system import AtomicSystem

def read_lmp(file: str) -> dict:
    """Read a LAMMPS datafile and return a LAMMPS datafile

    Parameters
    ----------
        file: str
            Input LAMMPS datafile

    Returns
    ----------
        topology: dict
            All the information contained in a dictionnary

    """

    headers = set(['Atom Type Labels', 'Bond Type Labels', 'Angle Type Labels', 'Dihedral Type Labels', 'Improper Type Labels', 'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs', 'Masses', 'Atoms', 'Velocities', 'Bonds', 'Angles', 'Dihedrals', 'Impropers', 'CS-Info']) # CS-Info is a label in datafile from pyCSH

    type_label = False # will be true if the Atom Type Label section is found

    def iterdata():
        global types_in_coeffs # True if the atom types can be read in the "Coeffs" sections
        types_in_coeffs = False
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#"):
                    # check in types can be read in the "Coeffs" sections
                    if "Pair Coeffs" in line:
                        types_in_coeffs = True
                    line = line.partition('#')[-1].strip()
                else:
                    line = line.partition('#')[0].strip()
                if line:
                    yield line

    lines = list(iterdata())
    starts = [i for i, line in enumerate(lines)
                if line in headers]
    starts += [None]
    sections = {lines[l]:lines[l+1:starts[i+1]] 
for i, l in enumerate(starts[:-1])}

    # read box information
    tilt_xy, tilt_xz, tilt_yz = 0, 0, 0
    with open(file, encoding='utf8') as data:
        for line in data:
            lss = line.strip().split()

            if line.strip().endswith('xlo xhi'):
                xlo, xhi = list(float(i) for i in lss[:2])
            if line.strip().endswith('ylo yhi'):
                ylo, yhi = list(float(i) for i in lss[:2])
            if line.strip().endswith('zlo zhi'):
                zlo, zhi = list(float(i) for i in lss[:2])
            if line.strip().endswith('xy xz yz'):
                tilt_xy, tilt_xz, tilt_yz = list(float(i) for i in lss[:3])

    data.close()
    lmp_box = ((xlo, xhi), (ylo, yhi), (zlo, zhi), (tilt_xy, tilt_xz, tilt_yz))

    if "Atom Type Labels" in sections:
        type_label = True
    else:
        type_label = False

    # Gather data
    for key, value in sections.items():

        values = [line.split() for line in value]
        array = np.array(values)
        num_cols = array.shape[1]

        if key == "Atoms":
            columns = ['id', 'type', 'charge', 'x', 'y', 'z']
            # if num_cols == 5:
            #     atom_style = 'atomic'
            #     array = np.insert(array, 2, np.zeros(array.shape[0]), axis=1)
            if num_cols == 6:
                atom_style = 'charge'
            if num_cols == 7:
                array = array[:, np.r_[0, 2:len(array[0])]]
                atom_style = 'full'
            if num_cols == 9:
                array = array[:,:-3]
                atom_style = 'charge' # with flags
            if num_cols == 10:
                array = array[:, np.r_[0, 2:len(array[0])-3]]
                atom_style = 'full' # with flags
        
            atoms = pd.DataFrame(array, columns=columns)
            cols_to_fix = ['id', 'charge', 'x', 'y', 'z']
            atoms[cols_to_fix] = atoms[cols_to_fix].apply(pd.to_numeric, errors='coerce')
            atoms.set_index('id', inplace=True)
            type_charge_map = atoms[['type', 'charge']].drop_duplicates()
            charge_dict = dict(zip(type_charge_map['type'], type_charge_map['charge']))
            
        if key == "Velocities":

            columns = ['id', 'vx', 'vy', 'vz']
            velocities = pd.DataFrame(array, columns=columns)
            velocities = velocities.apply(pd.to_numeric, errors='coerce')
            velocities.set_index('id', inplace=True)

        if key == "Bonds":
            columns = ['id', 'type', 'atom_1', 'atom_2']
            bonds = pd.DataFrame(array, columns=columns)
            cols_to_fix = ['id', 'atom_1', 'atom_2']
            bonds[cols_to_fix] = bonds[cols_to_fix].apply(pd.to_numeric, errors='coerce')
            bonds.set_index('id', inplace=True)
            
        if key == "Angles":
            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3']
            angles = pd.DataFrame(array, columns=columns)
            cols_to_fix = ['id', 'atom_1', 'atom_2', 'atom_3']
            angles[cols_to_fix] = angles[cols_to_fix].apply(pd.to_numeric, errors='coerce')
            angles.set_index('id', inplace=True)

        if key == "Dihedrals":
            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            dihedrals = pd.DataFrame(array, columns=columns)
            cols_to_fix = ['id', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            dihedrals[cols_to_fix] = dihedrals[cols_to_fix].apply(pd.to_numeric, errors='coerce')
            dihedrals.set_index('id', inplace=True)

        if key == "Impropers":
            columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            impropers = pd.DataFrame(array, columns=columns)
            cols_to_fix = ['id', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
            impropers[cols_to_fix] = impropers[cols_to_fix].apply(pd.to_numeric, errors='coerce')
            impropers.set_index('id', inplace=True)

        if key == "Masses":
            masses_list = list(map(float, array[:,1]))

        if key == "Atom Type Labels":
            atom_types = list(array[:,1])
            
        if key == "Bond Type Labels":
            bond_types = list(array[:,1])

        if key == "Angle Type Labels":
            angle_types = list(array[:,1])

        if key == "Dihedral Type Labels":
            dihedral_types = list(array[:,1])

        if key == "Improper Type Labels":
            improper_types = list(array[:,1])

        if not type_label and types_in_coeffs:
            if key == "Pair Coeffs":
                atom_types = list(array[:,1])

            if key == "Bond Coeffs":
                bond_types = list(array[:,1])

            if key == "Angle Coeffs":
                angle_types = list(array[:,1])

            if key == "Dihedral Coeffs":
                dihedral_types = list(array[:,1])

            if key == "Improper Coeffs":
                improper_types = list(array[:,1])


    # for each sections, replace integer type label by string type label
    unique_atom_types = sorted(atoms.type.unique().tolist())
    if "Atoms" in sections:
        if type_label or (not type_label and types_in_coeffs):
            atoms.type.replace(unique_atom_types, atom_types, inplace=True)
    else:
        raise ValueError("No 'Atoms' section was found in the LAMMPS datafile.")

    if "Bonds" in sections:
        if type_label or (not type_label and types_in_coeffs):
            unique_bond_types = sorted(bonds.type.unique().tolist())
            bonds.type.replace(unique_bond_types, bond_types, inplace=True)
    else:
        bonds = None

    if "Angles" in sections:
        if type_label or (not type_label and types_in_coeffs):
            unique_angle_types = sorted(angles.type.unique().tolist())
            angles.type.replace(unique_angle_types, angle_types, inplace=True)
    else:
        angles = None

    if "Dihedrals" in sections:
        if type_label or (not type_label and types_in_coeffs):
            unique_dihedral_types = sorted(dihedrals.type.unique().tolist())
            dihedrals.type.replace(unique_dihedral_types, dihedral_types, inplace=True)
    else:
        dihedrals = None

    if "Impropers" in sections:
        if type_label or (not type_label and types_in_coeffs):
            unique_improper_types = sorted(impropers.type.unique().tolist())
            impropers.type.replace(unique_improper_types, improper_types, inplace=True)
    else:
        impropers = None

    if "Velocities" not in sections:
        velocities = None

    mass_dict = dict(zip([str(t) for t in unique_atom_types], masses_list))

    topology = {
        'lmp_box': lmp_box,
        'masses': mass_dict,
        'charges': charge_dict,
        'atom_types': [str(t) for t in unique_atom_types],
        'atoms': atoms,
        'bonds': bonds,
        'angles': angles,
        'dihedrals': dihedrals,
        'impropers': impropers,
        'velocities': velocities,
        'atom_style': atom_style
    }

    return topology

def write_lmp(lmp_data: AtomicSystem, 
              fout: str, 
              atom_style: str, 
              oldstyle: bool=False) -> None:
    """Write a LAMMPS datafile from a LAMMPSData object.

    Parameters
    ----------
        lmp_data
            AtomicSystem to write
        fout
            Output LAMMPS datafile
        atom_style
            LAMMPS atom_style to adopt to format the output file. Can be "full" or "charge"
        oldstyle
            Allows to write atom, bond, angle, dihedral and impropers types in commented 'Coeffs' sections. Readable by VMD via topotools.

    """

    if isinstance(lmp_data.atom_types[0], int):
        type_label = False
        oldstyle = False
    else: type_label = True

    with open(fout, 'w', encoding='utf-8') as ostream:

        # write the header
        now = datetime.datetime.now()
        time_format = "%d/%m/%Y %H:%M:%S"
        time = now.strftime(time_format)
        ostream.write(f"LAMMPS data file generated by cemd on {time}\n")

        # write system information
        ostream.write(f" {lmp_data.num_atoms} atoms\n")
        if lmp_data.num_bonds != 0:
            ostream.write(f" {lmp_data.num_bonds} bonds\n")
        if lmp_data.num_angles != 0:
            ostream.write(f" {lmp_data.num_angles} angles\n")
        if lmp_data.num_dihedrals != 0:
            ostream.write(f" {lmp_data.num_dihedrals} dihedrals\n")
        if lmp_data.num_impropers != 0:
            ostream.write(f" {lmp_data.num_impropers} impropers\n")

        ostream.write(f" {lmp_data.num_atom_types} atom types\n")
        if lmp_data.num_bond_types != 0:
            ostream.write(f" {lmp_data.num_bond_types} bond types\n")
        if lmp_data.num_angle_types != 0:
            ostream.write(f" {lmp_data.num_angle_types} angle types\n")
        if lmp_data.num_dihedral_types != 0:
            ostream.write(f" {lmp_data.num_dihedral_types} dihedral types\n")
        if lmp_data.num_improper_types != 0:
            ostream.write(f" {lmp_data.num_improper_types} improper types\n")

        # Extraction et formatage propre des dimensions de la boîte
        xlo, xhi = lmp_data._lmp_box[0]
        ylo, yhi = lmp_data._lmp_box[1]
        zlo, zhi = lmp_data._lmp_box[2]

        ostream.write(f"   {xlo:>14.6f} {xhi:>14.6f} xlo xhi\n") # pylint: disable=W0212
        ostream.write(f"   {ylo:>14.6f} {yhi:>14.6f} ylo yhi\n") # pylint: disable=W0212
        ostream.write(f"   {zlo:>14.6f} {zhi:>14.6f} zlo zhi\n") # pylint: disable=W0212

        # Remplacement de i != 0 par un test de valeur absolue pour éliminer le bruit machine (< 1e-8)
        if any(abs(i) > 1e-8 for i in lmp_data._lmp_box[3]): # pylint: disable=W0212
            xy, xz, yz = lmp_data._lmp_box[3]
            ostream.write(f"   {xy:>14.6f} {xz:>14.6f} {yz:>14.6f} xy xz yz\n") # pylint: disable=W0212

        # write labels  
        if type_label is True and oldstyle is False:
            # Atom Type Labels
            ostream.write("\n Atom Type Labels\n\n")
            for i, atom_type in enumerate(lmp_data.atom_types):
                ostream.write(f"{i+1} {atom_type}\n")

            # Bond Type Labels
            if lmp_data.num_bond_types != 0:
                ostream.write("\n Bond Type Labels\n\n")
                for i, bond_type in enumerate(lmp_data.bond_types):
                    ostream.write(f"{i+1} {bond_type}\n")

            # Angle Type Labels
            if lmp_data.num_angle_types != 0:
                ostream.write("\n Angle Type Labels\n\n")
                for i, angle_type in enumerate(lmp_data.angle_types):
                    ostream.write(f"{i+1} {angle_type}\n")

            # Dihedral Type Labels
            if lmp_data.num_dihedral_types != 0:
                ostream.write("\n Dihedral Type Labels\n\n")
                for i, dihedral_type in enumerate(lmp_data.dihedral_types):
                    ostream.write(f"{i+1} {dihedral_type}\n")

            # Improper Type Labels
            if lmp_data.num_improper_types != 0:
                ostream.write("\n Improper Type Labels\n\n")
                for i, improper_type in enumerate(lmp_data.improper_types):
                    ostream.write(f"{i+1} {improper_type}\n")

        # write FF coeffs
        if type_label is True:
            # ---ATOMS COEFFS /COMMENTS ---
            if oldstyle is False:
                if lmp_data.pair_params:
                    # Calculation of LAMMPS regulatory thresholds
                    n = len(lmp_data.atom_types)
                    actual_entries = len(lmp_data.pair_params)

                    # Creating reverse mappings [Text Label -> Numeric ID] for search
                    atom_label_to_id = {label: i + 1 for i, label in enumerate(lmp_data.atom_types)}
                    bond_label_to_id = {label: i + 1 for i, label in enumerate(lmp_data.bond_types)}
                    angle_label_to_id = {label: i + 1 for i, label in enumerate(lmp_data.angle_types)}

                    # Internal helper forcing the use of integers at the beginning and adding the label as a comment
                    def format_pair_coeff_line(id_i, id_j, params, label_i, label_j, is_ij_format=False):
                        if len(params) == 3:
                            A, rho, C = params
                            prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                            return f"{prefix} {A:>12.6e} {rho:>10.5f} {C:>12.6e}   # {label_i}-{label_j}\n"
                        else:
                            eps, sigma = params
                            prefix = f"{int(id_i):<3} {int(id_j):<3}" if is_ij_format else f"{int(id_i):<3}"
                            return f"{prefix} {eps:>12.4e} {sigma:>10.4f}   # {label_i}-{label_j}\n"

                    # ---AUTOMATIC CHOICE OF FORMAT & ORDERED WRITING ---
                    if actual_entries == n:
                        # ---MODE PAIR COEFFS ---
                        ostream.write("\n Pair Coeffs\n\n")
                        # We loop from 1 to n to guarantee increasing numerical order
                        for i in range(1, n + 1):
                            label_i = lmp_data.atom_types[i - 1]
                            
                            # We are looking for the pair (label_i, label_i) in your parameters
                            # We handle the case where the order of the keys of the original tuple is reversed
                            params = lmp_data.pair_params.get((label_i, label_i))
                            
                            if params is not None:
                                ostream.write(format_pair_coeff_line(i, i, params, label_i, label_i, is_ij_format=False))

                    else:
                        # ---MODE PAIRIJ COEFFS ---
                        ostream.write("\n PairIJ Coeffs\n\n")
                        # Double nested ordered loop imposing i <= j from 1 to n
                        for i in range(1, n + 1):
                            for j in range(i, n + 1):
                                label_i = lmp_data.atom_types[i - 1]
                                label_j = lmp_data.atom_types[j - 1]
                                
                                # Finding parameter configuration (checking both ways of tuple)
                                params = lmp_data.pair_params.get((label_i, label_j))
                                if params == None:
                                    params = lmp_data.pair_params.get((label_j, label_i))
                                    
                                if params is not None:
                                    ostream.write(format_pair_coeff_line(i, j, params, label_i, label_j, is_ij_format=True))

            else:  # oldstyle is True
                ostream.write("\n# Pair Coeffs\n#\n")
                for i, atom_type in enumerate(lmp_data.atom_types):
                    ostream.write(f"# {i+1} {atom_type}\n")

            # ---WRITING ORDERED LINK SECTIONS ---
            if hasattr(lmp_data, 'bond_params') and lmp_data.bond_params:
                ostream.write("\n Bond Coeffs\n\n")
                # We go through the official list of connections in order (1, 2, 3...)
                for i, bond_label in enumerate(lmp_data.bond_types):
                    bid = i + 1
                    params = lmp_data.bond_params.get(bond_label)
                    if params is not None:
                        # Style harmonic : K, r0
                        ostream.write(f"{bid:<3} {params[0]:>12.4f} {params[1]:>10.4f}   # {bond_label}\n")

            # --- ÉCRITURE DES SÉCTIONS D'ANGLES ORDONNÉES ---
                if hasattr(lmp_data, 'angle_params') and lmp_data.angle_params:
                    ostream.write("\n Angle Coeffs\n\n")
                    # On parcourt la liste officielle des angles dans l'ordre (1, 2, 3...)
                    for i, angle_label in enumerate(lmp_data.angle_types):
                        aid = i + 1
                        params = lmp_data.angle_params.get(angle_label)
                        if params is not None:
                            # Style harmonic : K, theta0
                            ostream.write(f"{aid:<3} {params[0]:>12.4f} {params[1]:>10.4f}   # {angle_label}\n")

            # ---DIHEDRALS COEFFS ---
            if lmp_data.num_dihedral_types != 0:
                if oldstyle is False:
                    if lmp_data.dihedral_params:
                        ostream.write("\n Dihedral Coeffs\n\n")
                        # On parcourt la liste officielle des dièdres pour imposer l'ordre (1, 2, 3...)
                        for i, dihedral_label in enumerate(lmp_data.dihedral_types):
                            did = i + 1
                            params = lmp_data.dihedral_params.get(dihedral_label)
                            if params is not None:
                                params_str = ' '.join(f"{p:>12.4f}" if isinstance(p, (int, float)) else str(p) for p in params)
                                ostream.write(f"{did:<3} {params_str}   # {dihedral_label}\n")
                else:  # oldstyle is True
                    ostream.write("\n# Dihedral Coeffs\n#\n")
                    for i, dihedral_type in enumerate(lmp_data.dihedral_types):
                        ostream.write(f"# {i+1} {dihedral_type}\n")

            # ---IMPROPERS COEFFS ---
            if lmp_data.num_improper_types != 0:
                if oldstyle is False:
                    if lmp_data.improper_params:
                        ostream.write("\n Improper Coeffs\n\n")
                        # On parcourt la liste officielle des impropres pour imposer l'ordre (1, 2, 3...)
                        for i, improper_label in enumerate(lmp_data.improper_types):
                            impid = i + 1
                            params = lmp_data.improper_params.get(improper_label)
                            if params is not None:
                                k, t = params
                                line = f"{impid:<3} {k:>12.4f} {t:>10.4f}   # {improper_label}\n"
                                ostream.write(line)
                else:  # oldstyle is True
                    ostream.write("\n# Improper Coeffs\n#\n")
                    for i, improper_type in enumerate(lmp_data.improper_types):
                        ostream.write(f"# {i+1} {improper_type}\n")

        # reset atom types to number if the oldstyle is selected
        if type_label is True and oldstyle is True:
            atom_types = lmp_data.atom_types
            lmp_data.set_types(list(range(1, lmp_data.num_atom_types + 1)))

        # write masses
        ostream.write("\n Masses\n\n")
        for i, mass in enumerate(lmp_data.masses):
            ostream.write(f"{lmp_data.atom_types[i]} {mass}\n")

        # write atoms
        ostream.write(f"\n Atoms # {atom_style}\n\n")
        df_atoms = lmp_data.atoms.copy()
        if atom_style == 'full':
            molecules = pd.Series(np.ones(lmp_data.num_atoms), dtype=int)
            molecules.index = df_atoms.index
            df_atoms.insert(0, 'molecule', molecules)
        ostream.write(df_atoms.to_string(header=False, index_names=False))
        ostream.write("\n\n")

        # write velocities
        if lmp_data.velocities is not None:
            ostream.write(" Velocities\n\n")
            ostream.write(lmp_data.velocities.to_string(header=False, index_names=False))
            ostream.write("\n\n")

        # write bonds
        if lmp_data.bonds is not None:
            ostream.write(" Bonds\n\n")
            ostream.write(lmp_data.bonds.to_string(header=False, index_names=False))
            ostream.write("\n\n")

        # write angles
        if lmp_data.angles is not None:
            ostream.write(" Angles\n\n")
            ostream.write(lmp_data.angles.to_string(header=False, index_names=False))
            ostream.write("\n\n")

        # write dihedrals
        if lmp_data.dihedrals is not None:
            ostream.write(" Dihedrals\n\n")
            ostream.write(lmp_data.dihedrals.to_string(header=False, index_names=False))
            ostream.write("\n\n")

        # write impropers
        if lmp_data.impropers is not None:
            ostream.write(" Impropers\n\n")
            ostream.write(lmp_data.impropers.to_string(header=False, index_names=False))
            ostream.write("\n\n")

        # reset atom types to label if the oldstyle is selected
        if type_label is True and oldstyle is True:
            lmp_data.set_types(atom_types)

def write_pdb(self, fout: str) -> None:
        """
        Writes the current system to a PDB file format.
        Forces uppercase for atom names and elements for maximum compatibility.
        """
        lines = []
        lines.append(f"REMARK   GENERATED BY CEMD (AtomicSystem)")
        
        b = self.box
        lines.append(f"CRYST1{b[0]:9.3f}{b[1]:9.3f}{b[2]:9.3f}{b[3]:7.2f}{b[4]:7.2f}{b[5]:7.2f} P 1           1")

        for idx, row in self.atoms.iterrows():
            pdb_id = idx % 100000
            
            # ---EDIT HERE ---
            # We force capital letters and we clean the types (ex: Ca -> CA)
            raw_type = str(row['type']).upper()
            atom_name = raw_type[:4].center(4)
            element_symbol = raw_type[:2].strip() # Often the first 2 letters
            
            res_name = "MOL"
            res_seq = 1
            
            line = (
                f"ATOM  {pdb_id:>5} {atom_name} {res_name:>3}  "
                f"{res_seq:>4}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}          "
                f"{element_symbol:>2}" 
            )
            lines.append(line)

        lines.append("END")

        with open(fout, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

def read_mda(mda_universe: mda.Universe) -> dict:
    """Convert a MDAnalysis Universe to a LAMMPSData object.

    Parameters
    ----------
        mda_universe
            A MDanalysis Universe or AtomGroup.

    Returns
    ----------
        topology
            All the information contained in a dictionnary

    """

    def _remap2numerical(types):
        '''Remap bonds, angles, ... to numbers if atom types are numericals.'''
        unique_types = np.unique(types)
        remap_dic = {k:v for k,v in zip(unique_types, 
        list(range(1, len(types) + 1))
        )}
        mapped_func = np.vectorize(remap_dic.get)
        return mapped_func(types)

    if mda_universe.dimensions is None:
        box = np.array([10, 10, 10, 90, 90, 90])
    else:
        box = mda_universe.dimensions

    indices = mda_universe.atoms.indices

    if mda_universe.atoms.ids is None:
        new_indices = np.arange(len(mda_universe.atoms))
        indices_remapping = {k:v for k,v in zip(indices, new_indices)}
        ids = new_indices + 1
        # Remapping function for indices in bonds, angles, dihedrals and impropers
        remap_indices = np.vectorize(lambda x: indices_remapping.get(x, x))
    else:
        ids = mda_universe.atoms.ids
    
    types = mda_universe.atoms.types
    masses = mda_universe.atoms.masses
    positions = mda_universe.atoms.positions

    try:
        charges = mda_universe.atoms.charges
    except: # pylint: disable=bare-except
        charges = np.zeros( len(ids))

    univ_masses_dic = {t: m for t, m in zip(types, masses)}
    univ_masses_dic = dict(sorted(univ_masses_dic.items()))

    univ_charges_dic = {t: c for t, c in zip(types, charges)}
    univ_charges_dic = dict(sorted(univ_charges_dic.items()))

    if isinstance(types[0], str):
        if types[0].isdigit():
            numerical_types = True
        else:
            numerical_types = False
    else:
        numerical_types = True

    stacked_arrays = np.column_stack((ids, types, charges, positions))
    columns = ['id', 'type', 'charge', 'x', 'y', 'z']
    df_atoms = pd.DataFrame(stacked_arrays, columns=columns)
    df_atoms['id'] = df_atoms['id'].astype(int)
    df_atoms[['charge', 'x', 'y', 'z']] = df_atoms[['charge', 'x', 'y', 'z']].astype(float)
    df_atoms.set_index('id', inplace=True)
    
    try:
        velocities = mda_universe.atoms.velocities
        columns = ['id', 'vx', 'vy', 'vz']
        velocities = mda_universe.atoms.velocities
        stacked_arrays = np.column_stack((
            ids, velocities))
        df_velocities = pd.DataFrame(stacked_arrays, columns=columns)
        df_velocities['id'] = df_velocities['id'].astype(int)
        df_velocities.set_index('id', inplace=True)
    except: # pylint: disable=bare-except
        df_velocities = None

    if hasattr(mda_universe, 'bonds') and len(mda_universe.bonds) != 0:
        if mda_universe.atoms.ids is None:
            bonds = remap_indices(mda_universe.bonds.indices) + 1
        else:
            bonds = mda_universe.bonds.indices + 1 

        atom1_type = mda_universe.bonds.atom1.types
        atom2_type = mda_universe.bonds.atom2.types

        sorted_types = np.sort(np.column_stack((atom1_type, atom2_type)), axis=1)
        bond_types = np.array([f"{i}-{j}" for i, j in sorted_types])
        if numerical_types:
            bond_types = _remap2numerical(bond_types)

        columns = ['id', 'type', 'atom_1', 'atom_2']
        ids = np.arange(1, len(bonds) + 1)
        stacked_arrays = np.column_stack((
            ids, bond_types, bonds))
        df_bonds = pd.DataFrame(stacked_arrays, columns=columns)
        df_bonds.set_index('id', inplace=True)
    else: # pylint: disable=bare-except
        df_bonds = None

    if hasattr(mda_universe, 'angles') and len(mda_universe.angles) != 0:
        if mda_universe.atoms.ids is None:
            angles = remap_indices(mda_universe.angles.indices) + 1
        else:
            angles = mda_universe.angles.indices + 1

        atom1_type = mda_universe.angles.atom1.types
        atom2_type = mda_universe.angles.atom2.types
        atom3_type = mda_universe.angles.atom3.types

        angle_types = np.array([f"{i}-{j}-{k}" for i, j, k in zip(atom1_type, atom2_type, atom3_type)])
        if numerical_types:
            angle_types = _remap2numerical(angle_types)

        columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3']
        ids = np.arange(1, len(angles) + 1)
        stacked_arrays = np.column_stack((
            ids, angle_types, angles))
        df_angles = pd.DataFrame(stacked_arrays, columns=columns)
        df_angles.set_index('id', inplace=True)
    else: # pylint: disable=bare-except
        df_angles = None

    if hasattr(mda_universe, 'dihedrals') and len(mda_universe.dihedrals) != 0:
        if mda_universe.atoms.ids is None:
            dihedrals = remap_indices(mda_universe.dihedrals.indices) + 1
        else:
            dihedrals = mda_universe.dihedrals.indices + 1

        atom1_type = mda_universe.dihedrals.atom1.types
        atom2_type = mda_universe.dihedrals.atom2.types
        atom3_type = mda_universe.dihedrals.atom3.types
        atom4_type = mda_universe.dihedrals.atom4.types

        dihedral_types = np.array([f"{i}-{j}-{k}-{l}" for i, j, k, l in zip(atom1_type, atom2_type, atom3_type, atom4_type)])
        if numerical_types:
            dihedral_types = _remap2numerical(dihedral_types)

        columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
        ids = np.arange(1, len(dihedrals) + 1)
        stacked_arrays = np.column_stack((
            ids, dihedral_types, dihedrals))
        df_dihedrals = pd.DataFrame(stacked_arrays, columns=columns)
        df_dihedrals.set_index('id', inplace=True)
    else: # pylint: disable=bare-except
        df_dihedrals = None

    if hasattr(mda_universe, 'impropers') and len(mda_universe.impropers) != 0:
        if mda_universe.atoms.ids is None:
            impropers = remap_indices(mda_universe.impropers.indices) + 1
        else:
            impropers = mda_universe.impropers.indices + 1

        atom1_type = mda_universe.impropers.atom1.type
        atom2_type = mda_universe.impropers.atom2.type
        atom3_type = mda_universe.impropers.atom3.type
        atom4_type = mda_universe.impropers.atom4.type

        improper_types = np.array([f"{i}-{j}-{k}-{l}" for i, j, k, l in zip(atom1_type, atom2_type, atom3_type, atom4_type)])
        if numerical_types:
            improper_types = _remap2numerical(improper_types)

        columns = ['id', 'type', 'atom_1', 'atom_2', 'atom_3', 'atom_4']
        ids = np.arange(1, len(impropers) + 1)
        stacked_arrays = np.column_stack((
            ids, improper_types, impropers))
        df_impropers = pd.DataFrame(stacked_arrays, columns=columns)
        df_impropers.set_index('id', inplace=True)
    else: # pylint: disable=bare-except
        df_impropers = None

    topology = {
        'lmp_box': lattice2lammps(box),
        'masses': univ_masses_dic,
        'charges': univ_charges_dic,
        'atom_types': sorted(np.unique(types).tolist()),
        'atoms': df_atoms,
        'bonds': df_bonds,
        'angles': df_angles,
        'dihedrals': df_dihedrals,
        'impropers': df_impropers,
        'velocities': df_velocities
    }

    return topology


def read_pmg(pmg_structure: Structure, 
             target_abc: tuple[float, float, float]) -> Structure:
    """
    Converts a Pymatgen Structure to a LAMMPS topology with axis realignment.
    
    If orig_struct is provided, the axes are ordered according to the original.
    Otherwise, the longest axis is placed on Z by default.
    """

    abc = list(pmg_structure.lattice.abc)
    angles = list(pmg_structure.lattice.angles)
    positions = pmg_structure.cart_coords

    from itertools import permutations
    
    best_mapping = [0, 1, 2]
    min_diff = float('inf')
    
    # On display there are 6 possible permutations (abc, acb, bac, bca, cab, cba)
    for p in permutations([0, 1, 2]):
        current_diff = sum(abs(abc[i] - target_abc[j]) for j, i in enumerate(p))
        if current_diff < min_diff:
            min_diff = current_diff
            best_mapping = list(p)

    # We only apply the change if the order has actually changed
    if best_mapping != [0, 1, 2]:
        print(f"read_pmg : Réalignement des axes (mapping: {best_mapping})")
        
        # Reorder mesh parameters
        abc = [abc[i] for i in best_mapping]
        angles = [angles[i] for i in best_mapping]
        
        # Reorder coordinate columns (X, Y, Z)
        # We use a copy to avoid problems with numpy views
        positions = positions[:, best_mapping].copy()

    box = np.array(abc + angles)

    types = []
    for i in range(pmg_structure.num_sites):
        types.append(pmg_structure[i].species.elements[0].name)

    ids = np.arange(1, len(positions) + 1)
    types = np.array(types)
    unique_types = sorted(np.unique(types).tolist())
    
    masses_dict = {t: MASSES_DICT.get(t, 0.0) for t in unique_types}
    charges_dict = {t: CHARGES_DICT.get(t, 0.0) for t in unique_types}
    charges = np.array([CHARGES_DICT.get(t, 0.0) for t in types])

    # Creating the Atoms DataFrame
    stacked_arrays = np.column_stack((ids, types, charges, positions))
    columns = ['id', 'type', 'charge', 'x', 'y', 'z']
    df_atoms = pd.DataFrame(stacked_arrays, columns=columns)
    
    # Forced conversion of data types
    df_atoms['id'] = df_atoms['id'].astype(int)
    df_atoms[['charge', 'x', 'y', 'z']] = df_atoms[['charge', 'x', 'y', 'z']].astype(float)
    df_atoms.set_index('id', inplace=True)

    # Assembling the topology
    topology = {
        'lmp_box': lattice2lammps(box),
        'masses': masses_dict,
        'charges': charges_dict,
        'atoms': df_atoms,
        'atom_types': unique_types,
        'bonds': None,
        'angles': None,
        'dihedrals': None,
        'impropers': None,
        'velocities': None
    }

    return topology


def read_sdf_content(source: str) -> dict:
    
    if "\n" in source.strip() or not os.path.exists(source):
        lines = source.splitlines()
    else:
        with open(source, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

    if len(lines) < 4:
        raise ValueError("Source SDF invalide ou vide.")

    n_atoms = int(lines[3][0:3].strip())
    # n_bonds = int(lines[3][3:6].strip())

    # store the types of atoms by their ID (base 1) for bond mapping
    atom_id_to_type = {}
    atom_data = []
    atom_types_list = []
    
    for i in range(4, 4 + n_atoms):
        parts = lines[i].split()
        x, y, z, symbol = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]
        atom_id = i - 3
        atom_id_to_type[atom_id] = symbol # Mapping crucial
        atom_data.append([atom_id, symbol, 0.0, x, y, z])
        if symbol not in atom_types_list:
            atom_types_list.append(symbol)

    atoms_df = pd.DataFrame(atom_data, columns=['id', 'type', 'charge', 'x', 'y', 'z']).set_index('id')

    # Final preparation (same as read_lmp)
    standard_masses = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007}
    mass_dict = {sym: standard_masses.get(sym, 12.011) for sym in atom_types_list}
    charge_dict = {sym: 0.0 for sym in atom_types_list}
    
    # Calculation of the box
    coords = atoms_df[['x', 'y', 'z']].values
    mins, maxs = coords.min(axis=0) - 5, coords.max(axis=0) + 5
    lmp_box = ((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]), (0, 0, 0))

    topology = {
        'lmp_box': lmp_box,
        'masses': mass_dict,
        'charges': charge_dict,
        'atom_types': atom_types_list,
        'atoms': atoms_df,
        'bonds': None, 'angles': None, 'dihedrals': None, 'impropers': None, 'velocities': None,
    }

    return topology


def read_smiles(smiles: str):
    """
    Converts a SMILES into a dictionary compatible with _assign_topology.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError("SMILES invalide")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status == -1:
        print(f"Standard ETKDG failed for {smiles}. Trying with random coordinates...")
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
         
    if status != -1:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
    else:
        print("Failed to generate structure from SMILES.")
    conf = mol.GetConformer()
    
    # Preparation of ATOMS (Pandas DataFrame)
    atom_data = []
    atom_types = []
    masses = {}
    charges = {}
    
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        charge = float(atom.GetFormalCharge())
        
        # use the symbol as a type (eg: 'C', 'H', 'O')
        atom_type = symbol 
        if atom_type not in atom_types:
            atom_types.append(atom_type)
            masses[atom_type] = atom.GetMass()
            charges[atom_type] = charge # Default load for this type
            
        atom_data.append({
            'id': i + 1,
            'type': atom_type,
            'charge': charge,
            'x': pos.x,
            'y': pos.y,
            'z': pos.z
        })
    
    atoms_df = pd.DataFrame(atom_data).set_index('id')

    # Preparing BONDS (Pandas DataFrame)
    bond_data = []
    for i, bond in enumerate(mol.GetBonds()):
        bond_data.append({
            'id': i + 1,
            'type': f"{bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}",
            'atom_1': bond.GetBeginAtomIdx() + 1,
            'atom_2': bond.GetEndAtomIdx() + 1
        })
    bonds_df = pd.DataFrame(bond_data).set_index('id') if bond_data else None

    # Calculation of the LAMMPS box (Padding of 10A around the molecule)
    coords = atoms_df[['x', 'y', 'z']].values
    mins = coords.min(axis=0) - 10
    maxs = coords.max(axis=0) + 10
    lmp_box = ((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]), (0.0, 0.0, 0.0))

    # Assembling the topology dictionary
    topology = {
        'atoms': atoms_df,
        'bonds': bonds_df,
        'angles': None,
        'dihedrals': None,
        'impropers': None,
        'velocities': None,
        'lmp_box': lmp_box,
        'atom_types': atom_types,
        'masses': masses,
        'charges': charges,
        'atom_style': 'full',
    }
    
    return topology


def to_pmg(topology: AtomicSystem | dict) -> Structure:
    """
    Converts a topology dictionary (or an object with an .atoms attribute)
    into a pymatgen Structure.

    Settings
    ----------
    topology
        Object containing 'atoms' and 'lmp_box', or the system object.

    Returns
    -------
    pmg_structure: pymatgen.core.Structure
    """
    # If we pass the system object directly
    if hasattr(topology, 'atoms'):
        df_atoms = topology.atoms
        # We assume that the object has a method to retrieve [a, b, c, alpha, beta, gamma]
        # Otherwise we use topology.box
        box_params = topology.box 
    else:
        df_atoms = topology['atoms']
        # If we have lmp_box (xhi, xlo...), we must convert it back to [a, b, c, alpha, beta, gamma]
        # Here I assume you have access to the standard mesh settings
        box_params = topology.get('box_params') 

    # Reconstruction of the Lattice (box)
    # box_params must be [a, b, c, alpha, beta, gamma]
    lattice = Lattice.from_parameters(*box_params)

    # Extraction of species and coordinates
    # We ensure that the types are indeed character strings (eg: 'C', 'H')
    species = df_atoms['type'].tolist()
    coords = df_atoms[['x', 'y', 'z']].to_numpy()
    
    # Load recovery (optional but useful for pymatgen)
    site_properties = None
    if 'charge' in df_atoms.columns:
        site_properties = {"charge": df_atoms['charge'].tolist()}

    # Creation of the Structure
    pmg_structure = Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=True,
        site_properties=site_properties
    )

    return pmg_structure


def to_mda(lmp_data: AtomicSystem) -> mda.Universe:
    """Convert a LAMMPSData object to a MDAnalysis Universe.

    Parameters
    ----------
        lmp_data: LAMMPSData
            An input LAMMPSData object.

    Returns
    -------
        Universe
            A MDAnalysis universe

    """

    univ = mda.Universe.empty(lmp_data.num_atoms, trajectory=True)
    univ.add_TopologyAttr('type', lmp_data.atoms.type.to_numpy())
    univ.add_TopologyAttr('name', lmp_data.atoms.type.to_numpy())
    univ.add_TopologyAttr('charge', lmp_data.atoms.charge.to_numpy())
    univ.add_TopologyAttr('ids', lmp_data.atoms.index)
    univ.atoms.positions = lmp_data.atoms[['x','y','z']].to_numpy()
    univ.dimensions = lmp_data.box

    univ_masses_dic = {t:m for t, m in zip(lmp_data.atom_types, lmp_data.masses)}
    masses = []
    for atype in lmp_data.atoms.type:
        masses.append(univ_masses_dic[atype])
    univ.add_TopologyAttr('masses', masses)

    if lmp_data.bonds is not None:
        bonds = list(zip(
            lmp_data.bonds['atom_1'].astype(int) - 1,
            lmp_data.bonds['atom_2'].astype(int) - 1))
        univ.add_TopologyAttr('bonds', bonds)

    if lmp_data.angles is not None:
        angles = list(zip(
            lmp_data.angles['atom_1'].astype(int) - 1,
            lmp_data.angles['atom_2'].astype(int) - 1,
            lmp_data.angles['atom_3'].astype(int) - 1))
        univ.add_TopologyAttr('angles', angles)

    if lmp_data.dihedrals is not None:
        dihedrals = list(zip(
            lmp_data.dihedrals['atom_1'].astype(int) - 1,
            lmp_data.dihedrals['atom_2'].astype(int) - 1,
            lmp_data.dihedrals['atom_3'].astype(int) - 1,
            lmp_data.dihedrals['atom_4'].astype(int) - 1))
        univ.add_TopologyAttr('dihedrals', dihedrals)

    if lmp_data.impropers is not None:
        impropers = list(zip(
            lmp_data.impropers['atom_1'].astype(int) - 1,
            lmp_data.impropers['atom_2'].astype(int) - 1,
            lmp_data.impropers['atom_3'].astype(int) - 1,
            lmp_data.impropers['atom_4'].astype(int) - 1))
        univ.add_TopologyAttr('impropers', impropers)

    if lmp_data.velocities is not None:
        univ.trajectory.ts.velocities = lmp_data.velocities

    return univ

def read_log(log_file: str) -> list[pd.DataFrame]:
    """Extract data from a LAMMPS log and return a list of Panda DataFrame.

    Parameters
    ----------
        logfile
            Path to LAMMPS log file

    Returns
    -------
    output_list

    """

    read = 0
    current_run = 0
    output_list = []

    with open(log_file, 'r', encoding='utf-8') as log:

        for line in log:

            line = line.strip()

            if line.startswith("Loop"):
                #reached end of data; stop writing
                read = 0
                data = np.asarray(data, dtype=np.float64)
                odf = pd.DataFrame(data, columns=header)
                if "Time" in odf.columns:
                    odf.set_index("Time", inplace=True)
                if "Step" in odf.columns:
                    odf.set_index("Step", inplace=True)
                output_list.append(odf)

            if read == 1:
                #if we are in data, write the data line to the data file
                data.append(line.split())

            if line.startswith("Time") or line.startswith("Step"):
                #if we see the header for data, switch to writing data
                data = []
                header = line.split()
                read = 1
                current_run += 1

    log.close()

    return output_list

