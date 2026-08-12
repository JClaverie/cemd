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
from dataclasses import dataclass

from ..forcefield_database import ForceFieldDatabase
from ..models import (
    AtomType,
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicImproperParams,
    LJParams,
)
from ._base import BaseForceFieldParser, ParseResult


@dataclass
class GromosDihedralParams:
    """Paramètres de dièdre GROMOS."""
    m: int
    k: float
    n: int
    d: float


class GromosLTParser(BaseForceFieldParser):
    """
    Parser pour les fichiers GROMOS au format moltemplate/LAMMPS (.lt).
    """
    
    def __init__(self, model_name: str = "GROMOS_54A7_ATB"):
        super().__init__()
        self.model_name = model_name
    
    def parse(self, content: str) -> ParseResult:
        """Parse une chaîne LT."""
        lines = self._clean_content(content)
        
        result = ParseResult(model_name=self.model_name)
        result.metadata = {
            'description': 'GROMOS 54A7 force field with ATB modifications',
            'ref': 'Schmid et al. (2011), Malde et al. (2011), Stroet et al. (2018)',
            'tags': ['GROMOS', '54A7', 'ATB', 'moltemplate']
        }
        
        # Parser chaque section
        self._parse_masses(lines, result)
        self._parse_pair_coeff(lines, result)
        self._parse_bond_coeff(lines, result)
        self._parse_angle_coeff(lines, result)
        self._parse_dihedral_coeff(lines, result)
        self._parse_improper_coeff(lines, result)
        
        return result
    
    def parse_file(self, filepath: str) -> ParseResult:
        """Parse un fichier LT."""
        with open(filepath) as f:
            content = f.read()
        return self.parse(content)
    
    def _strip_comments(self, line: str) -> str:
        """Enlève les commentaires # du format LT."""
        if '#' in line:
            return line[:line.index('#')]
        return line
    
    def _parse_masses(self, lines: list[str], result: ParseResult) -> None:
        """Parse les masses atomiques."""
        pattern = r'mass\s+@atom:(\w+)\s+([\d.]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                atom_type, mass = match.groups()
                result.atoms[atom_type] = AtomType(
                    element=self._guess_element(atom_type),
                    charge=0.0,
                    mass=float(mass),
                    environment='',
                    model=result.model_name
                )
    
    def _guess_element(self, atom_type: str) -> str:
        """Devine l'élément à partir du nom du type d'atome."""
        # Gérer les cas spéciaux
        special = {
            'OM': 'O', 'OA': 'O', 'OE': 'O', 'OW': 'O',
            'NT': 'N', 'NL': 'N', 'NR': 'N', 'NZ': 'N', 'NE': 'N',
            'CH0': 'C', 'CH1': 'C', 'CH2': 'C', 'CH3': 'C', 'CH4': 'C',
            'CH2r': 'C', 'CR1': 'C',
            'HC': 'H', 'HS14': 'H',
            'CU1+': 'Cu', 'CU2+': 'Cu',
            'ZN2+': 'Zn', 'MG2+': 'Mg', 'CA2+': 'Ca',
            'NA+': 'Na', 'CL-': 'Cl',
            'CLAro': 'Cl', 'CLOpt': 'Cl', 'CLChl': 'Cl', 'CLCl4': 'Cl',
            'BROpt': 'Br',
        }
        
        if atom_type in special:
            return special[atom_type]
        
        # Par défaut, prendre le premier caractère
        return atom_type[0]
    
    def _parse_pair_coeff(self, lines: list[str], result: ParseResult) -> None:
        """Parse les paramètres LJ."""
        pattern = r'pair_coeff\s+@atom:(\w+)\s+@atom:(\w+)\s+([\d.E+-]+)\s+([\d.E+-]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                type1, type2, epsilon, sigma = match.groups()
                
                # Clé canonique (ordre alphabétique)
                key = f"{type1}-{type2}" if type1 <= type2 else f"{type2}-{type1}"
                
                # Ignorer les paramètres à zéro (polar hydrogens)
                if float(epsilon) == 0.0 and float(sigma) == 0.0:
                    continue
                
                result.lj[key] = LJParams(
                    epsilon=float(epsilon),
                    sigma=float(sigma),
                    model=result.model_name
                )
    
    def _parse_bond_coeff(self, lines: list[str], result: ParseResult) -> None:
        """Parse les paramètres de liaison."""
        pattern = r'bond_coeff\s+@bond:(\w+)\s+([\d.E+-]+)\s+([\d.]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                bond_id, k, r0 = match.groups()
                result.bonds[bond_id] = HarmonicBondParams(
                    k=float(k),
                    r0=float(r0),
                    model=result.model_name
                )
    
    def _parse_angle_coeff(self, lines: list[str], result: ParseResult) -> None:
        """Parse les paramètres d'angle."""
        pattern = r'angle_coeff\s+@angle:(\w+)\s+([\d.E+-]+)\s+([\d.]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                angle_id, k, theta0 = match.groups()
                result.angles[angle_id] = HarmonicAngleParams(
                    k=float(k),
                    theta0=float(theta0),
                    model=result.model_name
                )
    
    def _parse_dihedral_coeff(self, lines: list[str], result: ParseResult) -> None:
        """Parse les paramètres de dièdre."""
        pattern = r'dihedral_coeff\s+@dihedral:(\w+)\s+([\d.]+)\s+([\d.E+-]+)\s+([\d.]+)\s+([\d.]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                dihedral_id, m, k, n, d = match.groups()
                result.dihedrals[dihedral_id] = GromosDihedralParams(
                    m=int(float(m)),
                    k=float(k),
                    n=int(float(n)),
                    d=float(d)
                )
    
    def _parse_improper_coeff(self, lines: list[str], result: ParseResult) -> None:
        """Parse les paramètres d'impropres."""
        pattern = r'improper_coeff\s+@improper:(\w+)\s+([\d.E+-]+)\s+([\d.E+-]+)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                improper_id, k, chi0 = match.groups()
                result.impropers[improper_id] = HarmonicImproperParams(
                    k=float(k),
                    chi0=float(chi0),
                    model=result.model_name
                )


class GromosForceFieldLoader:
    """
    Chargeur pour intégrer les paramètres GROMOS dans ForceFieldDatabase.
    """
    
    def __init__(self, db: 'ForceFieldDatabase'):
        self.db = db
        self.parser = GromosLTParser()
    
    def load_from_file(self, filepath: str, model_name: str = None) -> None:
        """Charge un fichier LT GROMOS dans la base de données."""
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse_file(filepath)
        result.to_database(self.db)
    
    def load_from_string(self, content: str, model_name: str = None) -> None:
        """Charge une chaîne LT GROMOS dans la base de données."""
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse(content)
        result.to_database(self.db)