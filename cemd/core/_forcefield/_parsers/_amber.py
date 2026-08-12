"""
Parser pour les fichiers AMBER (format frcmod/parm7).
Copyright (c) 2022-2026 Jérôme Claverie.
"""

import re

from ..forcefield_database import ForceFieldDatabase
from ..models import (
    AtomType,
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicImproperParams,
    LJParams,
)
from ._base import BaseForceFieldParser, ParseResult


class AmberParser(BaseForceFieldParser):
    """
    Parser pour les fichiers AMBER (frcmod, parm7).
    
    Format frcmod exemple:
    MASS
       BB  14.007    ! N
    BOND
       C -N      265.0  1.335
    ANGLE
       C -N -C    50.0  120.0
    DIHE
       C -N -C -O 1.0 180.0 2.0
    IMPROPER
       C -N -C -O 50.0 180.0 2.0
    NONBON
       CA 1.5000 0.1000
    """
    
    def __init__(self, model_name: str = "AMBER"):
        super().__init__()
        self.model_name = model_name
        
    def parse(self, content: str) -> ParseResult:
        """Parse une chaîne AMBER."""
        lines = self._clean_content(content)
        
        result = ParseResult(model_name=self.model_name)
        result.metadata = {
            'description': 'AMBER force field',
            'ref': 'Cornell et al. (1995), Wang et al. (2004)',
            'tags': ['AMBER', 'frcmod', 'parm7']
        }
        
        self._parse_sections(lines, result)
        return result
    
    def parse_file(self, filepath: str) -> ParseResult:
        """Parse un fichier AMBER."""
        with open(filepath) as f:
            content = f.read()
        return self.parse(content)
    
    def _strip_comments(self, line: str) -> str:
        """Enlève les commentaires ! du format AMBER."""
        if '!' in line:
            return line[:line.index('!')]
        return line
    
    def _parse_sections(self, lines: list[str], result: ParseResult) -> None:
        """Parse les différentes sections du fichier AMBER."""
        i = 0
        current_section = None
        
        while i < len(lines):
            line = lines[i]
            
            # Identifier les sections
            if line == 'MASS':
                current_section = 'MASS'
            elif line == 'BOND':
                current_section = 'BOND'
            elif line == 'ANGLE':
                current_section = 'ANGLE'
            elif line == 'DIHE' or line == 'DIHEDRAL':
                current_section = 'DIHEDRAL'
            elif line == 'IMPROPER':
                current_section = 'IMPROPER'
            elif line == 'NONBON':
                current_section = 'NONBON'
            elif line.startswith('ATOM'):
                current_section = 'ATOM'
            elif line == 'LJ':
                current_section = 'LJ'
            elif current_section is not None and not line.startswith(('MASS', 'BOND', 'ANGLE', 'DIHE', 'IMPROPER', 'NONBON', 'ATOM', 'LJ')):
                # Traiter la ligne selon la section
                self._parse_line(line, current_section, result)
            
            i += 1
    
    def _parse_line(self, line: str, section: str, result: ParseResult) -> None:
        """Parse une ligne selon la section."""
        if section == 'MASS':
            self._parse_mass_line(line, result)
        elif section == 'BOND':
            self._parse_bond_line(line, result)
        elif section == 'ANGLE':
            self._parse_angle_line(line, result)
        elif section == 'DIHEDRAL':
            self._parse_dihedral_line(line, result)
        elif section == 'IMPROPER':
            self._parse_improper_line(line, result)
        elif section == 'NONBON' or section == 'LJ':
            self._parse_nonbond_line(line, result)
        elif section == 'ATOM':
            self._parse_atom_line(line, result)
    
    def _parse_mass_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne de masse."""
        pattern = r'(\w+)\s+([\d.]+)'
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
    
    def _parse_bond_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne de liaison."""
        pattern = r'(\w+)\s*-\s*(\w+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, line)
        if match:
            type1, type2, k, r0 = match.groups()
            key = f"{type1}-{type2}" if type1 <= type2 else f"{type2}-{type1}"
            result.bonds[key] = HarmonicBondParams(
                k=float(k),
                r0=float(r0),
                model=result.model_name
            )
    
    def _parse_angle_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne d'angle."""
        pattern = r'(\w+)\s*-\s*(\w+)\s*-\s*(\w+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, line)
        if match:
            type1, type2, type3, k, theta0 = match.groups()
            key = f"{type1}-{type2}-{type3}"
            result.angles[key] = HarmonicAngleParams(
                k=float(k),
                theta0=float(theta0),
                model=result.model_name
            )
    
    def _parse_dihedral_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne de dièdre."""
        pattern = r'(\w+)\s*-\s*(\w+)\s*-\s*(\w+)\s*-\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, line)
        if match:
            type1, type2, type3, type4, k, d, n = match.groups()
            key = f"{type1}-{type2}-{type3}-{type4}"
            result.dihedrals[key] = {
                'k': float(k),
                'n': int(float(n)),
                'd': float(d)
            }
    
    def _parse_improper_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne d'improper."""
        pattern = r'(\w+)\s*-\s*(\w+)\s*-\s*(\w+)\s*-\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, line)
        if match:
            type1, type2, type3, type4, k, chi0, *_ = match.groups()
            key = f"{type1}-{type2}-{type3}-{type4}"
            result.impropers[key] = HarmonicImproperParams(
                k=float(k),
                chi0=float(chi0),
                model=result.model_name
            )
    
    def _parse_nonbond_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne de paramètres non-liés."""
        pattern = r'(\w+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, line)
        if match:
            atom_type, rmin, eps = match.groups()
            # AMBER utilise rmin/2^1/6 comme sigma
            result.lj[f"{atom_type}-{atom_type}"] = LJParams(
                epsilon=float(eps),
                sigma=float(rmin) / 1.122462048,  # 2^(1/6)
                model=result.model_name
            )
    
    def _parse_atom_line(self, line: str, result: ParseResult) -> None:
        """Parse une ligne de type d'atome."""
        pattern = r'ATOM\s+(\w+)\s+([\d.]+)\s+([\d.+-]+)'
        match = re.search(pattern, line)
        if match:
            atom_type, mass, charge = match.groups()
            result.atoms[atom_type] = AtomType(
                element=self._guess_element(atom_type),
                charge=float(charge),
                mass=float(mass),
                environment='',
                model=result.model_name
            )
    
    def _guess_element(self, atom_type: str) -> str:
        """Devine l'élément à partir du nom du type d'atome."""
        # AMBER a des types comme C, CA, CT, N, NT, O, OH, H, HC, etc.
        if len(atom_type) == 1:
            return atom_type
        # Cas spéciaux
        special = {
            'CT': 'C', 'CA': 'C', 'CM': 'C', 'C2': 'C', 'C3': 'C',
            'NT': 'N', 'N2': 'N', 'N3': 'N', 'NA': 'N', 'NB': 'N',
            'O2': 'O', 'OH': 'O',
            'HC': 'H', 'HA': 'H', 'HO': 'H', 'HP': 'H',
            'HS': 'H', 'H1': 'H', 'H2': 'H', 'H3': 'H',
            'SH': 'S', 'S': 'S',
        }
        if atom_type in special:
            return special[atom_type]
        return atom_type[0] if atom_type[0] in 'CONHPS' else 'X'


class AmberForceFieldLoader:
    """
    Chargeur pour intégrer les paramètres AMBER dans ForceFieldDatabase.
    """
    
    def __init__(self, db: 'ForceFieldDatabase'):
        self.db = db
        self.parser = AmberParser()
    
    def load_from_file(self, filepath: str, model_name: str = None) -> None:
        """Charge un fichier frcmod AMBER dans la base de données."""
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse_file(filepath)
        result.to_database(self.db)
    
    def load_from_string(self, content: str, model_name: str = None) -> None:
        """Charge une chaîne frcmod AMBER dans la base de données."""
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse(content)
        result.to_database(self.db)