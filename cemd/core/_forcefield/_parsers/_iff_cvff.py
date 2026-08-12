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

from ..forcefield_database import ForceFieldDatabase
from ..models import (
    AtomType,
    Class2BondAngleParams,
    Class2BondBondParams,
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicImproperParams,
    LJParams,
)
from ._base import BaseForceFieldParser, ParseResult


class CVFFInterfaceParser(BaseForceFieldParser):
    """
    Parser for the CVFF Interface file (cvff_interface_v1_5.txt).
    
    This parser extracts parameters from sections:
    -#atom_types: Definitions of atom types
    -#quadratic_bond: Harmonic bonds
    -#quadratic_angle: Harmonic angles
    -#torsion_1: Dihedrals
    -#out_of_plane: Improper
    -#nonbond(12-6): LJ parameters
    -#bond-bond: Class2 bond-bond terms
    -#bond-angle: Class2 terms bond-angle
    """

    def __init__(self, model_name: str = "CVFF_INTERFACE"):
        super().__init__()
        self.model_name = model_name
        self._current_section = None
        self._line_number = 0
        
        # Patterns section for detection
        self._section_patterns = {
            "atom_types": ["#atom_types"],
            "quadratic_bond": ["#quadratic_bond"],
            "quadratic_angle": ["#quadratic_angle"],
            "torsion_1": ["#torsion_1"],
            "out_of_plane": ["#out_of_plane"],
            "nonbond": ["#nonbond(12-6)"],
            "bond-bond": ["#bond-bond"],
            "bond-angle": ["#bond-angle"],
            "angle-angle-torsion": ["#angle-angle-torsion_1"],
            "angle-angle": ["#angle-angle_1"],
            "morse_bond": ["#morse_bond"],
            "improper": ["#improper"],
            "oop-oop": ["#out_of_plane-out_of_plane"],
        }

    def parse(self, content: str) -> ParseResult:
        """Parses a CVFF Interface string."""
        lines = self._clean_content(content)

        result = ParseResult(model_name=self.model_name)
        result.metadata = {
            "description": "CVFF Interface force field for clay minerals, silicates, and cements",
            "ref": "CVFF, Heinz et al. (2013, 2014)",
            "tags": ["CVFF", "Interface", "Clay", "Silicate", "Cement"],
        }

        self._parse_sections(lines, result)
        return result

    def parse_file(self, filepath: str) -> ParseResult:
        """Parse a CVFF Interface file."""
        with open(filepath) as f:
            content = f.read()
        return self.parse(content)

    def _strip_comments(self, line: str) -> str:
        """Remove comments! from CVFF format."""
        if "!" in line:
            return line[: line.index("!")]
        return line

    def _detect_section(self, line: str) -> str | None:
        """Detect the section name from the line."""
        line_lower = line.lower()
        for section_name, patterns in self._section_patterns.items():
            for pattern in patterns:
                if pattern.lower() in line_lower:
                    return section_name
        return None

    def _parse_sections(self, lines: list[str], result: ParseResult) -> None:
        """Parses the different sections of the CVFF file."""
        i = 0
        self._current_section = None

        while i < len(lines):
            line = lines[i]
            self._line_number = i + 1

            # Detect a section change
            if line.startswith("#"):
                section = self._detect_section(line)
                if section:
                    self._current_section = section
                else:
                    self._current_section = None
                i += 1
                continue

            # Ignore comments and empty lines
            if not line or line.startswith("!") or line.startswith(">"):
                i += 1
                continue

            # Parse the line if you are in an active section
            if self._current_section is not None and line.strip():
                self._parse_line(line, self._current_section, result)

            i += 1

    def _parse_line(self, line: str, section: str, result: ParseResult) -> None:
        """Parse a line according to the section."""
        # Handlers are defined as private methods
        handlers = {
            "atom_types": self._parse_atom_type,
            "quadratic_bond": self._parse_bond,
            "quadratic_angle": self._parse_angle,
            "torsion_1": self._parse_torsion,
            "out_of_plane": self._parse_improper,
            "improper": self._parse_improper,
            "nonbond": self._parse_nonbond,
            "bond-bond": self._parse_bondbond,
            "bond-angle": self._parse_bondangle,
            "morse_bond": self._parse_morse_bond,
        }
        
        handler = handlers.get(section)
        if handler:
            handler(line, result)

    def _parse_atom_type(self, line: str, result: ParseResult) -> None:
        """
        Parse a line of atom type.
        Format: Ver Ref Type Mass Element Connections Comment
        """
        parts = line.split()
        if len(parts) >= 6:
            atom_type = parts[2]
            try:
                mass = float(parts[3])
            except ValueError:
                mass = 0.0
            element = parts[4]
            
            comment = " ".join(parts[6:]) if len(parts) > 6 else ""
            charge = 0.0
            
            # Extract load from comment
            charge_match = re.search(r"\(([+-]?\d+\.?\d*)\)", comment)
            if charge_match:
                try:
                    charge = float(charge_match.group(1))
                except ValueError:
                    pass

            result.atoms[atom_type] = AtomType(
                element=element,
                charge=charge,
                mass=mass,
                environment=comment,
                model=result.model_name,
            )

    def _parse_bond(self, line: str, result: ParseResult) -> None:
        """
        Parses a harmonic bond line.
        Format: Ver Ref I J R0 K
        """
        parts = line.split()
        if len(parts) >= 6:
            i = parts[2]
            j = parts[3]
            try:
                r0 = float(parts[4])
                k = float(parts[5])
                key = f"{i}-{j}" if i <= j else f"{j}-{i}"
                result.bonds[key] = HarmonicBondParams(
                    k=k, r0=r0, model=result.model_name
                )
            except ValueError:
                pass

    def _parse_angle(self, line: str, result: ParseResult) -> None:
        """
        Parse a harmonic angle line.
        Format: Ver Ref I J K Theta0 K2
        """
        parts = line.split()
        if len(parts) >= 7:
            i = parts[2]
            j = parts[3]
            k = parts[4]
            try:
                theta0 = float(parts[5])
                k2 = float(parts[6])
                key = f"{i}-{j}-{k}"
                result.angles[key] = HarmonicAngleParams(
                    k=k2, theta0=theta0, model=result.model_name
                )
            except ValueError:
                pass

    def _parse_torsion(self, line: str, result: ParseResult) -> None:
        """
        Parse a twist line.
        Format: Ver Ref I J K L Kphi n Phi0
        """
        parts = line.split()
        if len(parts) >= 9:
            i = parts[2]
            j = parts[3]
            k = parts[4]
            l_ = parts[5]
            try:
                kphi = float(parts[6])
                n = int(float(parts[7]))
                phi0 = float(parts[8])
                key = f"{i}-{j}-{k}-{l_}"
                result.dihedrals[key] = {"k": kphi, "n": n, "d": phi0}
            except ValueError:
                pass

    def _parse_improper(self, line: str, result: ParseResult) -> None:
        """
        Parse an improper line.
        Format: Ver Ref I J K L Kchi n Chi0
        """
        parts = line.split()
        if len(parts) >= 9:
            i = parts[2]
            j = parts[3]
            k = parts[4]
            l_ = parts[5]
            try:
                kchi = float(parts[6])
                chi0 = float(parts[8]) if len(parts) > 8 else 0.0
                key = f"{i}-{j}-{k}-{l_}"
                result.impropers[key] = HarmonicImproperParams(
                    k=kchi, chi0=chi0, model=result.model_name
                )
            except ValueError:
                pass

    def _parse_nonbond(self, line: str, result: ParseResult) -> None:
        """
        Parses a line of LJ parameters.
        Format: Ver Ref I A B
        """
        parts = line.split()
        if len(parts) >= 5:
            atom = parts[2]
            try:
                a = float(parts[3])
                b = float(parts[4])
                if a > 0 and b > 0:
                    epsilon = b**2 / (4 * a)
                    sigma = (a / b) ** (1 / 6)
                    key = f"{atom}-{atom}"
                    result.lj[key] = LJParams(
                        epsilon=epsilon, sigma=sigma, model=result.model_name
                    )
            except ValueError:
                pass

    def _parse_bondbond(self, line: str, result: ParseResult) -> None:
        """
        Parses a bond-bond line (class2).
        Format: Ver Ref I J K M
        """
        parts = line.split()
        if len(parts) >= 6:
            i = parts[2]
            j = parts[3]
            k = parts[4]
            try:
                m = float(parts[5])
                key = f"{i}-{j}-{k}"
                result.bondbond[key] = Class2BondBondParams(
                    m=m, r1=0.0, r2=0.0, model=result.model_name
                )
            except ValueError:
                pass

    def _parse_bondangle(self, line: str, result: ParseResult) -> None:
        """
        Parse a bond-angle line (class2).
        Format: Ver Ref I J K N1 N2
        """
        parts = line.split()
        if len(parts) >= 7:
            i = parts[2]
            j = parts[3]
            k = parts[4]
            try:
                n1 = float(parts[5])
                n2 = float(parts[6]) if len(parts) > 6 else 0.0
                key = f"{i}-{j}-{k}"
                result.bondangle[key] = Class2BondAngleParams(
                    n1=n1, n2=n2, r1=0.0, r2=0.0, model=result.model_name
                )
            except ValueError:
                pass

    def _parse_morse_bond(self, line: str, result: ParseResult) -> None:
        """
        Parse a line of Morse bond.
        Format: Ver Ref I J R0 D ALPHA
        """
        parts = line.split()
        if len(parts) >= 7:
            i = parts[2]
            j = parts[3]
            try:
                r0 = float(parts[4])
                d = float(parts[5])
                alpha = float(parts[6])
                key = f"{i}-{j}"
                if key not in result.bonds:
                    k = 2 * d * alpha * alpha
                    result.bonds[key] = HarmonicBondParams(
                        k=k, r0=r0, model=result.model_name
                    )
            except ValueError:
                pass


class CVFFInterfaceLoader:
    """Loader to integrate CVFF Interface parameters into ForceFieldDatabase."""

    def __init__(self, db: "ForceFieldDatabase"):
        self.db = db
        self.parser = CVFFInterfaceParser()

    def load_from_file(self, filepath: str, model_name: str = None) -> None:
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse_file(filepath)
        result.to_database(self.db)

    def load_from_string(self, content: str, model_name: str = None) -> None:
        if model_name:
            self.parser.model_name = model_name
        result = self.parser.parse(content)
        result.to_database(self.db)