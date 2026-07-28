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
CIF file reader.
"""

from pymatgen.io.cif import CifParser
from .base import BaseReader


class CifReader(BaseReader):
    """Read CIF files using Pymatgen."""

    @classmethod
    def read(cls, path: str) -> dict:
        """Read CIF file and return topology."""
        from .pmg import PmgReader

        parser = CifParser(path)
        structure = parser.parse_structures()[0]

        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure)
        refined = analyzer.get_refined_structure()

        topology = PmgReader.read(refined)
        topology['_pmg_struct'] = structure
        return topology