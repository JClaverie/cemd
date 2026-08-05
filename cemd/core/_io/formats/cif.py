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

import warnings

from pymatgen.io.cif import CifParser
from .base import BaseReader

class CifReader(BaseReader):
    """Reader for CIF files using Pymatgen."""

    @classmethod
    def read(cls, path: str, primitive=False, refine=False) -> dict:
        """
        Read CIF file and return topology.
        
        Parameters
        ----------
        path : str
            Path to the CIF file.
        primitive : bool, default=True
            If True, return the primitive cell.
            If False, return the conventional cell.
        refine : bool, default=True
            If True, refine the structure using SpacegroupAnalyzer.
        """
        from .pmg import PmgReader

        parser = CifParser(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = parser.parse_structures(primitive=primitive)[0]

        topology = PmgReader.read(structure, refine)

        return topology