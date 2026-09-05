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

from pathlib import Path

#: The structure library shipped with the package: the tobermorite and
#: ettringite models, and the small molecules Packmol packs by name
#: (h2o.lt, ho.sdf, co3.pdb, so4.pdb).
STRUCTURES_DIR = Path(__file__).resolve().parent / "build" / "_structures"
