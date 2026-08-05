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

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FF_DATABASE_DIR = os.path.join(BASE_DIR, "_data", "forcefields")
STRUCTURES_DIR = os.path.join(BASE_DIR, "_data", "structures")
PYCSH_DIR = os.path.join(BASE_DIR, "build", "pyCSH-main")