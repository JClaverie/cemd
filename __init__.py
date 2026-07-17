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

import shutil
import logging

from .core.atomic_system import AtomicSystem
from .core._io import read_log
from .builders.base import (
    add_droplet,
    add_liquid,
    build_glass,
    build_solution,
    build_surface,
    merge,
    split,
)
from .builders.hydrates import make_csh, csh_to_cash
from ._utils import concentration2count

__author__ = "Jérôme Claverie"

__copyright__ = "Copyright (c) 2024-2026 Jérôme Claverie"

__license__ = "GPL-3.0" 

__all__  = ['AtomicSystem', 
            'read_log', 
            'make_csh', 
            'csh_to_cash', 
            'build_surface', 
            'build_solution', 
            'build_glass', 
            'add_liquid', 
            'split', 
            'add_droplet', 
            'merge',
            'concentration2count']

logger = logging.getLogger(__name__)
_packmol_path = shutil.which("packmol")
if _packmol_path:
    logger.warning(f"✓ Packmol is installed here: {_packmol_path}")
else:
    logger.warning("✕ Packmol was not found in the $PATH.")

_vmd_path = shutil.which("vmd")
if _vmd_path:
    logger.warning(f"✓ VMD is installed here: {_vmd_path}")
else:
    logger.warning("✕ VMD was not found in the $PATH.")