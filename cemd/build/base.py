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

import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from ..core.atomic_system import AtomicSystem


@lru_cache
def require_program(name) -> str:
    """Return the path of an external executable.

    Raises
    ------
    RuntimeError
        If the executable is not found in PATH.
    """
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found")
    return path


class BaseBuilder(ABC):
    """Base class for all builders."""

    def __init__(self, system: AtomicSystem | None = None):
        self.system = system
        self._validate()

    def _validate(self) -> None:
        """Validate the system before building."""
        if self.system is not None:
            if not isinstance(self.system, AtomicSystem):
                raise TypeError(
                    f"Expected AtomicSystem, got {type(self.system).__name__}"
                )

    def _repr_info(self) -> dict:
        """Return info dict for __repr__."""
        info = {
            "class": self.__class__.__name__,
            "has_system": self.system is not None,
        }
        if self.system is not None:
            info["system"] = {
                "atoms": self.system.num_atoms,
                "box": self.system.box,
            }
        return info

    def __repr__(self) -> str:
        """Generic __repr__ for builders."""
        info = self._repr_info()
        parts = [f"<{info['class']}"]

        if info["has_system"]:
            parts.append(f"system={info['system']['atoms']} atoms")
            parts.append(f"box={info['system']['box']}")
        else:
            parts.append("no system")

        parts.append(">")
        return " ".join(parts)

    @abstractmethod
    def build(self, **kwargs) -> Any:
        """Build the structure."""
        pass
