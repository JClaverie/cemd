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
Base classes for format readers and writers.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReader(ABC):
    """Base class for format readers."""

    @classmethod
    @abstractmethod
    def read(cls, source: Any) -> dict:
        """Read from source and return topology dictionary."""
        pass


class BaseWriter(ABC):
    """Base class for format writers."""

    @classmethod
    @abstractmethod
    def write(cls, system, path: str, **kwargs) -> None:
        """Write system to file."""
        pass