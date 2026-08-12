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

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING

from .config import (
    VMD_BOND_CUTOFFS,
    VMD_ELEMENT_COLORS,
    VMD_ELEMENT_RADII,
    VMD_MATERIAL_OPTIONS,
    VMD_MATERIAL_SETTINGS,
)

if TYPE_CHECKING:
    from ..atomic_system import AtomicSystem


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


# ============================================================================
# Helpers: Color Conversion
# ============================================================================


def _get_view_script_path() -> str:
    """Returns the path to view.tcl (same folder as this file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "view.tcl")


def _hex_to_vmd_rgb(hex_color: str) -> tuple:
    """Convert #RRGGBB to VMD RGB values (0-1)."""
    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# ============================================================================
# TCL Generators
# ============================================================================


class TCLGenerator:
    """Generate TCL configuration for VMD."""

    @staticmethod
    def element_map(elements: dict[str | int, str] | list | set) -> str:
        """
        Generate element map TCL code.

        Groups all atom types belonging to the same element symbol.

        Parameters
        ----------
        elements : dict or list or set
            Dictionary mapping atom types to element symbols (system.elements),
            or a list/set of unique elements.

        Returns
        -------
        str
            TCL script defining the array 'element_map'.
        """
        lines = ["array set element_map {"]

        # if isinstance(elements, dict):
        elem_to_types: dict[str, list[str]] = {}
        for atom_type, elem_symbol in elements.items():
            elem_to_types.setdefault(str(elem_symbol), []).append(str(atom_type))

        for elem_symbol, type_list in sorted(elem_to_types.items()):
            types_str = " ".join(type_list)
            lines.append(f'    {elem_symbol}  "{types_str}"')

        # else:
        #     # Fallback si une simple liste/ensemble d'éléments est fournie
        #     for element in sorted(set(elements)):
        #         types = VMD_ELEMENT_TYPES.get(element, [element])
        #         types_str = " ".join(types) if isinstance(types, list) else str(types)
        #         lines.append(f'    {element}  "{types_str}"')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def element_colors(elements: dict | list | set) -> str:
        """Generate element colors TCL code."""
        lines = []
        color_id = 20

        # Extraction des symboles d'éléments uniques
        if isinstance(elements, dict):
            unique_elements = sorted(set(elements.values()))
        else:
            unique_elements = sorted(set(elements))

        for element in unique_elements:
            if element in VMD_ELEMENT_COLORS:
                color = VMD_ELEMENT_COLORS[element]
                r, g, b = _hex_to_vmd_rgb(color)
                lines.append(f"color change rgb {color_id} {r:.3f} {g:.3f} {b:.3f}")
                lines.append(f"color Element {element} {color_id}")
                color_id += 1

        return "\n".join(lines)

    @staticmethod
    def material(material: str) -> str:
        """Generate material TCL code."""
        lines = [f"mol material {material}"]
        lines.extend(
            f"material change {key} {material} {value}"
            for key, value in VMD_MATERIAL_SETTINGS.items()
        )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def representations(resolution: int, atomic_system: AtomicSystem) -> str:
        """Generate representations TCL code."""
        lines = []

        element_to_types = {}

        atom_types = atomic_system.atom_types
        elements = atomic_system.elements

        for atom_type, element in elements.items():
            element_to_types.setdefault(element, []).append(atom_type)

        # Atom representations
        for element, types in element_to_types.items():
            filtered = [t for t in types if atom_types is None or t in atom_types]
            if not filtered:
                continue

            radius = VMD_ELEMENT_RADII.get(element, 0.8)
            selection = " ".join(dict.fromkeys(filtered))

            lines.extend(
                [
                    f'mol selection "type {selection}"',
                    f"mol representation CPK {radius} {radius} {resolution} {resolution}",
                    "mol addrep $system",
                    "",
                ]
            )

        # Bond representations
        for pair, cutoff in VMD_BOND_CUTOFFS.items():
            if "-" not in pair:
                continue

            elem1, elem2 = pair.split("-")

            print(elem1, elem2)

            types_elem1 = element_to_types.get(elem1, [])
            types_elem2 = element_to_types.get(elem2, [])

            print(types_elem1, types_elem2)

            if not types_elem1 or not types_elem2:
                continue

            if elem1 == elem2:
                selected_types = sorted(set(types_elem1))
            else:
                selected_types = sorted(set(types_elem1 + types_elem2))

            if atom_types is not None:
                selected_types = [t for t in selected_types if t in atom_types]

            if not selected_types:
                continue

            selection = " ".join(selected_types)
            lines.extend(
                [
                    f'mol selection "type {selection}"',
                    f"mol representation DynamicBonds {cutoff} 0.2 {resolution}",
                    "mol addrep $system",
                    "",
                ]
            )

        return "\n".join(lines)

    @classmethod
    def generate_config(cls, system) -> str:
        """Generate full configuration TCL."""
        return "\n".join(
            [
                cls.element_map(system.elements),
                "",
                cls.element_colors(system.elements),
            ]
        )

    @classmethod
    def generate_representations(
        cls, material: str, resolution: int, atomic_system: AtomicSystem
    ) -> str:
        """Generate full representations TCL."""
        if material not in VMD_MATERIAL_OPTIONS:
            raise ValueError(
                f"Unknown VMD material '{material}'. Available: {VMD_MATERIAL_OPTIONS}"
            )
        print(
            "\n".join(
                [
                    cls.material(material),
                    cls.representations(resolution, atomic_system),
                ]
            )
        )
        return "\n".join(
            [
                cls.material(material),
                cls.representations(resolution, atomic_system),
            ]
        )


# ============================================================================
# File Management
# ============================================================================


class TempFileManager:
    """Manage temporary files for VMD."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.temp_dir = None
        self.files = {}

    def __enter__(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.base_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_dir.cleanup()

    def create_file(self, name: str, content: str) -> str:
        """Create a temporary file with content."""
        if not self.temp_dir:
            raise RuntimeError("TempFileManager not initialized")

        filepath = os.path.join(self.temp_dir.name, name)
        with open(filepath, "w") as f:
            f.write(content)
        self.files[name] = filepath
        return filepath

    def get_path(self, name: str) -> str:
        """Get path of a created file."""
        return self.files.get(name)


# ============================================================================
# VMD Launcher
# ============================================================================


class VMDLauncher:
    """Launch VMD with configuration."""

    def __init__(self):
        self.view_script = _get_view_script_path()
        require_program("vmd")

    def launch(
        self,
        topology: str,
        trajectory: str | None = None,
        config_file: str | None = None,
        rep_file: str | None = None,
    ) -> None:
        """Launch VMD with given files."""
        # Validate files
        if not os.path.exists(topology):
            raise FileNotFoundError(f"Topology file not found: {topology}")

        if trajectory and not os.path.exists(trajectory):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory}")

        # Build command
        cmd = ["vmd", "-e", self.view_script, "-args", topology]

        if config_file:
            cmd.extend(["--config", config_file])

        if rep_file:
            cmd.extend(["--rep", rep_file])

        if trajectory:
            cmd.append(trajectory)

        # Execute
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"VMD terminated with error: {e}") from e


# ============================================================================
# Main API
# ============================================================================


def view(
    system,
    trajectory: str | None = None,
    material: str = "AOEdgy",
    resolution: int = 12,
) -> None:
    """
    Open VMD to visualize a molecular system.

    Args:
        system: AtomicSystem
        trajectory: Optional trajectory file path
        material: VMD material name (default: "AOEdgy")
        resolution: CPK resolution, higher = smoother (default: 12)

    Raises:
        TypeError: If target is not supported
        FileNotFoundError: If required files are missing
        RuntimeError: If VMD fails
        ValueError: If material is not valid
    """

    if not hasattr(system, "write"):
        raise TypeError("Target must be a molecular object with 'write' method.")

    with TempFileManager() as tmp:
        topology_file = tmp.create_file("tmp.data", "")
        system.write(topology_file)

        config_file = tmp.create_file(
            "vmd_config.tcl", TCLGenerator.generate_config(system)
        )
        rep_file = tmp.create_file(
            "vmd_rep.tcl",
            TCLGenerator.generate_representations(material, resolution, system),
        )

        VMDLauncher().launch(topology_file, trajectory, config_file, rep_file)
