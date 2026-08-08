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
import subprocess
import tempfile

from ..._utils import require_program
from .config import (
    VMD_BOND_CUTOFFS,
    VMD_ELEMENT_COLORS,
    VMD_ELEMENT_RADII,
    VMD_ELEMENT_TYPES,
    VMD_MATERIAL_OPTIONS,
    VMD_MATERIAL_SETTINGS,
)

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
    def element_map(elements: list) -> str:
        """Generate element map TCL code."""
        lines = ["array set element_map {"]
        for element in elements:
            types = VMD_ELEMENT_TYPES.get(element, element)
            lines.append(f'    {element}  "{" ".join(types)}"')
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def element_colors(elements: list) -> str:
        """Generate element colors TCL code."""
        lines = []
        color_id = 20

        for element in elements:
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
    def representations(resolution: int, atom_types: set[str] = None) -> str:
        """Generate representations TCL code."""
        lines = []

        element_to_types = {}

        if atom_types is None:
            # Si pas d'atom_types, utiliser VMD_ELEMENT_TYPES
            for element, types in VMD_ELEMENT_TYPES.items():
                element_to_types[element] = types
        else:
            # Grouper les types par élément
            for atom_type in atom_types:
                # Essayer de trouver l'élément correspondant
                element = None
                for elem, types in VMD_ELEMENT_TYPES.items():
                    if atom_type in types:
                        element = elem
                        break

                # Si pas trouvé, utiliser le type comme élément
                if element is None:
                    element = atom_type

                if element not in element_to_types:
                    element_to_types[element] = []
                element_to_types[element].append(atom_type)

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
            types = pair.split("-")
            types_sorted = sorted(types)

            # Filtrer uniquement par atom_types
            if atom_types is not None:
                if not all(t in atom_types for t in types_sorted):
                    continue

            selection = " ".join(types_sorted)
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
        cls, material: str, resolution: int, atom_types: set[str] = None
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
                    cls.representations(resolution, atom_types),
                ]
            )
        )
        return "\n".join(
            [
                cls.material(material),
                cls.representations(resolution, atom_types),
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

        atom_types = set(system.atom_types)

        config_file = tmp.create_file(
            "vmd_config.tcl", TCLGenerator.generate_config(system)
        )
        rep_file = tmp.create_file(
            "vmd_rep.tcl",
            TCLGenerator.generate_representations(material, resolution, atom_types),
        )

        VMDLauncher().launch(topology_file, trajectory, config_file, rep_file)
