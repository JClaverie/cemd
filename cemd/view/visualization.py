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
import subprocess
import tempfile
from typing import Optional
from pathlib import Path

from .config import (
    VMD_ELEMENT_TYPES,
    VMD_ELEMENT_COLORS,
    VMD_ATOM_RADIUS,
    VMD_BOND_CUTOFF,
    VMD_MATERIAL_OPTIONS,
    VMD_MATERIAL_SETTINGS,
)

from .._utils import require_program
from .._paths import VMD_VIEW_PATH


# ============================================================================
# Helpers: Color Conversion
# ============================================================================

def _hex_to_vmd_rgb(hex_color: str) -> tuple:
    """Convert #RRGGBB to VMD RGB values (0-1)."""
    hex_color = hex_color.lstrip("#")
    
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    
    return tuple(
        int(hex_color[i:i+2], 16) / 255.0
        for i in (0, 2, 4)
    )


# ============================================================================
# TCL Generators
# ============================================================================

class TCLGenerator:
    """Generate TCL configuration for VMD."""
    
    @staticmethod
    def element_map(element_types: dict) -> str:
        """Generate element map TCL code."""
        lines = ["array set element_map {"]
        for element, types in element_types.items():
            lines.append(f'    {element}  "{" ".join(types)}"')
        lines.append("}")
        return "\n".join(lines)
    
    @staticmethod
    def element_colors(element_colors: dict) -> str:
        """Generate element colors TCL code."""
        lines = []
        color_id = 20
        
        for element, color in element_colors.items():
            r, g, b = _hex_to_vmd_rgb(color)
            lines.append(f"color change rgb {color_id} {r:.3f} {g:.3f} {b:.3f}")
            lines.append(f"color Element {element} {color_id}")
            color_id += 1
        
        return "\n".join(lines)
    
    @staticmethod
    def material(material: str, settings: dict) -> str:
        """Generate material TCL code."""
        lines = [f"mol material {material}"]
        for key, value in settings.items():
            lines.append(f"material change {key} {material} {value}")
        return "\n".join(lines)
    
    @staticmethod
    def representations(atom_radius: dict, bond_cutoff: dict, resolution: int) -> str:
        """Generate representations TCL code."""
        lines = []
        
        # Atom representations
        for selection, radius in atom_radius.items():
            lines.extend([
                f'mol selection "type {selection}"',
                f"mol representation CPK {radius} {radius} {resolution} {resolution}",
                "mol addrep $system",
                ""
            ])
        
        # Bond representations
        for selection, cutoff in bond_cutoff.items():
            lines.extend([
                f'mol selection "type {selection}"',
                f"mol representation DynamicBonds {cutoff} 0.2 {resolution}",
                "mol addrep $system",
                ""
            ])
        
        return "\n".join(lines)
    
    @classmethod
    def generate_config(cls) -> str:
        """Generate full configuration TCL."""
        parts = [
            cls.element_map(VMD_ELEMENT_TYPES),
            "",
            cls.element_colors(VMD_ELEMENT_COLORS),
        ]
        return "\n".join(parts)
    
    @classmethod
    def generate_representations(cls, material: str, resolution: int) -> str:
        """Generate full representations TCL."""
        if material not in VMD_MATERIAL_OPTIONS:
            raise ValueError(
                f"Unknown VMD material '{material}'. "
                f"Available materials: {VMD_MATERIAL_OPTIONS}"
            )
        
        parts = [
            cls.material(material, VMD_MATERIAL_SETTINGS),
            cls.representations(VMD_ATOM_RADIUS, VMD_BOND_CUTOFF, resolution),
        ]
        return "\n".join(parts)


# ============================================================================
# File Management
# ============================================================================

class TempFileManager:
    """Manage temporary files for VMD."""
    
    def __init__(self, base_dir: str = '.'):
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
    
    def __init__(self, view_script: str = VMD_VIEW_PATH):
        self.view_script = view_script
        require_program('vmd')
    
    def launch(
        self,
        topology: str,
        trajectory: Optional[str] = None,
        config_file: Optional[str] = None,
        rep_file: Optional[str] = None,
    ) -> None:
        """Launch VMD with given files."""
        # Validate files
        if not os.path.exists(topology):
            raise FileNotFoundError(f"Topology file not found: {topology}")
        
        if trajectory and not os.path.exists(trajectory):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory}")
        
        # Build command
        cmd = ['vmd', '-e', self.view_script, '-args', topology]
        
        if config_file:
            cmd.extend(['--config', config_file])
        
        if rep_file:
            cmd.extend(['--rep', rep_file])
        
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
    target, 
    trajectory: Optional[str] = None,
    material: str = "AOEdgy",
    resolution: int = 12
) -> None:
    """
    Open VMD to visualize a molecular system.
    
    Args:
        target: Molecular object with 'write()' method, or path to topology file
        trajectory: Optional trajectory file path
        material: VMD material name (default: "AOEdgy")
        resolution: CPK resolution, higher = smoother (default: 12)
    
    Raises:
        TypeError: If target is not supported
        FileNotFoundError: If required files are missing
        RuntimeError: If VMD fails
        ValueError: If material is not valid
    """
    # Handle file path
    if isinstance(target, (str, Path)):
        with TempFileManager() as tmp:
            # Generate configs with custom parameters
            config_file = tmp.create_file(
                "vmd_config.tcl", 
                TCLGenerator.generate_config()
            )
            rep_file = tmp.create_file(
                "vmd_rep.tcl", 
                TCLGenerator.generate_representations(material=material, resolution=resolution)
            )
            
            # Launch VMD
            launcher = VMDLauncher()
            launcher.launch(str(target), trajectory, config_file, rep_file)
        return
    
    # Handle object with write method
    if hasattr(target, "write"):
        with TempFileManager() as tmp:
            # Write topology
            topology_file = tmp.create_file("tmp.data", "")
            target.write(topology_file)
            
            # Generate configs
            config_file = tmp.create_file(
                "vmd_config.tcl", 
                TCLGenerator.generate_config()
            )
            rep_file = tmp.create_file(
                "vmd_rep.tcl", 
                TCLGenerator.generate_representations(material=material, resolution=resolution)
            )
            
            # Launch VMD
            launcher = VMDLauncher()
            launcher.launch(topology_file, trajectory, config_file, rep_file)
        return
    
    raise TypeError(
        f"Unsupported target type: {type(target)}. "
        "Must be a molecular object with 'write' method or a file path."
    )


# ============================================================================
# Convenience Functions
# ============================================================================

def view_with_material(
    target, 
    material: str = "AOEdgy", 
    trajectory: Optional[str] = None,
    resolution: int = 12
) -> None:
    """
    Visualize with a specific material.
    
    Args:
        target: Molecular object or file path
        material: Name of VMD material (default: "AOEdgy")
        trajectory: Optional trajectory file
        resolution: CPK resolution (default: 12)
    """
    view(target, trajectory=trajectory, material=material, resolution=resolution)


def view_high_res(
    target, 
    trajectory: Optional[str] = None,
    resolution: int = 24
) -> None:
    """
    Visualize with higher resolution.
    
    Args:
        target: Molecular object or file path
        trajectory: Optional trajectory file
        resolution: CPK resolution (default: 24)
    """
    view(target, trajectory=trajectory, resolution=resolution)