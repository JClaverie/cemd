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

from __future__ import annotations

import vtk
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from cemd.core.atomic_system import AtomicSystem
    from .main_window import AtomViewerGUI

class AtomicPlotter(QtInteractor):
    def __init__(self, 
                 parent: AtomViewerGUI=None, 
                 config: dict[str, Any]=None):
        super().__init__(parent)
        self.bg_cycle = ["black", "#1A1A1A", "white", "#B0C4DE"]
        self.bg_idx = 0
        self.color_map = {}
        self.radius_map = {}
        self.bond_map = {}
        self.global_scale = 1.0
        self.global_bond_radius = 0.1

        if config:
            self.apply_config(config)

        self.update_background_style()
        # self.render_window.SetMultiSamples(4)
        # self.enable_anti_aliasing('msaa')
        # self.enable_lightkit()
        # self.enable_eye_dome_lighting()

        self.add_axes()

    def get_current_config_dict(self) -> None:
        """Returns the current state of this plotter's settings in dictionary form."""
        return {
            "color_map": self.color_map.copy(),
            "radius_map": self.radius_map.copy(),
            "bond_map": self.bond_map.copy(),
            "global_scale": getattr(self, 'global_scale', 1.0),
            "global_bond_radius": getattr(self, 'global_bond_radius', 0.1),
            "bg_color": self.bg_cycle[self.bg_idx] if hasattr(self, 'bg_cycle') else "black"
        }
    
    def apply_config(self, config_dict:dict[str, Any]) -> None:
        """Applies a config dictionary without reading the disk."""
        self.color_map = config_dict.get("color_map", self.color_map)
        self.radius_map = config_dict.get("radius_map", self.radius_map)
        self.bond_map = config_dict.get("bond_map", self.bond_map)
        self.global_scale = config_dict.get("global_scale", self.global_scale)
        self.global_bond_radius = config_dict.get("global_bond_radius", self.global_bond_radius)
        
        new_bg = config_dict.get("bg_color", "black")
        if new_bg in self.bg_cycle:
            self.bg_idx = self.bg_cycle.index(new_bg)
        
        # force the visual refresh if necessary
        self.update_background_style()

    def hex_to_rgb(self, h: str) -> list[float]:
        """Converts colors from JSON to 3D format [0, 1]"""
        if h == "gray": return [0.5, 0.5, 0.5]
        try:
            h = h.lstrip('#')
            return [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]
        except:
            return [0.5, 0.5, 0.5]
        
    def add_axes(self) -> None:
        """Adds axes with white titles"""

        # Create axes actor
        axes_actor = vtk.vtkAxesActor()

        # Configure axes geometry
        axes_actor.SetTotalLength(3, 3, 3)
        axes_actor.SetShaftTypeToCylinder()
        axes_actor.SetCylinderRadius(0.1)
        axes_actor.SetConeRadius(1)

        # Configure labels
        for axis in [
            axes_actor.GetXAxisCaptionActor2D(),
            axes_actor.GetYAxisCaptionActor2D(),
            axes_actor.GetZAxisCaptionActor2D(),    
        ]:
            prop = axis.GetCaptionTextProperty()
            prop.SetColor(1, 1, 1)

        # Create orientation widget
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes_actor)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.3, 0.3)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

    def draw_atoms(self, df: pd.DataFrame) -> None:
        """Group atoms by type and render them as point-based spheres."""
        for actor in list(self.renderer.actors.keys()):
            if actor.startswith("atoms_") or actor.startswith("atom_single_"):
                self.remove_actor(actor)

        if df is None or df.empty:
            return

        for atype, group_df in df.groupby('type'):
            mesh = pv.PolyData(group_df[['x', 'y', 'z']].values)
            self._add_atom_mesh(mesh, atype, self.global_scale, name=f"atoms_{atype}")

    def _add_atom_mesh(self, 
                       mesh: pv.PolyData, 
                       atype: str | int, 
                       scale: float, 
                       name: str) -> None:
        """Helper to style and add an atom mesh to the renderer."""
        atype_str = str(atype).strip()

        # get the color from the object map (updated by logic.py)
        color_hex = self.color_map.get(atype_str, "gray")
        
        # make sure that the radius is there too
        base_radius = self.radius_map.get(atype_str, 1.5)
        display_size = base_radius * 10 * scale

        # sphere_source = pv.Sphere(radius=display_size, theta_resolution=16, phi_resolution=16)
        # styled_mesh = mesh.glyph(geom=sphere_source)

        if name in self.renderer.actors:
            self.remove_actor(name)

        # use color=color_hex directly. 
        # PyVista accepts names 'white', 'red' or HEX codes '#FFFFFF'
        self.add_mesh(
            mesh, 
            color=color_hex, 
            render_points_as_spheres=True,
            point_size=display_size, 
            name=name, 
            pickable=True, 
            reset_camera=False,
            lighting=True,
            style='points',
            ambient=0.5
        )
        
    def add_bonds_smart(self, 
                        xyz: np.ndarray, 
                        types: np.ndarray | list, 
                        exceptions: dict[str, Any] | None=None) -> None:
        """Calculate and render bonds between atoms based on distance rules."""
        if not exceptions or len(xyz) < 2:
            return
        
        if 'bonds' in self.renderer.actors:
            self.remove_actor('bonds')

        bond_thickness = exceptions.get("global_bond_radius", 0.1)

        str_types = [str(t).strip() for t in types]
        max_search = max(exceptions.values())
        tree = cKDTree(xyz)
        pairs = list(tree.query_pairs(max_search))

        rgb_cache = {
            t: self.hex_to_rgb(self.color_map.get(t, "gray"))
            for t in set(str_types)
        }
        
        pts, lines, rgb_colors = [], [], []

        for i, j in pairs:
            t1, t2 = str_types[i], str_types[j]
            pair_key = "-".join(sorted([t1, t2]))
            
            if pair_key in exceptions:
                limit = exceptions[pair_key]
                dist = np.linalg.norm(xyz[i] - xyz[j])
                if dist <= limit:
                    p1, p2, mid = xyz[i], xyz[j], (xyz[i] + xyz[j]) / 2.0
                    idx = len(pts)
                    pts.extend([p1, mid, mid, p2]) 
                    lines.extend([2, idx, idx+1, 2, idx+2, idx+3])
                    
                    # Here, use your rule: gray by default if unknown
                    c_i = rgb_cache[t1]
                    c_j = rgb_cache[t2]
                    rgb_colors.extend([c_i, c_i, c_j, c_j])
        if pts:
            
            b_mesh = pv.PolyData(np.array(pts))
            b_mesh.lines = np.array(lines)
            b_mesh.point_data['colors'] = (np.array(rgb_colors) * 255).astype(np.uint8)
            
            # use the 'name' parameter so that PyVista knows that it is the unique 'bonds' mesh
            self.add_mesh(b_mesh.tube(radius=bond_thickness, n_sides=8), 
                        scalars='colors', rgb=True, name='bonds', 
                        reset_camera=False, pickable=False)

    def draw_box(self, system: AtomicSystem) -> None:
        """Draws the simulation box with thickness and adaptive color"""
        if system is None or not hasattr(system, '_box_vectors'):
            return
        try:
            v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                        [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
            
            pts = (v @ np.array(system._box_vectors)) + np.array([system._lmp_box[0][0], 
                                                            system._lmp_box[1][0], 
                                                            system._lmp_box[2][0]])
            
            faces = np.hstack([[4,0,1,2,3], [4,4,5,6,7], [4,0,1,5,4], 
                            [4,1,2,6,5], [4,2,3,7,6], [4,3,0,4,7]])
            
            # Determine color based on current background
            current_bg = self.bg_cycle[self.bg_idx]
            # If the background is white, we trace the box in black, otherwise in white
            box_color = "black" if current_bg in ["white", "#b0c4de"] else "white"

            mesh = pv.PolyData(pts, faces)
            # box_tubes = mesh.extract_all_edges().tube(radius=0.03)
            
            self.add_mesh(mesh, 
                        color=box_color, 
                        style="wireframe", 
                        line_width=2,      # Thicker for visibility
                        opacity=0.5,       # A little more opaque to clearly see the edges
                        pickable=False, 
                        name='box',        # CRUCIAL NAME for update
                        reset_camera=False)
        except Exception as e:
            print(f"Erreur draw_box: {e}")

    def update_bonds_only(self, 
                          df: pd.DataFrame, 
                          bond_settings: dict[str, Any]) -> None:
        """Updates bindings only"""
        if df is None or df.empty:
            return

        xyz = df[['x', 'y', 'z']].values
        atom_types = df['type'].values
        
        self.add_bonds_smart(xyz, atom_types, bond_settings)

    def update_background_style(self) -> None:
        """Changes the background and forces the box to change color"""
        color = self.bg_cycle[self.bg_idx]
        self.set_background(color)
        
        # Definition of contrasting color
        # use RGB tuples (0,0,0) for black and (1,1,1) for white
        contrast_color = (0, 0, 0) if color == "white" else (1, 1, 1)
        
        # Axes Update
        if hasattr(self, 'axes_widget'):
            axes_actor = self.axes_widget.GetOrientationMarker()
            for ax in [axes_actor.GetXAxisCaptionActor2D(), 
                       axes_actor.GetYAxisCaptionActor2D(), 
                       axes_actor.GetZAxisCaptionActor2D()]:
                ax.GetCaptionTextProperty().SetColor(contrast_color)
        
        # REAL box color update
        if 'box' in self.renderer.actors:
            box_actor = self.renderer.actors['box']
            
            # force the color via the mapper object (deeper in VTK)
            box_actor.prop.SetColor(contrast_color) 
            
            # deactivate the scalar mode which could block the fixed color
            box_actor.mapper.scalar_visibility = False 
            
            # reset the thickness just in case
            box_actor.prop.line_width = 2
        
        self.render()

    def update_atom_sizes(self) -> None:
        """Only updates the point_size of existing atom actors."""
        for atype, actor_name in [(k, f"atoms_{k}") for k in self.radius_map]:
            if actor_name in self.renderer.actors:
                actor = self.renderer.actors[actor_name]
                base_radius = self.radius_map.get(str(atype), 1.5)
                actor.prop.point_size = base_radius * 10 * self.global_scale

    def update_atom_colors(self) -> None:
        """Only updates the colors of existing atom actors."""
        for atype, actor_name in [(k, f"atoms_{k}") for k in self.color_map]:
            if actor_name in self.renderer.actors:
                actor = self.renderer.actors[actor_name]
                color_hex = self.color_map.get(str(atype), "gray")
                actor.prop.color = color_hex

