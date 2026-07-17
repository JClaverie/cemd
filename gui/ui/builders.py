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

import os
import io
import subprocess
from typing import TYPE_CHECKING, Any
from fractions import Fraction

from PySide6 import QtWidgets, QtCore, QtGui
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

from cemd.core.atomic_system import AtomicSystem
from cemd.builders.base import build_surface
from cemd.builders.hydrates import pycsh, make_csh, csh_to_cash
from ui.base_dialog import BaseBuilderDialog
from ui.gui_utils import get_icon
from ..._utils import concentration2count
from ..._config import PYCSH_DIR

if TYPE_CHECKING:
    from cemd.core.atomic_system import AtomicSystem

# Disable RDKit console spam for invalid SMILES during typing
RDLogger.DisableLog('rdApp.*')

SMILES_FRAGMENTS = {
    "Alkyls": {"Methyl": "C", "Ethyl": "CC", "Propyl": "CCC", "i-Propyl": "C(C)C", "t-Butyl": "C(C)(C)C"},
    "Organics": {"Amine": "N", "Hydroxyl": "O", "Carboxyl": "C(=O)O", "Ketone": "C(=O)", "Nitro": "[N+](=O)[O-]"},
    "Rings": {"Benzene": "c1ccccc1", "Cyclohexane": "C1CCCCC1", "Pyridine": "c1ccncc1", "Phenyl": "c1ccccc1-"},
    "Ions/Salts": {"Carbonate": "[O-]C(=O)[O-]", "Sulfate": "[O-]S(=O)(=O)[O-]", "Sodium": "[Na+]", "Chloride": "[Cl-]"}
}

def get_extended_miller_str(reduced_miller: list[int, int, int], 
                            shift: float) -> str:
    """
    Suggests an extended index ONLY if the shift is a simple fraction of type 1/n.
    """
    if shift < 1e-5: 
        return f"({reduced_miller[0]} {reduced_miller[1]} {reduced_miller[2]})"
    
    # looking for the closest fraction with a max denominator of 24
    frac = Fraction(shift).limit_denominator(24)
    
    if abs(float(frac) - shift) < 1e-3 and frac.numerator == 1:
        ext = [i * frac.denominator for i in reduced_miller]
        return f"({ext[0]} {ext[1]} {ext[2]})"
    
    return "N/A"


class IonGridWidget(QtWidgets.QGroupBox):
    def __init__(self, builder, title="Composition", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.builder = builder
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # Registry to store: { "Tab_Name": AtomicSystem_Object }
        self.custom_structures: dict[str, any] = {}

        # ---Input Section ---
        input_layout = QtWidgets.QHBoxLayout()
        
        self.combo_ion = QtWidgets.QComboBox()
        self.combo_ion.setEditable(True)
        self.combo_ion.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.combo_ion.lineEdit().setPlaceholderText("Search tabs or type name for PubChem...")
    
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Concentration (M)", "Fixed Count"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        self.val_spin = QtWidgets.QDoubleSpinBox()
        
        # Add button using your builder's icon system
        self.btn_add = self.builder.create_icon_button("", "plus-square-white")
        self.btn_add.clicked.connect(self.add_row)
        self.btn_add.setDefault(True)
    
        input_layout.addWidget(self.combo_ion, 2)
        input_layout.addWidget(self.mode_combo, 2)
        input_layout.addWidget(self.val_spin, 1)
        input_layout.addWidget(self.btn_add)
        
        self.layout.addLayout(input_layout)

        # ---Table Section ---
        self.table = self.builder.create_table(["Component", "Count", ""])
        self.layout.addWidget(self.table)

        self.status_label = QtWidgets.QLabel("Ready")
        self.layout.addWidget(self.status_label)

        self._on_mode_changed()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Triggered automatically when the user opens/shows the widget."""
        super().showEvent(event)
        self.refresh_from_tabs()

    def _on_mode_changed(self) -> None:
        """Update constraints based on selection mode."""
        is_conc = "Concentration" in self.mode_combo.currentText()
        if is_conc:
            self.val_spin.setDecimals(2)
            self.val_spin.setSuffix(" M")
            self.val_spin.setRange(0.0, 20.0) 
        else:
            self.val_spin.setDecimals(0)
            self.val_spin.setSuffix("")
            self.val_spin.setRange(1, 10000)

    def add_row(self) -> None:
        """
        Handles adding a component. If the name is unknown, it imports from PubChem 
        to the selection list. If known, it adds it to the composition table.
        """
        display_name = self.combo_ion.currentText().strip()
        if not display_name:
            return

        # ---CASE 1: The structure is already known (Local tabs or previously imported) ---
        if display_name in self.custom_structures:
            selected_system = self.custom_structures[display_name]
            ion_key = display_name
            
            # Internal helper logic to proceed with table insertion
            self._insert_into_table(display_name, ion_key, selected_system)
        
        # ---CASE 2: Unknown structure -> Search and Import from PubChem ---
        else:
            from .pubchem import PubChemBrowserDialog
            
            # Initialize dialog with the typed text
            dialog = PubChemBrowserDialog(self)
            dialog.search_input.setText(display_name)
            
            # Auto-launch search to save the user a click
            dialog.run_search()
            
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                selected_system = dialog.selected_system
                
                if selected_system:
                    # Retrieve metadata from the selected row
                    row = dialog.table.currentRow()
                    name = dialog.table.item(row, 1).text()
                    formula = dialog.table.item(row, 2).text()
                    
                    # Format name for the selection list
                    # The cloud icon indicates it was imported from the web
                    imported_name = f"{formula} ({name})"
                    
                    # Register the structure globally in this widget
                    self.custom_structures[imported_name] = selected_system
                    
                    # Add the new molecule to the Combo Box for selection
                    self.combo_ion.addItem(get_icon("mol"), imported_name, imported_name)
                    
                    # Set the combo to the newly imported item
                    idx = self.combo_ion.findText(imported_name)
                    self.combo_ion.setCurrentIndex(idx)
                    
                    self.status_label.setText(f"Imported '{name}' to selection list. Click '+' to add to table.")
            else:
                self.status_label.setText("Search cancelled.")

    def _insert_into_table(self, display_name: str, ion_key: str, selected_system: any) -> None:
        """
        Private helper to handle the actual table row creation and math.
        """
        if not selected_system:
            self.status_label.setText("Error: No structure available.")
            return

        raw_value = self.val_spin.value()
        mode = self.mode_combo.currentText()
        error_pct = 0.0

        # Calculate final particle count
        final_count = int(raw_value)
        if "Concentration" in mode:
            # Assuming spins_box and concentration2count are accessible
            volume = self.builder.get_volume()
            final_count, error_pct = concentration2count(volume, raw_value)
        
        # Insert new row into the table
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Column 0: Display Name (stored with the unique key)
        item_name = QtWidgets.QTableWidgetItem(display_name)
        item_name.setData(QtCore.Qt.ItemDataRole.UserRole, ion_key)
        self.table.setItem(row, 0, item_name)

        # Column 1: Count (stored as integer for easy retrieval)
        item_val = QtWidgets.QTableWidgetItem(str(final_count))
        item_val.setData(QtCore.Qt.ItemDataRole.UserRole, final_count)
        item_val.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, item_val)

        # Column 2: Remove button
        btn_del = self.builder.create_action_button("Remove")
        btn_del.clicked.connect(self.remove_row)
        self.table.setCellWidget(row, 2, btn_del)

        # Update status bar with results
        msg = f"Added {final_count} {display_name}"
        if error_pct >= 1e-4:
            msg += f" (error: {error_pct*100:.2f}%)"
        self.status_label.setText(msg)

    def remove_row(self) -> None:
        """Removes the row and updates the status message."""
        button = self.sender()
        if button:
            index = self.table.indexAt(button.pos())
            row = index.row()
            
            name_item = self.table.item(row, 0)
            count_item = self.table.item(row, 1)
            
            if name_item and count_item:
                name = name_item.text()
                count = count_item.data(QtCore.Qt.ItemDataRole.UserRole)
                
                self.status_label.setText(f"Removed {count} {name}")

            self.table.removeRow(row)
            
            # They see the reset
            if self.table.rowCount() == 0:
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))
        
    def refresh_from_tabs(self) -> None:
        """
        Syncs the combo box with tabs from the main GUI.
        Only updates the selection list, doesn't affect the table.
        """
        main_gui = getattr(self.builder, "parent_gui", None)
        tabs = getattr(main_gui, "tabs", None)

        # keep the current text to not disturb the user typing
        current_text = self.combo_ion.currentText()
        
        # Clear the dropdown list only
        self.combo_ion.clear()
        self.custom_structures.clear()

        if tabs and isinstance(tabs, QtWidgets.QTabWidget):
            mol_icon = get_icon("mol")
            for i in range(tabs.count()):
                name = tabs.tabText(i)
                tab_widget = tabs.widget(i)
                
                if hasattr(tab_widget, "system") and tab_widget.system:
                    # Registry of AVAILABLE structures
                    self.custom_structures[name] = tab_widget.system
                    self.combo_ion.addItem(mol_icon, name, name)

        # Restore typing
        self.combo_ion.setEditText(current_text)

    def update_counts_display(self) -> None:
        """Updates the table display with calculated counts from box dimensions."""
        # Accès direct aux spins de la classe parente (SolutionDialog)
        if not hasattr(self.builder, "spins_box"):
            return
        
        for row in range(self.table.rowCount()):
            item_name = self.table.item(row, 0)
            item_val = self.table.item(row, 1)
            if not item_name or not item_val: continue

            key, mode = item_name.data(QtCore.Qt.ItemDataRole.UserRole)
            val = item_val.data(QtCore.Qt.ItemDataRole.UserRole)

            if "Concentration" in mode:
                volume = self.builder.get_volume()
                counts = concentration2count(volume, {}, {key: val})
                n_mols = counts[key]
                # We update the displayed text without touching the 'UserRole' data
                item_val.setText(f"{val} M ({n_mols} units)")
            else:
                item_val.setText(f"{int(val)} units")


    def get_data(self) -> tuple[dict[str, int], dict[str, AtomicSystem]]:
        """Returns only the count dictionary and structures."""
        solute_dict = {}
        relevant_structures = {}

        for i in range(self.table.rowCount()):
            item_name = self.table.item(i, 0)
            item_val = self.table.item(i, 1)
            if not item_name or not item_val: continue
            
            key = item_name.data(QtCore.Qt.ItemDataRole.UserRole)
            count = item_val.data(QtCore.Qt.ItemDataRole.UserRole)

            solute_dict[key] = count

            if key in self.custom_structures:
                relevant_structures[key] = self.custom_structures[key]

        return solute_dict, relevant_structures


class SolutionDialog(BaseBuilderDialog):
    def __init__(self, parent_gui) -> None:
        # Using the uniform base class
        super().__init__(parent_gui, "Aqueous Solution Builder", 450)
        self.parent_gui = parent_gui
        self.resize(500, 550)
        self.setup_ui()

    def setup_ui(self):
        layout = self.layout() 
        layout.setSpacing(15)

        phys_group = QtWidgets.QGroupBox("")
        grid = QtWidgets.QGridLayout(phys_group)
        
        grid.addWidget(QtWidgets.QLabel("Box size (x, y, z)"), 0, 0)
        self.spins_box = []
        for i in range(3):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(10.0, 500.0)
            sp.setValue(30.0)
            sp.setDecimals(1)
            sp.setSuffix(" Å")
            grid.addWidget(sp, 0, i + 1)
            self.spins_box.append(sp)

        grid.addWidget(QtWidgets.QLabel("Target density"), 1, 0)
        self.sp_density = QtWidgets.QDoubleSpinBox()
        self.sp_density.setRange(0.1, 5.0)
        self.sp_density.setValue(1.0)
        self.sp_density.setDecimals(3)
        self.sp_density.setSingleStep(0.01)
        self.sp_density.setSuffix(" g/cm³")

        grid.addWidget(self.sp_density, 1, 1) 
        
        layout.addWidget(phys_group)

        self.ion_widget = IonGridWidget(builder=self, title="")
        layout.addWidget(self.ion_widget)

        self.button_box = self.create_dialog_buttons("Build Solution")
        layout.addWidget(self.button_box)

    def get_volume(self) -> float:
        """Calculates the rectangular box volume."""
        import numpy as np
        dims = [sp.value() for sp in self.spins_box]
        return float(np.prod(dims))

    def get_values(self)-> tuple[list, float, dict[str, int], dict[str, AtomicSystem]]:
        """Returns values ​​for solution generation."""
        solutes, structures = self.ion_widget.get_data()
        box_size=[sp.value() for sp in self.spins_box]
        density=self.sp_density.value()

        return box_size, density, solutes, structures
    
class AddStructureDialog(BaseBuilderDialog):
    def __init__(self, parent_gui) -> None:
        super().__init__(parent_gui, "Add Structure to Surface", 500)
        self.parent_gui = parent_gui
        
        # Registry to store the mapping: { "Tab_Name": AtomicSystem_Object }
        self.available_structures: dict[str, AtomicSystem] = {}
        
        self.resize(500, 400)
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        
        # ---Section 1: Structure Selection (IonGrid Style) ---
        select_group = QtWidgets.QGroupBox("Select structure to add")
        select_lay = QtWidgets.QVBoxLayout(select_group)
        
        self.struct_combo = QtWidgets.QComboBox()
        self.struct_combo = QtWidgets.QComboBox()
        self.struct_combo.setEditable(True)
        self.struct_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.struct_combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.struct_combo.setPlaceholderText("Select structures or search on PubChem...")
        
        # Connect the activation to check for the special "Search PubChem" item
        self.struct_combo.activated.connect(self._on_combo_activated)
        
        # Initial population of the combo box
        self.refresh_from_tabs()
        
        select_lay.addWidget(self.struct_combo)
        layout.addWidget(select_group)

        # ---Section 2: Geometric Parameters ---
        geo_group = QtWidgets.QGroupBox("Positioning")
        geo_lay = QtWidgets.QFormLayout(geo_group)
        geo_lay.setSpacing(12)
        
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(['x', 'y', 'z'])
        self.axis_combo.setCurrentText('z')
        
        self.distance = QtWidgets.QDoubleSpinBox()
        self.distance.setRange(0.0, 500.0)
        self.distance.setValue(2.0)
        self.distance.setDecimals(2)
        self.distance.setSuffix(" Å")
        
        self.vacuum = QtWidgets.QDoubleSpinBox()
        self.vacuum.setRange(0.0, 500.0)
        self.vacuum.setValue(10.0)
        self.vacuum.setSuffix(" Å")
        
        geo_lay.addRow("Contact axis:", self.axis_combo)
        geo_lay.addRow("Distance from surface:", self.distance)
        geo_lay.addRow("Additional vacuum:", self.vacuum)
        
        layout.addWidget(geo_group)
        
        # Add stretch to push everything up
        layout.addStretch()

        # ---Dialog Buttons ---
        self.button_box = self.create_dialog_buttons("Add structure")
        layout.addWidget(self.button_box)

    def showEvent(self, event: QtGui.QShowEvent):
        """Automatically refresh the list of tabs whenever the dialog is shown."""
        super().showEvent(event)
        self.refresh_from_tabs()

    def refresh_from_tabs(self):
        """Syncs combo with tabs and adds the PubChem trigger."""
        self.struct_combo.clear()
        self.available_structures.clear()
        
        mol_icon = get_icon("mol")
        search_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView)

        if self.struct_combo.count() > 0:
            self.struct_combo.insertSeparator(self.struct_combo.count())
            
        self.struct_combo.addItem(get_icon("search"), "Search in PubChem...", "PUBCHEM_ACTION")
        
        # 1. Add Local Tabs
        tabs = self.parent_gui.tabs
        for i in range(tabs.count()):
            name = tabs.tabText(i)
            tab_widget = tabs.widget(i)
            if hasattr(tab_widget, "system") and tab_widget.system:
                self.available_structures[name] = tab_widget.system
                self.struct_combo.addItem(mol_icon, name, name)
    
    def _on_combo_activated(self, index) -> None:
        """Checks if the user selected the PubChem search option."""
        data = self.struct_combo.itemData(index)
        
        if data == "PUBCHEM_ACTION":
            self._open_pubchem_search()

    def _open_pubchem_search(self) -> None:
        """Launches the PubChem browser and injects the result into the combo."""
        from .pubchem import PubChemBrowserDialog # Adjust import path
        
        browser = PubChemBrowserDialog(self)
        if browser.exec() == QtWidgets.QDialog.Accepted:
            system = browser.selected_system
            if system:
                # Get info from browser selection
                row = browser.table.currentRow()
                name = browser.table.item(row, 1).text()
                cid = browser.table.item(row, 0).text()
                display_name = f"{name} (CID: {cid})"
                
                # Store the system locally in this dialog
                self.available_structures[display_name] = system
                
                # Insert it into the combo box just above the separator
                self.struct_combo.insertItem(0, get_icon("mol"), display_name, display_name)
                self.struct_combo.setCurrentIndex(0)

    def get_values(self)-> None | dict[str, Any]:
        """Retrieves and validates the parameters for the logic function."""
        # Retrieve the selected tab name from currentData()
        selected_name = self.struct_combo.currentData()
        
        if not selected_name or selected_name not in self.available_structures:
            QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select a valid structure.")
            return None
            
        return {
            'structure_to_add': self.available_structures[selected_name],
            'axis': self.axis_combo.currentText().lower(),
            'distance': self.distance.value(),
            'vacuum': self.vacuum.value()
        }
    
class AddDropletDialog(BaseBuilderDialog):
    def __init__(self, parent_gui) -> None:
        # Title and width of 500px as requested
        super().__init__(parent_gui, "Add Liquid Droplet", 500)
        self.resize(500, 480)
        self.parent_gui = parent_gui
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        
        self.tabs = QtWidgets.QTabWidget()

        geo_widget = QtWidgets.QWidget()
        geo_lay = QtWidgets.QFormLayout(geo_widget)
        geo_lay.setContentsMargins(15, 15, 15, 15)
        geo_lay.setSpacing(12)
        
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setRange(5.0, 500.0)
        self.radius.setValue(30.0)
        self.radius.setDecimals(1)
        self.radius.setSuffix(" Å")

        self.distance = QtWidgets.QDoubleSpinBox()
        self.distance.setRange(0.0, 100.0)
        self.distance.setValue(1.5)
        self.distance.setDecimals(2)
        self.distance.setSuffix(" Å")
        
        self.density = QtWidgets.QDoubleSpinBox()
        self.density.setRange(0.01, 10.0)
        self.density.setValue(1.0)
        self.density.setSuffix(" g/cm³")
        
        self.vacuum = QtWidgets.QDoubleSpinBox()
        self.vacuum.setRange(0.0, 500.0)
        self.vacuum.setValue(10.0)
        self.vacuum.setSuffix(" Å")
        
        geo_lay.addRow("Radius:", self.radius)
        geo_lay.addRow("Target density:", self.density)
        geo_lay.addRow("Distance from surface:", self.distance)
        geo_lay.addRow("Additional vacuum:", self.vacuum)
        
        info_lbl = QtWidgets.QLabel("<i>Note: The droplet will be automatically centered on the XY plane.</i>")
        info_lbl.setWordWrap(True)
        geo_lay.addRow(info_lbl)
        
        self.tabs.addTab(geo_widget, "Geometry")

        self.ion_widget = IonGridWidget(builder=self, title="")
        self.tabs.addTab(self.ion_widget, "Ions & Concentration")
        
        layout.addWidget(self.tabs)
        layout.addSpacing(5)
        
        self.button_box = self.create_dialog_buttons("Add droplet")
        layout.addWidget(self.button_box)

    def get_volume(self) -> float:
        """Calculates the volume of a sphere (4/3 *pi *r^3)."""
        import math
        radius = self.radius.value()
        return (4.0 / 3.0) * math.pi * (radius ** 3)

    def get_values(self) -> dict[str, Any]:
        """Get all parameters for the add droplet function."""
        count, structs = self.ion_widget.get_data()
        return {
            'radius': self.radius.value(),
            'distance': self.distance.value(),
            'density': self.density.value(),
            'vacuum': self.vacuum.value(),
            'structures_dict': structs,
            'solutes_dict': count
        }


class AddLiquidLayerDialog(BaseBuilderDialog):
    def __init__(self, parent_gui) -> None:
        # CORRECTION : Ordre des arguments (parent, titre, largeur)
        super().__init__(parent_gui, "Add Liquid Layer", 500)
        self.resize(500, 450)
        self.parent_gui = parent_gui
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        
        self.tabs = QtWidgets.QTabWidget()
        
        geo_widget = QtWidgets.QWidget()
        geo_lay = QtWidgets.QFormLayout(geo_widget)
        geo_lay.setContentsMargins(15, 15, 15, 15)
        geo_lay.setSpacing(12)
        
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(['x', 'y', 'z'])
        self.axis_combo.setCurrentText('z')
        
        self.thickness = QtWidgets.QDoubleSpinBox()
        self.thickness.setRange(1.0, 500.0); self.thickness.setValue(20.0); self.thickness.setDecimals(2); self.thickness.setSuffix(" Å")

        self.distance = QtWidgets.QDoubleSpinBox()
        self.distance.setRange(0.0, 100.0); self.distance.setValue(2); self.distance.setSuffix(" Å")
        self.distance.setDecimals(2)
        
        self.density = QtWidgets.QDoubleSpinBox()
        self.density.setRange(0.01, 10.0); self.density.setValue(1.0); self.density.setSuffix(" g/cm³")
        
        self.vacuum = QtWidgets.QDoubleSpinBox()
        self.vacuum.setRange(0.0, 500.0); self.vacuum.setValue(0.0); self.vacuum.setSuffix(" Å")
        
        geo_lay.addRow("Contact axis:", self.axis_combo)
        geo_lay.addRow("Target density:", self.density)
        geo_lay.addRow("Layer thickness:", self.thickness)
        geo_lay.addRow("Distance from surface:", self.distance)
        geo_lay.addRow("Additional vacuum:", self.vacuum)
        
        self.tabs.addTab(geo_widget, "Geometry")

        self.ion_widget = IonGridWidget(builder=self, title="")
        self.tabs.addTab(self.ion_widget, "Solutes")
        
        layout.addWidget(self.tabs)

        # Small breathing space before the buttons
        layout.addSpacing(5)
        
        self.button_box = self.create_dialog_buttons("Add liquid layer")
        layout.addWidget(self.button_box)
        
    def get_volume(self) -> float:
        """
        Calculates the volume of the layer, respecting the surface area 
        of the existing system (even if non-rectangular).
        """
        system = getattr(self.parent_gui, "system", None)
        thickness = self.thickness.value()
        
        # We try to recover the real volume of the parent system
        if system and hasattr(system, "get_volume"):
            current_v = system.get_volume()
            # We recover the length of the growth axis (ex: z)
            lengths = system.cell.lengths()
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            idx = axis_map[self.axis_combo.currentText().lower()]
            
            # Surface area = Total volume /Length of perpendicular axis
            surface_area = current_v / lengths[idx]
            return surface_area * thickness
            
        # Fallback if no system loaded: we assume a surface of 20x20
        return 20.0 * 20.0 * thickness
    
    def get_values(self)-> dict[str, Any]:
        """Retrieves all the parameters for the interface function."""
        count, structs = self.ion_widget.get_data()
        return {
            'axis': self.axis_combo.currentText().lower(),
            'thickness': self.thickness.value(),
            'distance': self.distance.value(), # Added to dictionary
            'density': self.density.value(),
            'vacuum': self.vacuum.value(),
            'solutes_dict': count,
            'structures_dict': structs
        }


class pyCSHGeneratorDialog(BaseBuilderDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, "pyCSH C-(A)-S-H Builder", 620)
        self.resize(620, 700)

        # Paths and storage
        self.output_path = os.path.abspath(os.path.join(PYCSH_DIR, "output"))
        self.generated_systems = []
        self.selected_systems = []
        self.real_as_list = []
        
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        layout.setSpacing(12)

        # Parameters
        params_group = QtWidgets.QGroupBox("Input parameters")
        form = QtWidgets.QFormLayout(params_group)
        form.setSpacing(8)
        
        self.cs_ratio = QtWidgets.QDoubleSpinBox(); self.cs_ratio.setRange(0.5, 3.0); self.cs_ratio.setValue(1.5)
        self.ws_ratio = QtWidgets.QDoubleSpinBox(); self.ws_ratio.setRange(0.5, 3.0); self.ws_ratio.setValue(1.2)
        self.as_ratio = QtWidgets.QDoubleSpinBox(); self.as_ratio.setRange(0.0, 1.0); self.as_ratio.setValue(0.0)
        self.nsamples = QtWidgets.QSpinBox(); self.nsamples.setRange(1, 100); self.nsamples.setValue(1)
        
        # Supercell (x y z)
        sc_lay = QtWidgets.QHBoxLayout()
        self.na, self.nb, self.nc = QtWidgets.QSpinBox(), QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sb, val in zip([self.na, self.nb, self.nc], [3, 5, 2]):
            sb.setRange(1, 50)
            sb.setValue(val)
            sc_lay.addWidget(sb)
        
        form.addRow("Target Ca/Si:", self.cs_ratio)
        form.addRow("Target H₂O/Si:", self.ws_ratio)
        form.addRow("Target Al/Si:", self.as_ratio)
        form.addRow("Number of samples:", self.nsamples)
        form.addRow("Supercell (x y z):", sc_lay)
        layout.addWidget(params_group)

        self.btn_run = self.create_action_button("Generate structures", primary=True)
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.on_run)
        layout.addWidget(self.btn_run)

        self.pdf_group = QtWidgets.QGroupBox("See characteristic properties")
        pdf_lay = QtWidgets.QHBoxLayout(self.pdf_group)
        for name in ["distributions", "MCL", "water", "XOX_X"]:
            btn = self.create_icon_button(name.upper(), "file-pdf")
            btn.clicked.connect(lambda chk=False, n=name: self.open_pdf(n))
            pdf_lay.addWidget(btn)
        self.pdf_group.setEnabled(False)
        layout.addWidget(self.pdf_group)

        self.table = self.create_table(["", "Al/Si", "Ca/Si", "SiOH/Si", "MCL"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(250)
        layout.addWidget(self.table)
        
        self.button_box = self.create_dialog_buttons("Accept Selection")
        self.button_box.accepted.disconnect()
        self.button_box.accepted.connect(self.validate_selection)
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.button_box)
        
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")

        layout.addWidget(self.status_bar)


    def on_run(self):
        """Starts the calculation with a wait cursor."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtCore.QTimer.singleShot(100, self.execute_calculation)

    def execute_calculation(self):

        params = {
            "cs_ratio": self.cs_ratio.value(), 
            "ws_ratio": self.ws_ratio.value(),
            "supercell": [self.na.value(), self.nb.value(), self.nc.value()],
            "nsamples": self.nsamples.value()
        }
        
            
        self.worker = TaskWorker(pycsh, **params)
    
        self.btn_run.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, self.nsamples.value())
        self.progress_bar.setValue(0)

        self.progress_bar.setFormat("%v/%m") 
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        
        self.worker.status_msg.connect(self.status_bar.showMessage)
        
        self.worker.finished.connect(self.on_calculation_finished)
        self.worker.error.connect(self.on_calculation_error)

        self.status_bar.showMessage("pyCSH is running, it could take some time...")
        
        self.worker.start()

    def on_calculation_finished(self, results) -> None:
            try:
                from cemd.builders.hydrates import csh_to_cash 
                
                raw_systems = results if isinstance(results, list) else [results]
                al_si = self.as_ratio.value()
                
                self.real_as_list = []
                self.generated_systems = []
                
                # C-A-S-H logic
                if al_si > 0:
                    for s in raw_systems:
                        new_sys, real_as = csh_to_cash(s, as_ratio=al_si)
                        self.generated_systems.append(new_sys)
                        self.real_as_list.append(real_as)
                else:
                    self.generated_systems = raw_systems
                    self.real_as_list = [0.0] * len(raw_systems)

                # UI update
                self.update_table_from_log()
                self.pdf_group.setEnabled(True)
                self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
                self.status_bar.showMessage("Generation completed!", 5000)
                
            except Exception as e:
                self.on_calculation_error(str(e))
            finally:
                self.progress_bar.hide()
                self.btn_run.setEnabled(True)
                QtWidgets.QApplication.restoreOverrideCursor()
                self.adjustSize()

    def on_calculation_error(self, error_str):
        self.progress_bar.hide()
        self.btn_run.setEnabled(True)
        self.status_bar.showMessage("Error during generation.")
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.critical(self, "Error", f"Generation failed: {error_str}")

    def update_table_from_log(self):
        log_path = os.path.join(self.output_path, "created_samples.log")
        if not os.path.exists(log_path): return
        
        self.table.setRowCount(0)
        import re
        pattern = re.compile(r"Sample:\s+(?P<id>\d+)\s+Ca/Si:\s+(?P<cs>[\d.]+)\s+SiOH/Si:\s+(?P<sioh>[\d.]+)\s+CaOH/Ca:\s+(?P<caoh>[\d.]+)\s+MCL:\s+(?P<mcl>[\d.]+)")
        
        with open(log_path, 'r') as f:
            for i, line in enumerate(f):
                match = pattern.search(line)
                if match:
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    
                    # Checkbox
                    ck_item = QtWidgets.QTableWidgetItem()
                    ck_item.setCheckState(QtCore.Qt.Checked)
                    self.table.setItem(row, 0, ck_item)
                    
                    # Real Al/Si
                    val_al = self.real_as_list[i] if i < len(self.real_as_list) else 0.0
                    self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val_al:.4f}"))
                    
                    # Log data
                    self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(match.group('cs')))
                    self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(match.group('sioh')))
                    self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(match.group('mcl')))

    def validate_selection(self):
        self.selected_systems = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == QtCore.Qt.Checked:
                if i < len(self.generated_systems):
                    self.selected_systems.append(self.generated_systems[i])
        
        if not self.selected_systems:
            QtWidgets.QMessageBox.warning(self, "Selection", "Please select at least one sample.")
            return
        self.accept()

    def open_pdf(self, name):
        pdf_path = os.path.abspath(os.path.join(self.output_path, f"{name}.pdf"))
        if not os.path.exists(pdf_path): return
        env = os.environ.copy()
        for v in ["LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH"]: env.pop(v, None)
        for cmd in [['firefox', '--new-window'], ['xdg-open']]:
            try:
                subprocess.Popen(cmd + [pdf_path], env=env)
                return
            except: continue


class ReplicateDialog(BaseBuilderDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "Supercell Builder", 300)
        self.setup_ui()

    def setup_ui(self):
        layout = self.main_layout
        
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignCenter)
        form.setSpacing(15)

        self.spins = []
        axes_labels = ["n<sub>a</sub>", "n<sub>b</sub>", "n<sub>c</sub>"]
        
        for label_text in axes_labels:
            sp = QtWidgets.QSpinBox()
            sp.setRange(1, 100)
            sp.setValue(1)
            sp.setFixedWidth(80) 
            
            form.addRow(QtWidgets.QLabel(f"{label_text}"), sp)
            self.spins.append(sp)
        
        layout.addLayout(form)
        
        self.add_separator(layout)
        self.button_box = self.create_dialog_buttons("Create supercell")
        layout.addWidget(self.button_box)

    def get_values(self):
        return [sp.value() for sp in self.spins]


class SurfaceDialog(BaseBuilderDialog):
    def __init__(self, data_to_process, parent=None):
        # Correction of the order: parent, title, width
        super().__init__(parent, "Slab Generator", 750) 
        self.resize(750, 650)

        self.data_in = data_to_process
        self.all_slabs = []
        self.selected_systems = []
        
        self.setup_ui()

    def setup_ui(self):
        layout = self.main_layout
        layout.setSpacing(10)

        params_group = QtWidgets.QGroupBox("Input parameters")
        form = QtWidgets.QFormLayout(params_group)
        
        miller_lay = QtWidgets.QHBoxLayout()
        self.h, self.k, self.l = QtWidgets.QSpinBox(), QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sb in [self.h, self.k, self.l]:
            sb.setRange(-10, 10)
            sb.setFixedWidth(60)
            miller_lay.addWidget(sb)
        self.l.setValue(1)
        
        self.slab_size = QtWidgets.QDoubleSpinBox(); self.slab_size.setRange(1, 500); self.slab_size.setValue(25.0); self.slab_size.setSuffix(" Å")
        self.vac_size = QtWidgets.QDoubleSpinBox(); self.vac_size.setRange(0, 500); self.vac_size.setValue(15.0); self.vac_size.setSuffix(" Å")
        
        form.addRow("Miller indices (h k l):", miller_lay)
        form.addRow("Min. slab size:", self.slab_size)
        form.addRow("Min. vacuum size:", self.vac_size)
        layout.addWidget(params_group)

        self.btn_search = self.create_action_button("Generate Surface Candidates", primary=True)
        self.btn_search.setMinimumHeight(40)
        self.btn_search.clicked.connect(self.on_search)
        layout.addWidget(self.btn_search)

        self.table = self.create_table([
            "", "Candidate", "Shift (frac)", "Cutting Plan", "Dipole (e)"
        ])
        layout.addWidget(self.table)

        self.info_lbl = QtWidgets.QLabel("Ready.")
        layout.addWidget(self.info_lbl)

        self.button_box = self.create_dialog_buttons("Accept Selection")
        self.btn_ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setEnabled(False) 
        
        self.button_box.accepted.disconnect()
        self.button_box.accepted.connect(self.validate_selection)
        layout.addWidget(self.button_box)

    def on_search(self):
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            miller = [self.h.value(), self.k.value(), self.l.value()]
            
            self.all_slabs, shifts, dipoles, broken = build_surface(
                self.data_in, miller, self.slab_size.value(), self.vac_size.value()
            )
            
            self.table.setRowCount(len(self.all_slabs))
            for i, slab_as in enumerate(self.all_slabs):
                # Checkbox
                ck_item = QtWidgets.QTableWidgetItem()
                ck_item.setCheckState(QtCore.Qt.Unchecked)
                self.table.setItem(i, 0, ck_item)
                
                # Infos
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"Slab #{i+1}"))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{shifts[i]:.4f}"))
                
                # Cutting plan (Miller extended)
                hkl_ext = get_extended_miller_str(miller, shifts[i]) 
                item_hkl = QtWidgets.QTableWidgetItem(hkl_ext)
                item_hkl.setToolTip(f"Shift value: {shifts[i]:.6f}")
                self.table.setItem(i, 3, item_hkl)
                
                # Dipole
                self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{dipoles[i]:.4f}"))
            
            self.info_lbl.setText(f"Found {len(self.all_slabs)} structures. (Broken bonds: {broken})")
            self.btn_ok.setEnabled(True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Slab generation failed:\n{str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def validate_selection(self):
        """Recovers systems checked by the user."""
        self.selected_systems = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == QtCore.Qt.Checked:
                self.selected_systems.append(self.all_slabs[i])

        if not self.selected_systems:
            QtWidgets.QMessageBox.warning(self, "Selection", "Please check at least one surface candidate.")
            return
            
        self.accept()

    def get_values(self):
        """Useful for persistence of parameters if necessary."""
        return ([self.h.value(), self.k.value(), self.l.value()], 
                self.slab_size.value(), self.vac_size.value())


class CASHBuilderDialog(BaseBuilderDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "CEMD C-(A)-S-H Builder", 450)
        self.resize(450, 450)
        self.generated_system = None 
        
        self.setup_ui()

    def setup_ui(self):
        layout = self.main_layout
        
        input_group = QtWidgets.QGroupBox("Input parameters")
        form = QtWidgets.QFormLayout(input_group)
        form.setSpacing(10)
        
        self.cs = QtWidgets.QDoubleSpinBox(); self.cs.setRange(0.5, 3.0); self.cs.setValue(1.5); self.cs.setSingleStep(0.1)
        self.ws = QtWidgets.QDoubleSpinBox(); self.ws.setRange(0, 10); self.ws.setValue(1.8); self.ws.setSingleStep(0.1)
        self.as_r = QtWidgets.QDoubleSpinBox(); self.as_r.setRange(0, 1); self.as_r.setValue(0.0); self.as_r.setSingleStep(0.1)
        self.min_mcl = QtWidgets.QDoubleSpinBox(); self.min_mcl.setRange(1, 20); self.min_mcl.setValue(3.0); self.min_mcl.setSingleStep(0.1)
        
        sc_lay = QtWidgets.QHBoxLayout()
        self.sx, self.sy, self.sz = QtWidgets.QSpinBox(), QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for sb, v in zip([self.sx, self.sy, self.sz], [3, 5, 1]):
            sb.setRange(1, 20)
            sb.setValue(v)
            sc_lay.addWidget(sb)
            
        form.addRow("Target Ca/Si:", self.cs)
        form.addRow("Target H₂O/Si:", self.ws)
        form.addRow("Target Al/Si:", self.as_r)
        form.addRow("Minimum MCL:", self.min_mcl)
        form.addRow("Supercell (x y z):", sc_lay)
        layout.addWidget(input_group)

        self.btn_calc = self.create_action_button("Build the C-(A)-S-H structure", primary=True)
        self.btn_calc.setMinimumHeight(40)
        self.btn_calc.clicked.connect(self.on_calculate)
        layout.addWidget(self.btn_calc)

        self.res_group = QtWidgets.QGroupBox("System information")
        self.res_group.setVisible(False)
        res_form = QtWidgets.QFormLayout(self.res_group)
        self.l_cs = QtWidgets.QLabel("-")
        self.l_ws = QtWidgets.QLabel("-")
        self.l_al = QtWidgets.QLabel("-")
        res_form.addRow("Real Ca/Si:", self.l_cs)
        res_form.addRow("Real H₂O/Si:", self.l_ws)
        res_form.addRow("Real Al/Si:", self.l_al)
        layout.addWidget(self.res_group)

        self.button_box = self.create_dialog_buttons("Accept Structure")
        self.btn_ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setEnabled(False) 
        layout.addWidget(self.button_box)

        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setFixedHeight(25)
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.showMessage("Ready")
        
        
        layout.addWidget(self.status_bar)

    def on_calculate(self):
        """Generation logic (unchanged but secure)"""
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            self.show_message("Generating C-(A)-S-H structure... Please wait")
            
            params = {
                "cs_ratio": self.cs.value(),
                "ws_ratio": self.ws.value(),
                "supercell": [self.sx.value(), self.sy.value(), self.sz.value()],
                "min_mcl": self.min_mcl.value(),
                "model": 'tob11a_hamid.cif'
            }

            self.worker = TaskWorker(make_csh, **params)

            self.worker.status_msg.connect(self.set_status)
            self.worker.finished.connect(self.on_success)
            self.worker.error.connect(lambda e: self.show_error("Calculation Error", e))
            
            self.worker.start()
            

        except Exception as e:
            self.show_error("Error", f"Generation failed: {str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_success(self, system: AtomicSystem) -> None:
        """This method is called automatically when the Worker has finished"""
        try:
            if self.as_r.value() > 0:
                res_al = csh_to_cash(system, as_ratio=self.as_r.value())
                system = res_al[0] if isinstance(res_al, (tuple, list)) else res_al

            nsi = system.get_count('Si')
            nca = system.get_count('Ca') + system.get_count('Cw')
            nh = 0
            for i in ['H', 'Hw', 'Hsi' 'Hh']:
                nh += system.get_count(i)
            nal = system.get_count('Al')
            
            self.l_cs.setText(f"<b>{nca/nsi:.3f}</b>" if nsi > 0 else "0")
            self.l_al.setText(f"<b>{nal/nsi:.4f}</b>" if nsi > 0 else "0")
            self.l_ws.setText(f"<b>{(nh/2)/nsi:.3f}</b>" if nsi > 0 else "0")
            
            self.res_group.setVisible(True)
            self.generated_system = system
            self.btn_ok.setEnabled(True)
            self.adjustSize()

        except Exception as e:
            self.handle_error(str(e))

    def get_system(self):
        """Returns the generated system to add it to the main interface."""
        return self.generated_system

    def accept(self):
        """Ensures that the dialog only closes if a system exists."""
        if self.generated_system:
            super().accept()
        else:
            self.show_warning("Warning", "Please build a structure first.")

    def set_status(self, message: str):
        """Updates the status bar text."""
        self.status_bar.showMessage(message)

    def handle_error(self, error_msg: str):
        """Manages the display of errors coming from the Worker or post-processing."""
        # restore the button and the cursor
        self.btn_calc.setEnabled(True) 
        QtWidgets.QApplication.restoreOverrideCursor()
        
        # display the message to the user
        QtWidgets.QMessageBox.critical(self, "Calculation Error", f"An error occurred: {error_msg}")
        self.set_status("Error.")

class SplitterDialog(BaseBuilderDialog):
    def __init__(self, parent_gui=None) -> None:
        super().__init__(parent_gui, "Split structure", 500)
        self.resize(500, 500)
        self.parent_gui = parent_gui
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        
        self.tabs = QtWidgets.QTabWidget()
        
        # Geometry
        geo_w = QtWidgets.QWidget()
        geo_lay = QtWidgets.QFormLayout(geo_w)
        geo_lay.setContentsMargins(15, 15, 15, 15)
        geo_lay.setSpacing(12)
        
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["a", "b", "c"])
        self.axis.setCurrentIndex(2) # Default on 'c'
        
        self.coord = QtWidgets.QDoubleSpinBox()
        self.coord.setRange(-5000, 5000)
        self.coord.setSuffix(" Å")
        
        self.gap = QtWidgets.QDoubleSpinBox()
        self.gap.setRange(0, 1000)
        self.gap.setValue(30.0)
        self.gap.setSuffix(" Å")
        
        self.add_sol = QtWidgets.QCheckBox("Fill the gap with a liquid solution")
        
        geo_lay.addRow("Split axis:", self.axis)
        geo_lay.addRow("Split coordinate:", self.coord)
        geo_lay.addRow("Gap width:", self.gap)
        geo_lay.addRow("", self.add_sol)
        
        self.tabs.addTab(geo_w, "Geometry")

        # Solution (Parameters + Ions)
        sol_w = QtWidgets.QWidget()
        sol_lay = QtWidgets.QVBoxLayout(sol_w)
        sol_lay.setContentsMargins(10, 10, 10, 10)
        
        form_sol = QtWidgets.QFormLayout()
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setRange(0.1, 10.0); self.tol.setValue(2.0); self.tol.setSuffix(" Å")
        
        self.dens = QtWidgets.QDoubleSpinBox()
        self.dens.setRange(0.01, 10.0); self.dens.setValue(1.0); self.dens.setSuffix(" g/cm³")
        
        form_sol.addRow("Distance from surface:", self.tol)
        form_sol.addRow("Target density:", self.dens)
        sol_lay.addLayout(form_sol)
        
        self.ion_widget = IonGridWidget(builder=self, title="")
        sol_lay.addWidget(self.ion_widget)
        self.tabs.addTab(sol_w, "Solution")
        
        layout.addWidget(self.tabs)
        
        # Dialog Buttons
        self.button_box = self.create_dialog_buttons("Apply Transformation")
        layout.addWidget(self.button_box)
        
        # Activation logic
        self.add_sol.toggled.connect(self._toggle_solution_tab)
        self._toggle_solution_tab(False)

    def _toggle_solution_tab(self, enabled: bool) -> None:
        """Activates or deactivates the Solution tab based on the checkbox."""
        self.tabs.setTabEnabled(1, enabled)
        if not enabled:
            self.tabs.setCurrentIndex(0)

    def get_volume(self) -> float:
        """Calculates the volume of the gap (channel) to be filled."""
        system = getattr(self.parent_gui, "system", None)
        gap_width = self.gap.value()
        
        if system and hasattr(system, "get_volume"):
            # Same logic as for the liquid layer
            current_v = system.get_volume()
            lengths = system.cell.lengths()
            # Here the axis is the index selected in the combo (0, 1, 2)
            idx = self.axis.currentIndex() 
            
            surface_area = current_v / lengths[idx]
            return surface_area * gap_width
            
        return 20.0 * 20.0 * gap_width

    def get_values(self)-> dict[str, Any]:
        count, structs = self.ion_widget.get_data()
        return {
            "axis": self.axis.currentIndex(), 
            "coordinate": self.coord.value(), 
            "gap_size": self.gap.value(),
            "tolerance": self.tol.value() if self.add_sol.isChecked() else 0,
            "add_solution": self.add_sol.isChecked(), 
            "density": self.dens.value(),
            "solutes_dict": count,
            "structures_dict": structs
        }

class GlassBuilderDialog(BaseBuilderDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, "Glass Builder", 450)
        self.resize(500, 650)
        
        self.raw_recipes = [
            "SiO2", "Al2O3", "B2O3", "P2O5", "TiO2", "ZrO2",
            "Na2O", "K2O", "Li2O", "CaO", "MgO", "BaO", "SrO", "ZnO",
            "FeO", "Fe2O3", "MnO", "Cr2O3", "H2O"
        ]
        self.setup_ui()

    def _to_subscript(self, formula: str) -> str:
        """Transform 'SiO2' into 'SiO₂'."""
        sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return formula.translate(sub_map)

    def setup_ui(self) -> None:
        layout = self.main_layout
        layout.setSpacing(10)

        # Physical Parameters
        box_group = QtWidgets.QGroupBox("")
        grid = QtWidgets.QGridLayout(box_group)
        
        grid.addWidget(QtWidgets.QLabel("Box size (x, y, z)"), 0, 0)
        self.spin_box = []
        for i in range(3):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(5.0, 500.0)
            s.setValue(30.0)
            s.setDecimals(1)
            s.setSuffix(" Å")
            grid.addWidget(s, 0, i + 1)
            self.spin_box.append(s)

        grid.addWidget(QtWidgets.QLabel("Target density"), 1, 0)
        self.spin_density = QtWidgets.QDoubleSpinBox()
        self.spin_density.setRange(0.01, 20.0)
        self.spin_density.setValue(2.50)
        self.spin_density.setDecimals(3)
        self.spin_density.setSingleStep(0.1)
        self.spin_density.setSuffix(" g/cm³")
        grid.addWidget(self.spin_density, 1, 1)
        layout.addWidget(box_group)

        # Entry of components
        input_group = QtWidgets.QGroupBox("Components")
        input_layout = QtWidgets.QHBoxLayout(input_group)
        
        self.combo_comp = QtWidgets.QComboBox()
        for r in sorted(self.raw_recipes):
            self.combo_comp.addItem(self._to_subscript(r), r)
        self.combo_comp.setEditable(True)
        self.combo_comp.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        
        self.spin_coeff = QtWidgets.QDoubleSpinBox()
        self.spin_coeff.setRange(0.001, 10000.0)
        self.spin_coeff.setValue(1.0)
        self.spin_coeff.setDecimals(1)
        self.spin_coeff.setFixedWidth(80)

        self.btn_add = self.create_icon_button("", "plus-square")
        self.btn_add.clicked.connect(self.add_component)

        self.btn_add.setSizePolicy(
        QtWidgets.QSizePolicy.MinimumExpanding, 
        QtWidgets.QSizePolicy.MinimumExpanding
        )
        
        input_layout.addWidget(self.combo_comp, 1)
        input_layout.addWidget(self.spin_coeff)
        input_layout.addWidget(self.btn_add, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(input_group)

        self.table = self.create_table(["Component", "Molar Coeff", ""])
        layout.addWidget(self.table)

        self.button_box = self.create_dialog_buttons("Generate Structure")
        layout.addWidget(self.button_box)

    def add_component(self) -> None:
        idx = self.combo_comp.currentIndex()
        if idx != -1 and self.combo_comp.currentText() == self.combo_comp.itemText(idx):
            formula = self.combo_comp.itemData(idx)
        else:
            formula = self.combo_comp.currentText().strip()
        
        coeff = self.spin_coeff.value()
        if not formula or coeff <= 0:
            return

        row = self.table.rowCount()
        self.table.insertRow(row)
        
        item_name = QtWidgets.QTableWidgetItem(self._to_subscript(formula))
        item_name.setData(QtCore.Qt.UserRole, formula)
        
        self.table.setItem(row, 0, item_name)
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{coeff:.3f}"))

        btn_del = self.create_action_button("Remove")
        btn_del.clicked.connect(lambda: self.table.removeRow(self.table.indexAt(btn_del.pos()).row()))

        self.table.setCellWidget(row, 2, btn_del)

    def get_values(self) -> tuple[list, float, dict]:
        """Returns values ready for make_glass."""
        box = [s.value() for s in self.spin_box]
        density = self.spin_density.value()
        
        stoich = {}
        for i in range(self.table.rowCount()):
            key = self.table.item(i, 0).data(QtCore.Qt.UserRole)
            try:
                val = float(self.table.item(i, 1).text())
                stoich[key] = val
            except (ValueError, AttributeError):
                continue
                
        return box, density, stoich


class SmilesDialog(QtWidgets.QDialog):
    """
    A dialog for creating an AtomicSystem from a SMILES string.
    Includes a 2D preview and quick-access fragment buttons.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SMILES Molecule Builder")
        self.setMinimumWidth(500)
        
        self.system = None  # Holds the resulting AtomicSystem
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # ---Fragment Toolbar (Tabs) ---
        self.tabs = QtWidgets.QTabWidget()
        for category, items in SMILES_FRAGMENTS.items():
            container = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(container)
            grid.setContentsMargins(5, 5, 5, 5)
            grid.setSpacing(4)
            
            for i, (name, sm) in enumerate(items.items()):
                btn = QtWidgets.QPushButton(name)
                btn.setToolTip(f"Insert {sm}")
                # Capture fragment string in lambda
                btn.clicked.connect(lambda _, s=sm: self._insert_fragment(s))
                grid.addWidget(btn, i // 4, i % 4)
            
            self.tabs.addTab(container, category)
        self.main_layout.addWidget(self.tabs)

        # ---Input Section ---
        self.main_layout.addWidget(QtWidgets.QLabel("<b>SMILES String:</b>"))
        input_row = QtWidgets.QHBoxLayout()
        self.smiles_input = QtWidgets.QLineEdit()
        self.smiles_input.setPlaceholderText("e.g., CCO or [O-]C(=O)[O-]")
        self.smiles_input.textChanged.connect(self.update_preview)
        
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.smiles_input.clear)
        
        input_row.addWidget(self.smiles_input)
        input_row.addWidget(self.clear_btn)
        self.main_layout.addLayout(input_row)

        # ---Preview Section ---
        self.preview_label = QtWidgets.QLabel("Valid SMILES required for preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(350, 350)
        self.main_layout.addWidget(self.preview_label)

        # ---Info/Status Section ---
        self.info_label = QtWidgets.QLabel("")
        self.main_layout.addWidget(self.info_label)

        # ---Action Buttons ---
        self.btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self.create_system)
        self.btns.rejected.connect(self.reject)
        self.btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.main_layout.addWidget(self.btns)

    def _insert_fragment(self, fragment: str) -> None:
        """Appends a SMILES fragment to the current input text."""
        current_text = self.smiles_input.text()
        # Add a dot separator if it's a new ionic species, otherwise append
        if current_text and not current_text.endswith((".", "(", "=", "#", "-")):
            self.smiles_input.setText(current_text + "." + fragment)
        else:
            self.smiles_input.setText(current_text + fragment)
        self.smiles_input.setFocus()

    def update_preview(self) -> None:
        """Generates a 2D rendering of the molecule using RDKit."""
        smiles = self.smiles_input.text().strip()
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            try:
                # Generate image using RDKit
                img = Draw.MolToImage(mol, size=(350, 350))
                
                # Convert PIL image to QPixmap
                byte_io = io.BytesIO()
                img.save(byte_io, format='PNG')
                qimg = QtGui.QImage.fromData(byte_io.getvalue())
                self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
                
                # Update metadata display
                charge = Chem.GetFormalCharge(mol)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                self.info_label.setText(f"Formula: {formula} | Global Charge: {charge}")
                
                self.btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            except Exception:
                self._show_invalid_state()
        else:
            self._show_invalid_state()

    def _show_invalid_state(self) -> None:
        """Resets the UI elements to an 'Invalid' or 'Empty' display state."""
        self.preview_label.clear()
        self.preview_label.setText("Invalid SMILES syntax")
        self.info_label.setText("")
        self.btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def create_system(self) -> None:
        """Converts the SMILES to an AtomicSystem and closes the dialog."""
        smiles = self.smiles_input.text().strip()
        try:
            from cemd.core.atomic_system import AtomicSystem
            # Uses the classmethod we defined earlier
            self.system = AtomicSystem.from_smiles(smiles)
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Conversion Error", f"Failed to generate 3D structure:\n{str(e)}")

class TranslateAtomsDialog(BaseBuilderDialog):
    def __init__(self, parent_gui) -> None:
        super().__init__(parent_gui, "Translate Atoms", 350)
        self.parent_gui = parent_gui
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = self.main_layout
        
        # ---Grid for X, Y, Z inputs ---
        vec_group = QtWidgets.QGroupBox("Translation vector")
        grid = QtWidgets.QGridLayout(vec_group)
        grid.setSpacing(10)
        
        self.spins = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            grid.addWidget(QtWidgets.QLabel(f"{axis.upper()}:"), i, 0)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1000.0, 1000.0)
            sp.setValue(0.0)
            sp.setDecimals(2)
            sp.setSuffix(" Å")
            grid.addWidget(sp, i, 1)
            self.spins[axis] = sp
            
        layout.addWidget(vec_group)
        layout.addStretch()

        self.button_box = self.create_dialog_buttons("Apply")
        layout.addWidget(self.button_box)

    def get_values(self) -> list[float]:
        """Returns the translation vector [dx, dy, dz]."""
        return [self.spins[ax].value() for ax in ['x', 'y', 'z']]

class TaskWorker(QtCore.QThread):
    progress_changed = QtCore.Signal(int)
    status_msg = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, func, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            def update_ui(percent, msg) -> None:
                self.progress_changed.emit(percent)
                self.status_msg.emit(msg)

            result = self.func(**self.kwargs, progress_callback=update_ui)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))