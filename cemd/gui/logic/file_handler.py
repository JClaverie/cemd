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
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from cemd.core.atomic_system import AtomicSystem

if TYPE_CHECKING:
    from ..main_window import AtomViewerGUI


def open_file(parent: AtomViewerGUI) -> None:
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent, "Open", "", "Atomic (*.data *.pdb *.cif *.sdf)"
    )

    if not file_path:
        return None, None

    try:
        parent.setUpdatesEnabled(False)
        system = AtomicSystem.from_file(file_path)

        return system, file_path

    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Unable to read file:\n{e}")
        return None, None

    finally:
        parent.setUpdatesEnabled(True)


def save_file(
    parent: AtomViewerGUI, current_system: AtomicSystem, current_path: str
) -> None:
    """Saves on the current path, but forces 'Save As' if it is not .data"""
    if not current_system:
        return False

    if not current_path or not current_path.lower().endswith(".data"):
        parent.save_file_as()
        return False

    try:
        current_system.write(current_path, atom_style=current_system._atom_style)
        parent.statusBar().showMessage(f"Saved file : {current_path}", 3000)
        return True
    except Exception as e:
        QtWidgets.QMessageBox.critical(parent, "Error", f"Cannot save :\n{e}")
        return False


def save_file_as(
    parent: AtomViewerGUI, current_system: AtomicSystem, current_path: str = None
) -> None:
    """Saves on the given path."""
    if not current_system:
        return None

    # Dialog setup
    dialog = QtWidgets.QFileDialog(parent, "Save as LAMMPS datafile")
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    dialog.setNameFilter("LAMMPS Data (*.data)")
    dialog.setDefaultSuffix("data")

    # ESSENTIAL: Forces Qt to use its own dialog (allows layout modification)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

    if current_path:
        dialog.setDirectory(os.path.dirname(current_path))

    # Creating the options widget
    option_group = QtWidgets.QGroupBox("Export options")
    layout = QtWidgets.QHBoxLayout()

    style_label = QtWidgets.QLabel("LAMMPS atom style :")
    style_combo = QtWidgets.QComboBox()
    style_combo.addItems(["atomic", "charge", "full"])
    style_combo.setCurrentText(current_system._atom_style)

    layout.addWidget(style_label)
    layout.addWidget(style_combo)
    option_group.setLayout(layout)

    # Secure addition to the layout
    dialog_layout = dialog.layout()
    if dialog_layout:
        # In a classic QFileDialog, the layout is a QGridLayout
        # We add the widget at the bottom of the grid
        row_count = dialog_layout.rowCount()
        dialog_layout.addWidget(option_group, row_count, 0, 1, -1)
    else:
        # Fallback solution if the layout is still not found
        # (very rare with DontUseNativeDialog)
        print("Warning: Could not access QFileDialog layout.")

    if dialog.exec() == QtWidgets.QDialog.Accepted:
        file_path = dialog.selectedFiles()[0]
        selected_style = style_combo.currentText()

        try:
            current_system.write(file_path, atom_style=selected_style)
            return file_path
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                parent, "Error", f"Save failed ({selected_style}) :\n{e}"
            )
            return None

    return None
