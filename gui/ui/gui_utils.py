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
from PySide6 import QtWidgets, QtGui
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtCore import QObject

def _get_icon_path(name: str) -> str:
    """Get the absolute path of an SVG icon located in the icons folder."""
    # os.path.abspath(__file__) is '.../cemd/gui/tabs.py' for example
    # os.path.dirname(...) gives us '.../cemd/gui'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We build the path to icons/
    icon_path = os.path.join(base_dir, "icons", f"{name}.svg")
    
    # If the file is not there, it may be that we are in a gui subfolder (like gui/ui)
    if not os.path.exists(icon_path):
        parent_dir = os.path.dirname(base_dir)
        icon_path = os.path.join(parent_dir, "icons", f"{name}.svg")

    return icon_path.replace('\\', '/')

def get_icon(name: str) -> QtGui.QIcon:
        """Returns a QIcon object from the SVG file name."""
        path = _get_icon_path(name) # Use the get_icon_path function we did before
        if os.path.exists(path):
            return QtGui.QIcon(path)
        return QtGui.QIcon() # Returns an empty icon if not found

def create_icon_button(text: str, 
                       icon_name: str, 
                       primary: bool=False, 
                       parent_to_filter: QObject | None=None) -> QtWidgets.QPushButton:
    """
    Universal function to create the button.
    If parent_to_install_filter is provided, we install the eventFilter.
    """
    btn = QtWidgets.QPushButton(text)
    btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    
    if primary:
        btn.setObjectName("PrimaryAction")
    
    icon = get_icon(icon_name)
    if not icon.isNull():
        btn.setIcon(icon)
        # If we want the icon to resize, we must 
        # the object that contains the 'eventFilter' logic monitors this button.
        if parent_to_filter:
            btn.installEventFilter(parent_to_filter)
            
    return btn

def create_action_button(text: str, primary: bool=False) -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton(text)
    if primary:
        btn.setObjectName("PrimaryAction")
    return btn