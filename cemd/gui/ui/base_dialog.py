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

from PySide6 import QtWidgets, QtGui, QtCore
from .gui_utils import create_icon_button, create_action_button

class BaseBuilderDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Builder", width=450):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(width)

        self.main_layout = QtWidgets.QVBoxLayout(self) 
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

    def show_message(self, text, timeout=3000):
        """Displays a temporary message at the bottom left."""
        self.status_bar.showMessage(text, timeout)

    def show_error(self, title, message):
        """Displays a critical error popup."""
        QtWidgets.QMessageBox.critical(self, title, message)

    def show_warning(self, title, message):
        """Displays a warning popup."""
        QtWidgets.QMessageBox.warning(self, title, message)
    
    def create_icon_button(self, text, icon_name, primary=False):
        """
        Creates a button with an icon that adapts to its height (90%).
        """
        return create_icon_button(text, icon_name, primary, parent_to_filter=self)
    
    def create_action_button(self, text, primary=False):
        """Create a stylized button. primary=True gives it the color blue."""
        return create_action_button(text, primary)
    
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize and isinstance(obj, QtWidgets.QPushButton):
            if not obj.icon().isNull():
                h = obj.height()
                icon_dim = int(h * 0.7) 
                obj.setIconSize(QtCore.QSize(icon_dim, icon_dim))
        return super().eventFilter(obj, event)

    # def create_dialog_buttons(self, ok_text="OK"):
    #     """Creates the standard ButtonBox (Ok/Cancel)."""
    #     btns = QtWidgets.QDialogButtonBox(
    #         QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    #     )
        
    #     # This forces the button to display only the text
    #     btns.setCenterButtons(False)
        
    #     # Recover the buttons to remove the icon manually
    #     btns.button(QtWidgets.QDialogButtonBox.Ok).setIcon(QtGui.QIcon())
    #     btns.button(QtWidgets.QDialogButtonBox.Cancel).setIcon(QtGui.QIcon())

    #     btn_ok = btns.button(QtWidgets.QDialogButtonBox.Ok)
    #     btn_ok.setText(ok_text)
        
    #     btns.accepted.connect(self.accept)
    #     btns.rejected.connect(self.reject)
    #     return btns
    
    def create_dialog_buttons(self, ok_text="OK"):
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        
        btns.setCenterButtons(False)
        
        btn_ok = btns.button(QtWidgets.QDialogButtonBox.Ok)
        btn_ok.setIcon(QtGui.QIcon())
        btn_ok.setText(ok_text)
        btn_ok.setAutoDefault(False) 
        btn_ok.setDefault(False)
        
        btn_cancel = btns.button(QtWidgets.QDialogButtonBox.Cancel)
        btn_cancel.setIcon(QtGui.QIcon())
        btn_cancel.setAutoDefault(False)
        btn_cancel.setDefault(False)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        return btns

    def add_separator(self, layout):
        """Adds a horizontal separation line."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("color: #eee;")
        layout.addWidget(line)

    def create_table(self, headers, stretch_column=None, editable=False, selectable=False):
        """
        Creates a pre-configured QTableWidget to fill the space.
        """
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        header = table.horizontalHeader()
        
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
        if stretch_column is not None:
            for i in range(len(headers)):
                header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(stretch_column, QtWidgets.QHeaderView.Stretch)

        if editable is False:
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            
        if selectable is False:
            table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        return table