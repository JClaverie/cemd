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
import json
import re
import webbrowser
import requests
from io import StringIO
from PySide6 import QtWidgets, QtCore
from pymatgen.io.cif import CifParser
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cemd.core.atomic_system import AtomicSystem
    from main_window import AtomViewerGUI

from ui.base_dialog import BaseBuilderDialog

def formula_to_elements(formula: str) -> list[dict[str, Any]]:
    """
    Transforms a chemical formula into a list of unique elements.
    Example: "Ca(OH)2" -> ['Ca', 'O', 'H']
              "Al2O3" -> ['Al', 'O']
    """
    
    return list(dict.fromkeys(re.findall(r'[A-Z][a-z]?', formula)))

def cod_search_by_elements(elements_list: list) -> list[dict[str, Any]]:
    """
    EXCLUSIVE search: only finds compound structures 
    STRICTLY items provided.
    """
    # Cleaning and sorting
    elements = sorted([el.strip() for el in elements_list if el.strip()])
    num_el = len(elements)
    
    base_url = "https://www.crystallography.net/cod/result.php"
    params = {
        'el[]': elements,
        'strictmin': num_el, # Minimum n elements
        'strictmax': num_el, # Maximum n elements
        'format': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code != 200: return []
        
        results = response.json()
        if not isinstance(results, list): return []

        cleaned_results = []
        # Define a set of our elements in capital letters for comparison
        target_set = set(el.upper() for el in elements)

        for r in results:
            formula = r.get('formula', '')
            if not formula: continue
            
            # Extracting the elements of the formula returned by the COD
            # "Ca1 H2 O2" -> ['Ca', 'H', 'O']
            found_elements = set(formula_to_elements(formula))
            found_set_upper = set(el.upper() for el in found_elements)

            # Exclusive filter
            if found_set_upper == target_set:
                cleaned_results.append({
                    'id': r.get('id') or r.get('file'),
                    'formula': formula,
                    'spacegroup': r.get('sg', 'N/A'),
                    'cell': f"a={r.get('a', '?')} b={r.get('b', '?')} c={r.get('c', '?')}",
                    'common_name': r.get('mineral') or r.get('compound') or 'N/A',
                    'doi': r.get('doi', 'N/A') # add the DOI here
                })
                
        return cleaned_results

    except Exception as e:
        print(f"Search error: {e}")
        return []
    
def cod_search_by_name(mineral_name: str) -> list[dict[str, Any]]:
    """
    Search the COD by mineral name or free text.
    Returns a formatted list identical to advanced_cod_search.
    """
    base_url = "https://www.crystallography.net/cod/result.php"
    params = {
        'text': mineral_name,
        'format': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            try:
                results = response.json()
            except:
                return []

            if not isinstance(results, list):
                return []

            cleaned_results = []
            for r in results:
                # Use the same format as advanced cod search
                cleaned_results.append({
                    'id': r.get('id') or r.get('file'),
                    'formula': r.get('formula', 'N/A'),
                    'spacegroup': r.get('sg', 'N/A'),
                    # Make sure to have default values ​​if a, b or c are missing
                    'cell': f"a={r.get('a', '?')} b={r.get('b', '?')} c={r.get('c', '?')}",
                    'common_name': r.get('mineral') or r.get('compound') or 'N/A',
                    'doi': r.get('doi', 'N/A')
                })
            return cleaned_results
            
    except Exception as e:
        print(f"Connexion error (Name Search): {e}")
        
    return []

def cod_search_by_id(cod_id: int) -> list[dict[str, Any]]:
    """Performs a direct search on a specific ID."""
    base_url = "https://www.crystallography.net/cod/result.php"
    params = {'id': cod_id, 'format': 'json'}
    try:
        r = requests.get(base_url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Format to match 'cleaned_results' format
            return [{
                'id': d.get('id'),
                'formula': d.get('formula', 'N/A'),
                'spacegroup': d.get('sg', 'N/A'),
                'cell': f"a={d.get('a')} b={d.get('b')} c={d.get('c')}",
                'common_name': d.get('mineral') or d.get('compound') or 'N/A'
            } for d in data] if isinstance(data, list) else []
    except: return []
    return []

def get_structure_by_cod_id(cod_id: int) -> AtomicSystem:
    """
    Retrieves a structure without the need for MySQL.
    We download the raw CIF and parse it with Pymatgen.
    """
    # Direct URL of CIF file
    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Check if the download was successful
        
        # We use StringIO so that CifParser believes that it is a file
        parser = CifParser(StringIO(response.text))
        structure = parser.parse_structures()[0]
        return structure
        
    except Exception as e:
        print(f"Error retrieving COD {cod_id}: {e}")
        return None


class CODBrowserDialog(BaseBuilderDialog):
    def __init__(self, parent:AtomViewerGUI =None):
        # Use the BaseBuilderDialog constructor (parent, title, width)
        super().__init__(parent, "COD Database Explorer", 850)
        self.setMinimumHeight(600)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        gui_dir = os.path.dirname(current_script_dir)
        self.cache_file = os.path.join(gui_dir, "cod_cache.json")
        
        self.selected_system = None 
        self.last_cod_id = None
        self.results_data = [] 
        
        self.setup_ui()

        QtCore.QTimer.singleShot(100, self.load_last_search)

    def setup_ui(self) -> None:
        # Use self.main_layout provided by BaseBuilderDialog
        layout = self.main_layout

        # Search section
        search_group = QtWidgets.QGroupBox("Search in COD")
        search_layout = QtWidgets.QVBoxLayout(search_group)
        
        top_row = QtWidgets.QHBoxLayout()
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems(["Mineral Name", "Chemical Formula", "COD ID"])
        self.combo_type.setFixedWidth(150)
        
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Enter name, formula or ID...")
        self.search_input.returnPressed.connect(self.run_search)
        
        self.btn_search = self.create_icon_button("Search", "search", primary=True)
        self.btn_search.clicked.connect(self.run_search)
        
        top_row.addWidget(self.combo_type)
        top_row.addWidget(self.search_input)
        top_row.addWidget(self.btn_search)
        search_layout.addLayout(top_row)
        layout.addWidget(search_group)

        # Result table
        self.table = self.create_table([
            "COD ID", "Common Name", "Formula", "Space Group", "Cell", "Ref"
        ], selectable=True)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(0, 100) 
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(3, 100) 
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(5, 50)
        
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.doubleClicked.connect(self.load_selected)

        self.table.itemSelectionChanged.connect(lambda: self.btn_import.setEnabled(True))
        self.table.setSortingEnabled(True)
        
        layout.addWidget(self.table)

        # Action section
        bottom_layout = QtWidgets.QHBoxLayout()
        
        self.btn_import = self.create_action_button("Open selected structure", primary=True)
        self.btn_import.setMinimumHeight(35)
        self.btn_import.setEnabled(False)
        self.btn_import.clicked.connect(self.load_selected)
        
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_import)

        layout.addLayout(bottom_layout)

        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.showMessage("Ready")
        layout.addWidget(self.status_bar)

    def run_search(self) -> None:
        query = self.search_input.text().strip()
        if not query: return

        search_mode = self.combo_type.currentText()
        self.table.setRowCount(0)
        self.status_bar.showMessage(f"⏱ Searching in COD for '{query}'...")
        QtCore.QCoreApplication.processEvents() # to show the message

        try:
            results = []
            if search_mode == "Mineral Name":
                results = cod_search_by_name(query)
            elif search_mode == "Chemical Formula":
                elements = formula_to_elements(query)
                results = cod_search_by_elements(elements)
            elif search_mode == "COD ID":
                results = cod_search_by_id(query)
            
            if not results:
                self.status_bar.showMessage("No results found")
            
            self.display_results(results)
            self.save_last_search(query, search_mode, results)

        except Exception as e:
            self.status_bar.showMessage("⚠ Error during search")
            QtWidgets.QMessageBox.critical(self, "Search Error", str(e))
        finally:
            self.status_bar.showMessage(f"{len(results)} structures found")

    def save_last_search(self, 
                         query: str, 
                         mode: str, 
                         results: list[dict[str, Any]]) -> None:
        """Saves the search to a local JSON file."""
        data = {
            "query": query,
            "mode": mode,
            "results": results
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving JSON cache: {e}")

    def load_last_search(self) -> None:
        """Reads the JSON file and populates the interface."""
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            query = data.get("query", "")
            mode = data.get("mode", "Mineral Name")
            results = data.get("results", [])

            if query:
                self.search_input.setText(query)
                self.combo_type.setCurrentText(mode)
                self.results_data = results
                self.display_results(results)
                self.status_bar.showMessage(f"Last search restored ({len(results)} results)")
        except Exception as e:
            print(f"Error loading JSON cache: {e}")

    def display_results(self, results: dict[str, Any]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        for r in results:
            row = self.table.rowCount()
            self.table.insertRow(row)

            id_item = QtWidgets.QTableWidgetItem(str(r.get('id', '')))
            id_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 0, id_item)
            
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(r.get('common_name', 'N/A')))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(r.get('formula', 'N/A')))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(r.get('spacegroup', 'N/A')))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(r.get('cell', 'N/A')))

            doi = r.get('doi')
            if doi and doi not in ['N/A', 'None']:
                container = QtWidgets.QWidget()
                btn_layout = QtWidgets.QHBoxLayout(container)
                btn_layout.setContentsMargins(0, 0, 0, 0)
                btn_layout.setAlignment(QtCore.Qt.AlignCenter)

                btn_ref = self.create_icon_button("", "reference")
                btn_ref.setIconSize(QtCore.QSize(18, 18))
                btn_ref.setFixedSize(24, 24)
                btn_ref.setFlat(True)
                btn_ref.setCursor(QtCore.Qt.PointingHandCursor)
                btn_ref.setToolTip(f"Open DOI: {doi}")
                
                doi_url = f"https://doi.org/{doi}"
                btn_ref.clicked.connect(lambda checked=False, url=doi_url: webbrowser.open(url))
                
                btn_layout.addWidget(btn_ref)
                self.table.setCellWidget(row, 5, container)
            else:
                item_dash = QtWidgets.QTableWidgetItem("-")
                item_dash.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, 5, item_dash)

        self.table.setSortingEnabled(True)

    def load_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0: return
        
        self.last_cod_id = self.table.item(row, 0).text()
        
        try:
            structure = get_structure_by_cod_id(self.last_cod_id)
            if structure:
                self.selected_system = structure
                self.accept()
            else:
                raise ValueError("Could not parse CIF.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to load CIF: {e}")