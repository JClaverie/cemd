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
import webbrowser
import requests
from PySide6 import QtWidgets, QtCore
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from main_window import AtomViewerGUI

from cemd.gui.ui.base_dialog import BaseBuilderDialog
from cemd.core.atomic_system import AtomicSystem


def pubchem_search_by_name(name: str) -> list[dict[str, Any]]:
    """Search for CIDs by common name or IUPAC."""
    base_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/description/json"
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code != 200: return []
        
        data = response.json()
        results = []
        for info in data.get('InformationList', {}).get('Information', []):
            cid = info.get('CID')
            if cid:
                # We retrieve the details (Formula, Weight) for display
                details = get_pubchem_details(cid)
                results.append({
                    'id': str(cid),
                    'common_name': info.get('Title', name),
                    'formula': details.get('formula', 'N/A'),
                    'weight': details.get('weight', 'N/A'),
                    'smiles': details.get('smiles', 'N/A')
                })
        return results
    except: return []
    

def pubchem_search_by_formula(formula: str) -> list[dict[str, Any]]:
    """Search for CIDs by chemical formula."""
    base_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/{formula}/listkey/json"
    try:
        # Formula search is asynchronous on PubChem (ListKey)
        response = requests.get(base_url, timeout=10)
        if response.status_code != 200: return []
        
        listkey = response.json().get('IdentifierList', {}).get('ListKey')
        # We retrieve the first 20 results for the responsiveness of the UI
        fetch_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listkey}/cids/json?maxrecords=20"
        fetch_res = requests.get(fetch_url, timeout=10)
        cids = fetch_res.json().get('IdentifierList', {}).get('CID', [])

        results = []
        for cid in cids:
            details = get_pubchem_details(cid)
            results.append({
                'id': str(cid),
                'common_name': details.get('title', 'N/A'),
                'formula': formula,
                'weight': details.get('weight', 'N/A'),
                'smiles': details.get('smiles', 'N/A')
            })
        return results
    except: return []

def get_pubchem_details(cid: int) -> dict:
    """Retrieves properties. If SMILES is missing, performs a dedicated fallback request."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,Title,IUPACName,CanonicalSMILES/json"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return {}
            
        props = r.json().get('PropertyTable', {}).get('Properties', [{}])[0]
        smiles = props.get('ConnectivitySMILES')

        if not smiles:

            fallback_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/json"
            r_s = requests.get(fallback_url, timeout=3)
            if r_s.status_code == 200:
                s_props = r_s.json().get('PropertyTable', {}).get('Properties', [{}])[0]
                smiles = s_props.get('ConnectivitySMILES')

        return {
            'formula': props.get('MolecularFormula'),
            'weight': props.get('MolecularWeight'),
            'smiles': smiles,
            'title': props.get('Title'),
            'iupac': props.get('IUPACName')
        }
    except:
        return {}

def get_structure(cid: int, smiles: str = None) -> dict | None:
    """
    Tries to fetch 3D SDF, then 2D SDF. 
    If both fail (404), generates structure from SMILES using read_smiles.
    """
    from cemd.core._io import IOMixin
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"

    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        print(f"Loaded 3D SDF for CID {cid}")
        return IOMixin.read_sdf_content(response.text)

    if smiles:
        print(f"SDF not found for {cid}. Generating structure from SMILES...")
        try:
            return IOMixin.read_smiles(smiles)
        except Exception as e:
            print(f"SMILES conversion error: {e}")

    else:
        print("Failer to find structure.")

    return None

def pubchem_sdq_search(query: str, limit: int = 50) -> list[dict]:
    """
    Retrieves exact results from PubChem site via SDQ API.
    """
    # Query cleanup and word separation for "AND"
    words = [w.strip() for w in query.split() if w.strip()]
    if not words:
        return []

    # Construction of the 'where' clause (identical to your URL)
    ands = [{"*": word} for word in words]

    query_params = {
        "download": "*",
        "collection": "compound",
        "order": ["relevancescore,desc"],
        "start": 1,
        "limit": limit,
        "where": {"ands": ands}
    }

    params = {
        "infmt": "json",
        "outfmt": "json",
        "query": json.dumps(query_params)
    }

    url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sphinxql.cgi"

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status() # Throw an error if the query fails
        
        data = response.json()
        
        results = []
        for item in data:
            # We map the keys from JSON SDQ to your table format
            results.append({
                'id': str(item.get('cid', '')),
                'common_name': item.get('cmpdname', 'N/A'),
                'formula': item.get('mf', 'N/A'),
                'weight': str(item.get('mw', 'N/A')),
                'smiles': item.get('smiles', 'N/A'),
                'iupac': item.get('iupacname', 'N/A')
            })
        return results

    except Exception as e:
        print(f"Error querying SDQ: {e}")
        return []


class PubChemBrowserDialog(BaseBuilderDialog):
    def __init__(self, parent: AtomViewerGUI = None):
        super().__init__(parent, "PubChem Database Explorer", 850)
        self.setMinimumHeight(600)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_file = os.path.join(os.path.dirname(current_dir), "pubchem_cache.json")
        
        self.selected_system = None 
        self.setup_ui()
        QtCore.QTimer.singleShot(100, self.load_last_search)

    def setup_ui(self) -> None:
        layout = self.main_layout

        # Filters
        search_group = QtWidgets.QGroupBox("Search in PubChem")
        search_layout = QtWidgets.QVBoxLayout(search_group)
        
        top_row = QtWidgets.QHBoxLayout()
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems(["Compound Name or Formula", "PubChem CID"])
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

        # Table
        self.table = self.create_table([
            "CID", "Name", "Formula", "Mol. Weight", "SMILES", "Ref."
        ], selectable=True)
        
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(5, 50)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.doubleClicked.connect(self.load_selected)
        layout.addWidget(self.table)

        # Actions
        bottom_layout = QtWidgets.QHBoxLayout()
        self.btn_import = self.create_action_button("Open selected structure", primary=True)
        self.btn_import.clicked.connect(self.load_selected)
        self.btn_import.setEnabled(False)
        
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_import)
        layout.addLayout(bottom_layout)

        self.table.itemSelectionChanged.connect(lambda: self.btn_import.setEnabled(True))

        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.status_bar.showMessage("Ready")
        layout.addWidget(self.status_bar)

    # def run_search(self) -> None:
    #     query = self.search_input.text().strip()
    #     if not query: return

    #     mode = self.combo_type.currentText()
    #     self.status_bar.showMessage(f"⏱ Searching in PubChem for '{query}'...")
    #     QtCore.QCoreApplication.processEvents()

    #     try:
    #         results = []
    #         if mode == "Compound Name or Formula":
    #             results = pubchem_search_by_name(query)
    #         elif mode == "PubChem CID":
    #             details = get_pubchem_details(query)
    #             if details:
    #                 results = [{'id': query, 'common_name': details.get('title'), 'formula': details.get('formula'), 
    #                             'weight': details.get('weight'), 'smiles': details.get('smiles')}]

    #         self.display_results(results)
    #         self.save_last_search(query, mode, results)
    #         self.status_bar.showMessage(f"{len(results)} compounds found")
    #     except Exception as e:
    #         self.status_bar.showMessage("⚠ Error during search")
    #         QtWidgets.QMessageBox.critical(self, "Search Error", str(e))

    def run_search(self) -> None:
        query = self.search_input.text().strip()
        if not query: return

        mode = self.combo_type.currentText()
        self.status_bar.showMessage(f"⏱ Searching in PubChem for '{query}'...")
        # We force the refresh of the UI to display the message
        QtCore.QCoreApplication.processEvents()

        try:
            results = []
            if mode == "Compound Name or Formula":
                results = pubchem_search_by_name(query)
                if len(results) == 0:
                   results = pubchem_sdq_search(query) 
            
            elif mode == "PubChem CID":
                # Direct search by ID (we keep your details logic)
                details = get_pubchem_details(query)
                if details:
                    results = [{
                        'id': query, 
                        'common_name': details.get('title', 'N/A'), 
                        'formula': details.get('formula', 'N/A'), 
                        'weight': details.get('weight', 'N/A'), 
                        'smiles': details.get('smiles', 'N/A')
                    }]

            # Display and feedback
            if results:
                self.display_results(results)
                self.save_last_search(query, mode, results)
                self.status_bar.showMessage(f"{len(results)} compounds found")
            else:
                self.table.setRowCount(0)
                self.status_bar.showMessage(f"No results found for '{query}'")

        except Exception as e:
            self.status_bar.showMessage("⚠ Error during search")
            QtWidgets.QMessageBox.critical(self, "Search Error", f"Search failed: {str(e)}")

    def display_results(self, results: list) -> None:
        self.table.setRowCount(0)
        for r in results:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(r['id'])))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(r['common_name']))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(r['formula']))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(r['weight'])))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(r['smiles']))

            # Web link button
            btn_web = self.create_icon_button("", "reference") # Reuses COD icon
            url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{r['id']}"
            btn_web.clicked.connect(lambda chk=False, u=url: webbrowser.open(u))
            self.table.setCellWidget(row, 5, btn_web)

    def load_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0: return
        
        cid = self.table.item(row, 0).text()
        smiles = self.table.item(row, 4).text()
        self.status_bar.showMessage(f"Fetching 3D coordinates for CID {cid}...")
        topo = get_structure(cid, smiles)
        
        if topo:
            self.selected_system = AtomicSystem.from_dict(topo)
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "3D Error", "No 3D conformer available for this compound on PubChem.")

    def save_last_search(self, query, mode, results):
        try:
            with open(self.cache_file, "w") as f:
                json.dump({"query": query, "mode": mode, "results": results}, f)
        except: pass

    def load_last_search(self):
        if not os.path.exists(self.cache_file): return
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                self.search_input.setText(data['query'])
                self.combo_type.setCurrentText(data['mode'])
                self.display_results(data['results'])
        except: pass