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

import json
import webbrowser
from typing import Any

import requests
import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout import ScrollOffsets
from prompt_toolkit.layout.controls import FormattedTextControl

from ..core.atomic_system import AtomicSystem

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

def get_structure(cid: int, smiles: str = None) -> AtomicSystem | None:
    """Fetch 3D structure from PubChem, fallback to SMILES generation."""
    from cemd.core._io import IOMixin
    from cemd.core.atomic_system import AtomicSystem

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        topo = IOMixin.read_sdf_content(response.text)
        return AtomicSystem.from_dict(topo) if topo else None

    if smiles:
        try:
            topo = IOMixin.read_smiles(smiles)
            return AtomicSystem.from_dict(topo) if topo else None
        except Exception as e:
            print(f"SMILES conversion error: {e}")

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
    
def explore_pubchem() -> Any:
    """Interactive PubChem Explorer aligned with the COD interface."""
    
    query = questionary.text("Enter name or formula:").ask()
    if not query: return None

    print(f"Searching for '{query}' on PubChem...")
    results = pubchem_sdq_search(query)

    if not results:
        print("No results found.")
        return None

    index = 0
    selected = None
    kb = KeyBindings()

    # Formatting suitable for PubChem columns
    # ID(10) | NAME(30) | FORMULA(15) | WEIGHT(10)
    ROW_FORMAT = "{cursor} {id:>10} | {name:<30.30} | {formula:<15.15} | {weight:>10}"

    def render():
        header = ROW_FORMAT.format(
            cursor=" ", id="ID", name="NAME", formula="FORMULA", weight="WEIGHT"
        )
        lines = [
            f"Found {len(results)} structures",
            "↑↓ Move   [Enter] Load   [p] PubChem page   [q] Quit",
            "",
            header, "-" * len(header)
        ]

        for i, r in enumerate(results):
            cursor = "➜" if i == index else " "
            lines.append(ROW_FORMAT.format(
                cursor=cursor,
                id=r['id'],
                name=r['common_name'],
                formula=r['formula'],
                weight=r['weight']
            ))
        return "\n".join(lines)

    # Window configuration for scrolling
    window = Window(
        content=FormattedTextControl(render),
        always_hide_cursor=True,
        scroll_offsets=ScrollOffsets(top=1, bottom=1),
        allow_scroll_beyond_bottom=False
    )

    @kb.add("up")
    def _(e):
        nonlocal index
        if index > 0: index -= 1
        e.app.invalidate()

    @kb.add("down")
    def _(e):
        nonlocal index
        if index < len(results) - 1: index += 1
        e.app.invalidate()

    @kb.add("enter")
    def _(e):
        nonlocal selected
        selected = results[index]
        e.app.exit()

    # Hotkey 'p' to open PubChem page
    @kb.add("p")
    def _(e):
        cid = results[index]['id']
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}")

    @kb.add("q")
    @kb.add("escape")
    def _(e): e.app.exit()

    Application(layout=Layout(HSplit([window])), key_bindings=kb).run()

    if selected:
        print(f"Loading CID {selected['id']}...")
        return get_structure(int(selected['id']), selected['smiles'])
    return None