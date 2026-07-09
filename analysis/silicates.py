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

import numpy as np
from functools import singledispatch
import MDAnalysis as mda
from ..core.atomic_system import AtomicSystem

@singledispatch
def analyze_silicates(source: AtomicSystem | mda.Universe, 
                      si_types: str, 
                      o_types: str, 
                      al_types: str = "", 
                      ca_types: str = "", 
                      cutoff: float = 1.85) -> dict[str: float, np.ndarray]:
    """Analyze structural properties, chemical ratios, and Q^n units of a silicate.

    Processes an atomic configuration to calculate key metrics used in cement chemistry,
    including calcium-to-silica ratios, water content, and the polymerization state 
    of silicon tetrahedra (Q^n distribution) based on bridging oxygens.

    Parameters
    ----------
    source
        The input system container, accepting either an AtomicSystem or an mda.Universe.
    si_types
        Selection string for Silicon atom types (e.g., "Si").
    o_types
        Selection string for Oxygen atom types (e.g., "O").
    al_types
        Optional selection string for Aluminum atom types (e.g., "Al").
    ca_types
        Optional selection string for Calcium atom types (e.g., "Ca").
    cutoff
        Maximum distance threshold in Angstroms to consider a stable chemical bond.
    """
    
    raise TypeError(f"Type {type(source)} is not supported by analyze_silicates.")

@analyze_silicates.register(AtomicSystem)
def _(source: AtomicSystem, 
      si_types: str, 
      o_types: str, 
      al_types: str = "", 
      ca_types: str = "", 
      cutoff: float = 1.85) -> dict[str: float, np.ndarray]:
    u = source.to_mda()
    return analyze_silicates(u, si_types, o_types, al_types, ca_types, cutoff)

@analyze_silicates.register(mda.Universe)
def _(source: mda.Universe, 
      si_types: str, 
      o_types: str, 
      al_types: str = "", 
      ca_types: str = "", 
      cutoff: float = 1.85) -> dict[str: float, np.ndarray]:
    # Secure MDAnalysis selections
    def safe_select(types_str):
        if not types_str.strip(): return None
        try: return source.select_atoms(f"type {types_str}")
        except: return None

    sel_si = safe_select(si_types)
    sel_o  = safe_select(o_types)
    sel_al = safe_select(al_types)
    sel_ca = safe_select(ca_types)
    
    try: sel_h = source.select_atoms("type H*h*")
    except: sel_h = None

    if sel_si is None or sel_o is None:
        raise ValueError("The atom types for Silicon (Si) and Oxygen (O) must be defined and valid.")

    n_si = len(sel_si)
    n_al = len(sel_al) if sel_al is not None else 0
    n_ca = len(sel_ca) if sel_ca is not None else 0
    n_h  = len(sel_h)  if sel_h  is not None else 0
    
    # Calculation of chemical stoichiometric ratios
    denom = n_si + n_al
    ca_ratio = n_ca / denom if denom > 0 else 0.0
    al_si_ratio = n_al / n_si if n_si > 0 else 0.0
    h2o_ratio = (n_h / 2) / denom if denom > 0 else 0.0

    # Polymerization Analysis (Q^n Units)
    cqsi = np.zeros(5)
    
    # Definition of network formers (Si + Al)
    al_text = f" {al_types}" if al_types.strip() else ""
    network_formers = source.select_atoms(f"type {si_types}{al_text}")
    nf_indices = set(network_formers.indices)
    o_indices = set(sel_o.indices)

    for s_idx in sel_si.indices:
        neighbors_o = source.select_atoms(f"around {cutoff} Index {s_idx}")
        n_bridging = 0
        for o_idx in neighbors_o.indices:
            if o_idx in o_indices:
                potential_bridges = source.select_atoms(f"around {cutoff} Index {o_idx}")
                # If oxygen touches ANOTHER network trainer, it's an oxygen bridge
                bridges = [idx for idx in potential_bridges.indices if idx in nf_indices and idx != s_idx]
                if len(bridges) > 0:
                    n_bridging += 1
        if n_bridging <= 4:
            cqsi[n_bridging] += 1
    
    pqsi = (cqsi / n_si) * 100

    # Calculation of Average Chain Length (MCL)
    mcl = 2 * (1 + pqsi[2] / pqsi[1]) if pqsi[1] > 1e-3 else float('In')

    return {
        "Ca/(Si+Al)": ca_ratio,
        "Al/Si": al_si_ratio,
        "H2O/(Si+Al)": h2o_ratio,
        "MCL": mcl,
        "Qn_distribution": pqsi
    }