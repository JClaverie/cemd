import numpy as np
from functools import singledispatch
import MDAnalysis as mda
from ..core.atomic_system import AtomicSystem

# Centralized Defaults
TYPES_PRESET = {
    "si_types": "Si",
    "o_types":  "O Ob Osi Osih Oa Oah Oh Ow Oc Os Obs",
    "al_types": "Al",
    "ca_types": "Ca Cw",
}

def _process_analyze(source, si_types, o_types, al_types, ca_types, cutoff):
    """Heart of calculation: identical to your old mda.Universe logic."""
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
        neighbors_o = source.select_atoms(f"around {cutoff} index {s_idx}")
        n_bridging = 0
        for o_idx in neighbors_o.indices:
            if o_idx in o_indices:
                potential_bridges = source.select_atoms(f"around {cutoff} index {o_idx}")
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
        "MCL": float(mcl),
        "Qn_distribution": pqsi
    }

@singledispatch
def analyze_silicates(source, types_map: dict = None, cutoff: float = 1.85):
    """Entry point: merge types and delegate."""
    config = TYPES_PRESET.copy()
    if types_map:
        config.update(types_map)
    
    # Call from dispatch
    func = analyze_silicates.dispatch(type(source))
    return func(source, config, cutoff)

@analyze_silicates.register(AtomicSystem)
def _(source: AtomicSystem, config=None, cutoff: float = 1.85):
    # Use TYPES_PRESET if no dictionary is provided
    config = config if config is not None else TYPES_PRESET
    u = source.to_mda()
    return _process_analyze(
        u, 
        si_types=config.get("si_types", "Si"),
        o_types=config.get("o_types", "O"),
        al_types=config.get("al_types", ""),
        ca_types=config.get("ca_types", ""),
        cutoff=cutoff
    )

@analyze_silicates.register(mda.Universe)
def _(source: mda.Universe, config=None, cutoff: float = 1.85):
    # Use TYPES_PRESET if no dictionary is provided
    config = config if config is not None else TYPES_PRESET
    return _process_analyze(
        source, 
        si_types=config.get("si_types", "Si"),
        o_types=config.get("o_types", "O"),
        al_types=config.get("al_types", ""),
        ca_types=config.get("ca_types", ""),
        cutoff=cutoff
    )