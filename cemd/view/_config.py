
VMD_RESOLUTION = 12

VMD_MATERIAL = "AOEdgy"

VMD_ELEMENT_TYPES = {
    "H": ["H", "Hw", "Hw_aq", "Hw_s", "Hh", "Hh_aq", "Hh_s", "H", "Hsi", "Ha"],
    "O": ["O", "Ow", "Ow_aq", "Ow_s", "Oh", "Oh_aq", "Oh_s", "Ob", "Obs", "Osi", "Osih", "Oa", "Oah", "Oc"],
    "Ca": ["Ca", "Cw", "Ca_s", "Ca_aq"],
    "Si": ["Si"],
    "Al": ["Al"],
    "Na": ["Na"],
    "Cl": ["Cl"],
    "K": ["K"],
}

VMD_ELEMENT_COLORS = {
    "H": "#FFFFFF",
    "C": "#808080",
    "O": "#FF0000",
    "Si": "#F5B800",
    "Al": "#8080FF",
    "S": "#FFFF00",
    "Ca": "#00E580",
    "Na": "#4500FA",
    "Cl": "#00C2FF",
    "K": "#80E666",
}

VMD_ATOM_RADIUS = {
    "Ow Hw Ow1 Hw1 O H": 0.8,
    "Hh Oh Hh1 Oh1 Oh_aq Hh_aq Oh_s Hh_s": 0.8,
    "Ob Obs Osi Hsi Osih Oa Ha Oah": 0.8,
    "S Si C Al": 0.8,
    "Os Oc": 0.8,
    "Cl Na Ca K Ca_s Ca_aq Cw": 1.8,
}

VMD_BOND_CUTOFF = {
    "Ow Hw Ow_aq Hw_aq Ow_s Hw_s": 2.0,
    "Oh Hh Oh_aq Hh_aq Oh_s Hh_s": 2.0,
    "Oah Ha Osih Hsi Oh H O Oh1 Hh1": 2.0,
    "C H": 2.0,

    # C-O bonds
    "Oc C": 1.6,
    "O C": 1.6,

    # S-O bonds
    "Os S": 1.6,
    "O S": 1.6,

    # Si-O bonds
    "Osi Si": 2.0,
    "Ob Si": 2.0,
    "Obs Si": 2.0,
    "Osih Si": 2.0,
    "O Si": 2.0,

    # Al-O bonds
    "Oa Al": 2.0,
    "Ob Al": 2.0,
    "Obs Al": 2.0,
    "Oh Al": 2.0,
    "Oah Al": 2.0,
    "O Al": 2.0,
}

VMD_MATERIAL_OPTIONS = [
    "AOEdgy",
    "AOShiny",
    "AOChalky",
]

VMD_MATERIAL_SETTINGS = {
    "outlinewidth": 0.6,
    "outline": 1.5,
    "shininess": 0.8,
}