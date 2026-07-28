
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
    # Non-metals
    "H": "#FFFFFF",
    "C": "#808080",
    "O": "#FF0000",
    "N": "#304FF0",
    "S": "#FFFF00",
    "P": "#FF8000",
    
    # Metalloids
    "Si": "#F5B800",
    "B": "#FFB6C1",
    
    # Alkali metals
    "Na": "#4500FA",
    "K": "#80E666",
    "Li": "#B22222",
    "Rb": "#702670",
    "Cs": "#8B008B",
    
    # Alkaline earth metals
    "Ca": "#00E580",
    "Mg": "#8B4513",
    "Ba": "#00BFFF",
    "Sr": "#FF6347",
    
    # Transition metals
    "Fe": "#FF4500",
    "Al": "#8080FF",
    "Ti": "#808080",
    "Mn": "#9B30FF",
    "Cr": "#00FF7F",
    "Ni": "#A9A9A9",
    "Cu": "#FF6347",
    "Zn": "#7F7FFF",
    "Zr": "#94E5FF",
    "V": "#FF69B4",
    "Co": "#FF1493",
    "Mo": "#C0C0C0",
    "W": "#B0C4DE",
    
    # Noble metals
    "Au": "#FFD700",
    "Ag": "#C0C0C0",
    "Pt": "#E5E4E2",
    "Pd": "#D3D3D3",
    
    # Lanthanides
    "Ce": "#FFFFE0",
    "Eu": "#FFE4B5",
    
    # Others
    "Pb": "#C0C0C0",
    "U": "#00FF00",
    "Th": "#00BFFF",
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
    "Ow Hw Ow_aq Hw_aq Ow_s Hw_s": 1.1,
    "Oh Hh Oh_aq Hh_aq Oh_s Hh_s": 1.1,
    "Oah Ha Osih Hsi Oh H O Oh1 Hh1": 1.1,
    "C H": 1.1,

    # C-O bonds
    "Oc C": 1.6,
    "O C":  1.6,

    # S-O bonds
    "Os S": 1.6,
    "O S":  1.6,

    # Si-O bonds
    "Osi Si":   1.75,
    "Ob Si":    1.75,
    "Obs Si":   1.75,
    "Osih Si":  1.75,
    "O Si":     1.75,

    # Al-O bonds
    "Oa Al":    1.75,
    "Ob Al":    1.75,
    "Obs Al":   1.75,
    "Oh Al":    1.75,
    "Oah Al":   1.75,
    "O Al":     1.75,
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