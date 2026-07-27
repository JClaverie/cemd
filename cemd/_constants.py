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

import numpy as np

AVOGADRO = 6.02214076e23

MASSES_DICT = {
    'H' : 1.007947,    'He': 4.0026022,   'Li': 6.9412,      'Be': 9.0121823,
    'B' : 10.8117,     'C' : 12.01078,    'N' : 14.00672,    'O' : 15.99943,
    'F' : 18.99840325, 'Ne': 20.17976,    'Na': 22.989769282,'Mg': 24.30506,
    'Al': 26.981538613,'Si': 28.08553,    'P' : 30.9737622,  'S' : 32.0655,
    'Cl': 35.4532,     'Ar': 39.9481,     'K' : 39.09831,    'Ca': 40.0784,
    'Sc': 44.9559126,  'Ti': 47.8671,     'V' : 50.94151,    'Cr': 51.99616,
    'Mn': 54.9380451,  'Fe': 55.8452,     'Co': 58.9331955,  'Ni': 58.69344,
    'Cu': 63.5463,     'Zn': 65.382,      'Ga': 69.7231,     'Ge': 72.6308,
    'As': 74.921602,   'Se': 78.963,      'Br': 79.9041,     'Kr': 83.7982,
    'Rb': 85.46783,    'Sr': 87.621,      'Y' : 88.905852,   'Zr': 91.2242,
    'Nb': 92.906382,   'Mo': 95.962,      'Tc': 98.0,        'Ru': 101.072,
    'Rh': 102.905502,  'Pd': 106.421,     'Ag': 107.86822,   'Cd': 112.4118,
    'In': 114.8181,    'Sn': 118.7107,    'Sb': 121.7601,    'Te': 127.603,
    'I' : 126.904473,  'Xe': 131.2936,    'Cs': 132.90545196,
    'Ba': 137.3277,    'D' : 2.01410178,  'T' : 3.01604928
}

CHARGES_DICT = {
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "B": 3, "Al": 3, "Ga": 3, "In": 3,
    "C": 4, 
    "Si": 4, 
    "Ge": 4,
    "Sn": 2,
    "Pb": 2,
    "N": -3, "P": -3, "As": -3, "Sb": 3, "Bi": 3,
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 3, "Mn": 2,
    "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2, "Zn": 2,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Ag": 1,
    "Cd": 2, "W": 6, "Pt": 4, "Au": 3, "Hg": 2,
    "La": 3, "Ce": 4, "Nd": 3, "Gd": 3, "Yb": 3
}

INV_MASSES = {float(v): k for k, v in MASSES_DICT.items()}
MASS_KEYS = np.array(list(INV_MASSES.keys()))