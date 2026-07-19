# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.cif import CifWriter
import numpy as np

#%%

plan = [0,0,1]
struct = Structure.from_file("mumme_m3.cif")
struct.add_oxidation_state_by_guess()
slabgen = SlabGenerator(struct, plan, 15, 10)

i = 0
nb = 0
while nb == 0:
    slabs = slabgen.get_slabs(bonds={("Si4+", "O2-"):1.8}, tol=0.1, 
                                     max_broken_bonds=i)
    nb = len(slabs)
    i += 1

d0 = 1000
for s in slabs:
    d = abs(s.dipole[2])
    if d <= d0:
        slab = s.get_sorted_structure(key = lambda site:site.c)
        d0 = d

print("Number of broken bonds: {}".format(i-1))

w_cif = CifWriter(slab)
w_cif.write_file("c3s_{}{}{}.cif".format(plan[0], plan[1], plan[2]))

# Get slab information
nlayers_total = int(round(slab.lattice.c /
                          slab.oriented_unit_cell.lattice.c))
nlayers_slab = int(round((slab.sites[-1].c - slab.sites[0].c)
                         * nlayers_total))
t = (nlayers_slab / nlayers_total) * slab.lattice.abc[2]

print("Number of slabs in selection: {}".format(nb))
print("Miller Index: ({} {} {})".format(plan[0], plan[1], plan[2]))
print("Slab size: {:.3f} x {:.3f} x {:.3f} A^3".format(slab.lattice.abc[0], 
      slab.lattice.abc[1], t))
print("Dipole: {:.2f}e".format(d0))
