from core._topology import set_topology_rule
from cemd.core._io import read_mda, to_pmg
from cemd.builders.hydrates import make_csh, pycsh, calculate_csh_modifiers, csh_to_cash, substitute_si_by_al
from cemd.builders.base import build_surfaces, add_liquid, build_solution, protonate, split, add_droplet
from cemd.core.atomic_system import AtomicSystem

from pymatgen.core.surface import SlabGenerator

# con_dict = {
#     "Na": 2,
#     "Cl":2
# }

# a = AtomicSystem.from_file("tob11a_hamid.cif")
# a.replicate([3,5,1])
# nsi = a.get_count("Si")
# nca = a.get_count("Ca")
# print(a)

# res = _calculate_csh_modifiers(nsi,nca, 1.5, 3)
# print(res)

# b = make_csh(1.4, 1.1)
# b.write("csh_15.data")

a = AtomicSystem.from_file("csh_15.data")

c = csh_to_cash(a, 0.1)
print(c)

# a = AtomicSystem.from_file("calcite_slab_001.data")
# b = add_liquid(a, 20)
# print(b)

# a = AtomicSystem('cs1.5_ws1.2_1.data')
# print(a.atom_types)

# data = AtomicSystem('test.data')
# data.set_types(['Hw', 'Ow'])

# print(data)
# data = AtomicSystem('calcite.cif')
# struct = data.to_pmg()
# struct.add_oxidation_state_by_guess()
# slabgen = SlabGenerator(struct, [0,0,1], min_slab_size=25, min_vacuum_size=15)
# slablist = slabgen.get_slabs()
# for s in slablist:
#     print(s.shift)

# bonds_dic = {
#     ('Al4+', 'O2-') : 1.8,
#     ('Al', 'O'): 1.8,
#     ('Si4+', 'O2-') : 1.8,
#     ('Si', 'O') : 1.8,
#     ('C4+', 'O2-') : 1.4,
#     ('C', 'O') : 1.4, 
#     ('H+', 'O2-') : 1.1,
#     ('H', 'O') : 1.1
# }


# slabs_list = slabgen.get_slabs(bonds=bonds_dic, max_broken_bonds=0)

# print(len(slabs_list))

# b = make_interface(a, 10)
# b.show()



# print(a.masses)
# a.remove_atoms([1])
# print(a)


# a.set_types(['Ca', 'Ca', 'B', 'A', 'C', 'D', 'HA', 'HG'])
# a.reset_types()
# print(a)
