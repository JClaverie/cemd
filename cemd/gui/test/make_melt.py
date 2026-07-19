
from cemd.builders.base import build_glass

# 2 compositions from Lolli2018:
# 3SiO2·2Al2O3·2Na2O·6H2O, M = 616.22 g/mol
# 4SiO2·2Al2O3·2Na2O·6H2O, M = 676.31 g/mol
# 6H2O, M = 108.09 g/mol
# Higher Si/Al?

# To reach a wet density of 2.2-2.3: 
# for 3SiO2·2Al2O3·2Na2O·6H2O: d = 1.80
# for 4SiO2·2Al2O3·2Na2O·6H2O: d = 1.85

comp1 = {
    "Si" : 3,
    "Al" : 4,
    "Na" : 4,
    "O" : 14,
}

comp2 = {
    "Si" : 4,
    "Al" : 4,
    "Na" : 4,
    "O" : 16,
    # "H2O": 6
}

comp = {
    "SiO2": 3,
    "Al2O3": 2,
    "Na2O": 2,
    "H2O": 6,
}

data = build_glass([20, 20, 20], 2.3, comp)

print(data)



# for comp, name, d in zip([comp1, comp2], ["sa0.75", "sa1"], [2.3, 2.3]):
#     model.make_melt([30, 30, 30], comp, d, output_datafile='{}.data'.format(name))

# for d in [2.3, 2.5]:
#     model.make_melt([20, 20, 20], comp2, d, fout=f'sa0.75_{d}.data')

#     model.keep_topo(f'sa0.75_{d}.data')
#     type_list = ['algp', 'nagp', 'ogp', 'sigp']

#     model.make_ff(f'sa0.75_{d}.data', type_list, mix=None, fout="ff2.lmp")