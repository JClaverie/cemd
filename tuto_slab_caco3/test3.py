# test.py
from cemd import AtomicSystem
from cemd.build import SolutionBuilder

# 1. Charger la surface
system = AtomicSystem.from_file("surface_104_super.data")
# print(f"Surface: {system.num_atoms} atomes")
# print(f"Boîte: {system.box}")

print(system.atoms)

# system.replicate([2,2,1])
# builder = SolutionBuilder.from_water(density=1.0)

# system = system.add_droplet(builder, 20)
# print(system)

aspirin = AtomicSystem.from_file("aspirin.data")
system.add_structure(aspirin)

# # 3. Ajouter la couche
# print("Ajout de la couche d'eau...")
# # system = system.add_layer(builder, thickness=10.0, distance=2.0)
# print(f"Système final: {system.num_atoms} atomes")
# print(f"Boîte finale: {system.box}")

# # 4. Visualiser
system.view()