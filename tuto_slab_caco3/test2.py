from cemd import AtomicSystem
from cemd.build import SurfaceBuilder

system = AtomicSystem.from_file("9016705.cif")

builder = SurfaceBuilder(system)

surface = builder.build(
    (1, 0, 4), min_slab_size=25
)

# surface2 = builder.build2(
#     (1, 0, 4)
# )

# # print(surface)

print(surface)