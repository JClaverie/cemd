from cemd.build import CSHBuilder

builder = CSHBuilder(1.2, 0.85)

print(builder)

system = builder.build()

print(system)

system.view()