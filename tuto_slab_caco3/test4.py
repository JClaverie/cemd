from cemd import AtomicSystem
from cemd.build import SolutionBuilder, Splitter

system = AtomicSystem.from_file("csh_12_reacted.data")
system.reset_types()

builder = SolutionBuilder.from_water()
splitter = Splitter(system, coordinate=10)
splitter.add_solution(builder)

system_splitted = splitter.split()
system.set_topo("cshff")
print(system_splitted)

system_splitted.view()
