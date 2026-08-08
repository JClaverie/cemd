from cemd._paths import FF_DATABASE_DIR
from cemd.core._forcefield.forcefield_database import ForceFieldDatabase

database = ForceFieldDatabase(FF_DATABASE_DIR)

# print(database.bond)
# print(database.bond['raiteri2015.Ow-Ow'])
print(database.bondangle)
