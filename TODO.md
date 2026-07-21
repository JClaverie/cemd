- reset_types ne change pas les bonds, angles
- corriger read_sdf, read_smiles pour return AtomicSystem directement

Home
│
├── Installation
├── Quick Start
├── Tutorials
│     Build a crystal
│     Build a solution
│     Build an interface
│     Export to LAMMPS
│
├── User Guide
│     AtomicSystem
│     Builders
│     Force fields
│     Topology
│     Analysis
│
└── API Reference
      cemd.core
      cemd.builders
      cemd.analysis
      cemd.io


cd /home/jerome/Documents/Recherche/Codes/cemd
python -c "
import sys
import traceback
try:
    import cemd.analysis.density
except Exception as e:
    traceback.print_exc()
"

python -c "from cemd.builders import build_csh, build_solution, split" 2>&1
