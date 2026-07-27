- défilement dans les fonctions explore


cd /home/jerome/Documents/Recherche/Codes/cemd
python -c "
import sys
import traceback
try:
    import cemd.analysis.density
except Exception as e:
    traceback.print_exc()
"

python -c "from cemd.build import build_csh, build_solution, split" 2>&1
