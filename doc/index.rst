==================
cemd documentation
==================


Overview
========

The **cemd** Python package contains a **Creation Module** to create atomistic models of cementitious materials and clays in particular, including: 

- bulk and surface solids
- solutions
- liquid/solid interfaces

Data files and force field parameters can be generated for simulations with `LAMMPS <https://www.lammps.org/>`__.


Requirements
============

The following programs must be installed on your computer and accessible in the $PATH variable:

- `Packmol <https://github.com/m3g/packmol>`__

The following Python packages must be installed in your Python distribution:

- `MDAnalysis <https://www.mdanalysis.org/>`__
- `pymatgen <https://pymatgen.org/>`__

Modules
=======

.. toctree::
   :maxdepth: 2

   source/atomic_system
   source/builders
   
