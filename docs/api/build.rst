build Module
===============

The **build** module provides the core functionality for constructing atomistic systems.
It acts as a high-level interface to generate crystalline surfaces, aqueous solutions, 
and complex interfaces suitable for LAMMPS simulations.

Build a system
--------------
Functions for creating bulk structures and liquid environments.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.build.build_csh
   cemd.build.build_solution
   cemd.build.build_glass

Generate an interface
---------------------
Tools for combining solid surfaces with liquid phases or droplets.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.build.add_structure
   cemd.build.add_liquid
   cemd.build.add_droplet
   cemd.build.build_surfaces

Split, merge, refine the structure
----------------------------------
Utilities to repair or modify atomic frameworks (e.g., silanol group addition).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.build.merge
   cemd.build.split
