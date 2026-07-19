Builders Module
===============

The **builders** module provides the core functionality for constructing atomistic systems.
It acts as a high-level interface to generate crystalline surfaces, aqueous solutions, 
and complex interfaces suitable for LAMMPS simulations.

System Construction
-------------------
Functions for creating bulk structures and liquid environments.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.builders.hydrates.pycsh
   cemd.builders.hydrates.build_csh
   cemd.builders.base.build_solution
   cemd.builders.base.build_glass

Interface Generation
--------------------
Tools for combining solid surfaces with liquid phases or droplets.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.builders.base.add_structure
   cemd.builders.base.add_liquid
   cemd.builders.base.add_droplet
   cemd.builders.base.build_surface

Structural Refinement
---------------------
Utilities to repair or modify atomic frameworks (e.g., silanol group addition).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.builders.base.rebuild_silicates
   cemd.builders.base.protonate
   cemd.builders.base.merge
   cemd.builders.base.split
