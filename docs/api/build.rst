Build Module
===============

The **build** module provides the core functionality for constructing atomistic systems from scratch.
It acts as a high-level interface to generate crystalline surfaces, aqueous solutions, 
and complex interfaces suitable for LAMMPS simulations.

Builders
--------
Advanced builders for creating and modifying complex structures.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cemd.build.SolutionBuilder
   cemd.build.SurfaceBuilder
   cemd.build.Splitter
   cemd.build.GlassBuilder
   cemd.build.CSHBuilder
   cemd.build.AFBuilder