Using the Builders Module
=========================

The **builders** module is the central engine of the **cemd** package. Its primary purpose is to automate the construction of complex atomistic systems, from simple crystal bulks to intricate solid-liquid interfaces.

Workflow Overview
-----------------
Building a simulation-ready system generally follows these three steps:

1. **Initialization**: Load an existing structure or define composition parameters.
2. **Construction**: Generate the system (e.g., creating a liquid box or cutting a surface).
3. **Refinement**: Repair broken bonds or add functional groups (like silanols).

Construction Examples
---------------------

### Building a Liquid Solution
To create a water-based solution with solutes, you can use the `build_solution` function. It automatically calculates the number of solvent molecules required to reach a specific density.

.. code-block:: python

   from cemd.builders.base import build_solution

   # Create a solution box of 30x30x30 Angstroms with a density of 1.0 g/cm³
   # containing 10 ions of a custom type 'Na' and 'Cl'
   solution = build_solution(
       box=[30, 30, 30],
       density=1.0,
       solutes_dict={'Na': 10, 'Cl': 10}
   )

### Building Complex C-S-H Models
For cementitious materials, the `hydrates` submodule provides the `pycsh` tool, which generates structures based on specific Calcium/Silicate (C/S) and Water/Silicate (W/S) ratios.

Example: Building a C-S-H model
-------------------------------
.. code-block:: python

   from cemd.builders.hydrates import build_csh

   # Create a C-S-H model with specific Ca/Si and Water/Si ratios
   csh_system = build_csh(cs_ratio=1.5, ws_ratio=1.5)

Creating Surfaces
-----------------
The **builders** module simplifies the generation of crystalline surfaces. The process involves cutting a bulk structure along specific crystallographic planes, defined by their Miller indices, and adding a vacuum layer to avoid interactions between periodic images.

Example: Cutting a Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~
To create a surface, use the `build_surface` function. You need to provide the bulk structure and the Miller indices for the desired cut.

.. code-block:: python

   from cemd.builders.base import build_surface

   # Generate surfaces cut along the (0, 0, 1) plane
   # min_slab_size ensures the thickness of the crystal layer
   # min_vacuum_size adds empty space along the normal axis
   surfaces, shifts, dipoles, broken_bonds = build_surface(
       data=my_bulk_structure,
       miller_indices=[0, 0, 1],
       min_slab_size=20.0,
       min_vacuum_size=15.0
   )

   # Select the first generated surface
   my_surface = surfaces[0]

Understanding the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Miller Indices**: These define the orientation of the cutting plane. Common cuts include [1, 0, 0], [1, 1, 0], or [1, 1, 1].
* **min_slab_size**: Defines the minimum thickness of your material. A slab too thin might not represent the bulk properties correctly.
* **min_vacuum_size**: This creates a "gap" in the periodic boundary conditions. This is essential for surface studies to prevent the surface from interacting with its own periodic image in the simulation.
* **Broken Bonds**: The `build_surface` function automatically reports how many bonds were broken during the cut. This is a helpful diagnostic metric to ensure your structure is stable.

Creating a Solid/Liquid interface
---------------------------------
Once you have your solid and your liquid, you can combine them using the interface tools.

.. code-block:: python

   from cemd.builders.base import add_liquid

   # Create an interface between a solid surface and a liquid layer
   interface = add_liquid(
       solid_system=my_surface,
       thickness=20.0,
       density=1.0
   )

.. note::
   Many of these functions rely on **Packmol** for the placement of molecules. Ensure that Packmol is installed and correctly configured in your system's PATH.