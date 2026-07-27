Using the build Module
=========================

The **build** module is the central engine of the **cemd** package. Its primary purpose is to automate the construction of complex atomistic systems, from simple crystal bulks to intricate solid-liquid interfaces.

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

   from cemd.build import build_solution, concentration2count

   # --- Configuration ---
   # Define the simulation box dimensions in Angstroms (Å)
   my_box = [30, 30, 30]

   # Define target molar concentrations for each solute species (mol/L)
   concentrations = {"Na": 1.0, "Cl": 1.0}

   # --- Step 1: Calculate Particle Counts ---
   # Convert molar concentrations into integer molecule counts based on box volume.
   # This ensures physical accuracy while adhering to the integer constraints of Packmol.
   solutes_counts, errors = concentration2count(concentrations, my_box)

   # --- Step 2: Build the Atomic System ---
   # Use the calculated counts to generate the final liquid solution.
   # The builder will automatically balance the total density with H2O molecules.
   system = build_solution(
      box=my_box, 
      density=1.04, 
      solutes_dict=solutes_counts
   )

.. code-block:: none

   <AtomicSystem with 2693 atoms, 1774 bonds>

   Box
   a (Å)  b (Å)  c (Å)  α (°)  β (°)  γ (°)
      30     30     30     90     90     90

   Atoms
   type  number     %      mass  charge
     Cl      16  0.59 35.453200     0.0
      H    1774 65.87  1.007947     0.0
     Na      16  0.59 22.989769     0.0
      O     887 32.94 15.999430     0.0

   Bonds
   type  number
   H-O    1774

   Total charge: 0.000e
   Volume: 27.00 nm3
   Density: 1.04 g/cm3

### Building Complex C-S-H Models
For cementitious materials, the `hydrates` submodule provides the `pycsh` tool, which generates structures based on specific Calcium/Silicate (C/S) and Water/Silicate (W/S) ratios.

Example: Building a C-S-H model
-------------------------------
.. code-block:: python

   from cemd.build.hydrates import build_csh

   # Create a C-S-H model with specific Ca/Si and Water/Si ratios
   csh_system = build_csh(cs_ratio=1.5, ws_ratio=1.5)

Creating Surfaces
-----------------
The **build** module simplifies the generation of crystalline surfaces. The process involves cutting a bulk structure along specific crystallographic planes, defined by their Miller indices, and adding a vacuum layer to avoid interactions between periodic images.

Example: Cutting a Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~
To create a surface, use the `build_surface` function. You need to provide the bulk structure and the Miller indices for the desired cut.

.. code-block:: python

   from cemd.build.base import build_surface

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

   from cemd.build.base import add_liquid

   # Create an interface between a solid surface and a liquid layer
   interface = add_liquid(
       solid_system=my_surface,
       thickness=20.0,
       density=1.0
   )

.. note::
   Many of these functions rely on **Packmol** for the placement of molecules. Ensure that Packmol is installed and correctly configured in your system's PATH.