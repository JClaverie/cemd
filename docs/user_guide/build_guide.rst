Building Systems
================

The **build** module is the central engine of the **cemd** package. Its primary purpose is to automate the construction of complex atomistic systems, from simple crystal bulks to intricate solid-liquid interfaces.

.. contents:: Table of Contents
   :local:
   :depth: 2

Workflow Overview
-----------------

Building a simulation-ready system generally follows these three steps:

1. **Initialization**: Load an existing structure or define composition parameters.
2. **Construction**: Generate the system (e.g., creating a liquid box or cutting a surface).
3. **Refinement**: Repair broken bonds or add functional groups (like silanols).

Construction Examples
---------------------

Building a Liquid Solution
^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a water-based solution with solutes, use `build_solution`. It automatically calculates the number of solvent molecules required to reach a specific density.

.. code-block:: python

   from cemd.build import build_solution, concentration2count

   # --- Configuration ---
   # Define the simulation box dimensions in Angstroms (Å)
   my_box = [30, 30, 30]

   # Define target molar concentrations for each solute species (mol/L)
   concentrations = {"Na": 1.0, "Cl": 1.0}

   # --- Step 1: Calculate Particle Counts ---
   # Convert molar concentrations into integer molecule counts based on box volume.
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

Building a Glass Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a glass structure from oxide compositions:

.. code-block:: python

   from cemd.build import build_glass

   # Define a calcium silicate glass composition
   box = [25, 25, 25]
   glass = build_glass(
       box=box,
       density=2.5,
       stoichiometry_dic={
           "SiO2": 0.65,
           "CaO": 0.25,
           "Al2O3": 0.10
       }
   )
   
   print(glass.density)  # Should be close to 2.5 g/cm³

Building Complex C-S-H Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For cementitious materials, the `hydrates` submodule provides tools to generate C-S-H structures based on specific Calcium/Silicate (C/S) and Water/Silicate (W/S) ratios.

.. code-block:: python

   from cemd.build.hydrates import build_csh

   # Create a C-S-H model with specific Ca/Si and Water/Si ratios
   csh_system = build_csh(
       cs_ratio=1.5,
       ws_ratio=1.5,
       supercell=[3, 5, 2]  # Replication factors
   )

   # Print system summary
   print(csh_system)
   
   # Get actual ratios
   n_si = csh_system.get_count("Si")
   n_ca = csh_system.get_count("Ca") + csh_system.get_count("Cw")
   n_h2o = (csh_system.get_count("Hw") + csh_system.get_count("Hh") + csh_system.get_count("H")) / 2
   
   print(f"Real C/S ratio: {n_ca/n_si:.3f}")
   print(f"Real H/S ratio: {n_h2o/n_si:.3f}")

C-A-S-H Models (Aluminum Substitution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cemd.build.hydrates import csh_to_cash

   # Convert a C-S-H structure to C-A-S-H by substituting Si with Al
   cash_system, actual_ratio = csh_to_cash(
       atomic_system=csh_system,
       as_ratio=0.10  # Target Al/Si ratio
   )
   
   print(f"Al/Si ratio: {actual_ratio:.3f}")

Building AFt/AFm Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cemd.build.hydrates import build_af

   # Build an AFt structure (ettringite) with specific water content
   aft_system = build_af(
       ws_ratio=10.0,  # Water/sulfate ratio
       supercell=[1, 1, 1]
   )
   
   print(aft_system)

Creating Surfaces
-----------------

The **build** module simplifies the generation of crystalline surfaces. The process involves cutting a bulk structure along specific crystallographic planes and adding a vacuum layer.

Basic Surface Generation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cemd.build import build_surfaces, explore_surfaces

   # Load a bulk structure
   bulk = AtomicSystem.from_file("tobermorite.cif")
   
   # Generate surfaces cut along the (0, 0, 1) plane
   surfaces, shifts, dipoles, broken_bonds = build_surfaces(
       data=bulk,
       miller_indices=[0, 0, 1],
       min_slab_size=25.0,  # Å
       min_vacuum_size=15.0  # Å
   )
   
   # Select the first generated surface
   my_surface = surfaces[0]
   
   print(f"Generated {len(surfaces)} surfaces")
   print(f"Broken bonds: {broken_bonds}")

Interactive Surface Exploration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Interactive surface explorer
   selected_surface = explore_surfaces(bulk)
   
   if selected_surface:
       selected_surface.view()

Understanding the Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Miller Indices**: Define the cutting plane orientation (`[1,0,0]`, `[1,1,0]`, `[1,1,1]`, etc.)
- **min_slab_size**: Minimum thickness of the material (too thin = bulk properties not representative)
- **min_vacuum_size**: "Gap" in PBC to prevent surface-surface interactions
- **Broken Bonds**: Number of bonds broken during the cut (diagnostic metric)

Creating Solid/Liquid Interfaces
--------------------------------

Once you have your solid and liquid, combine them using interface tools.

.. code-block:: python

   from cemd.build import add_liquid, add_droplet

   # Create a flat liquid layer on top of a surface
   interface = add_liquid(
       solid_system=my_surface,
       thickness=20.0,  # Å
       density=1.0,
       distance=2.0,  # Gap between solid and liquid
       solutes_dict={"Na": 10, "Cl": 10},  # Add ions
       vacuum=10.0  # Vacuum above the liquid
   )

   # Create a droplet instead of a flat layer
   droplet_interface = add_droplet(
       solid_system=my_surface,
       radius=15.0,  # Å
       density=1.0,
       distance=2.0,
       solutes_dict={"Ca": 5, "OH": 10}
   )

Adding Custom Structures
------------------------

You can add any atomic system to a surface:

.. code-block:: python

   from cemd.build import add_structure

   # Load a custom structure
   custom = AtomicSystem.from_file("custom_molecule.pdb")
   
   # Add it to the surface
   combined = add_structure(
       solid_system=my_surface,
       structure_to_add=custom,
       distance=3.0,
       axis='z',
       vacuum=15.0
   )

Merging and Splitting Systems
-----------------------------

.. code-block:: python

   from cemd.build import merge, split

   # Merge two systems
   merged = merge(system1, system2, box=[50, 30, 30, 90, 90, 90])
   
   # Split a system and add space
   splitted = split(
       solid_system=merged,
       axis=2,  # z-axis
       gap_size=20.0,
       add_solution=True,  # Fill the gap with liquid
       density=1.0
   )