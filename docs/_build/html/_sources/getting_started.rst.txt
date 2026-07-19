Getting Started
===============

The **cemd** package revolves around the ``AtomicSystem`` class, which serves as the primary container for your molecular system data.

Loading a System
----------------

To start, you need to load your atomic data. The system currently supports initialization from a dictionary containing the topology (or directly from files via the `IOMixin` methods).

.. code-block:: python

   from cemd import AtomicSystem

   # Initialize with a topology dictionary
   topology = {
       "atoms": ...,          # pandas.DataFrame with x, y, z, type, charge
       "lmp_box": ...,        # LAMMPS box parameters
       "atom_types": ...,     # List of types
       "masses": ...,         # Dictionary of masses
       "charges": ...         # Dictionary of charges
   }

   system = AtomicSystem(topology)
   print(system)

Key Features
------------

The ``AtomicSystem`` object provides immediate access to physical properties and system topology:

* **System Analysis**: Use properties to get density, total charge, or center of mass.
  
  .. code-block:: python

     print(f"Density: {system.density:.2f} g/cm³")
     print(f"Center of Mass: {system.get_center_of_mass()}")

* **Data Inspection**: The `type_summary` property returns a formatted summary of your atom types, masses, and counts.

Visualization
-------------

You can visualize your system directly using VMD:

.. code-block:: python

   # View the current system
   system.view()

   # View with an associated trajectory
   system.view(trajectory="md_production.dcd")

Next Steps
----------
Explore the `analysis` module to start computing radial distribution functions (RDF) or mean squared displacements (MSD) on your systems.