.. _tutorial_topology_forcefield:

=======================================================================
Tutorial: Building Topology and Apply Forcefield parameters for Calcite
=======================================================================

This tutorial demonstrates how to build topology for a calcite structure
using known bond distances from the literature.

.. contents::
   :local:
   :depth: 2

Prerequisites
==============

.. code-block:: python

   from cemd import AtomicSystem, TopologyRule, NeighborCriterion

Step 1: Load Calcite Structure
===============================

Load the calcite structure from :ref:`tutorial_calcite_surface` and understand what we're working with.

.. code-block:: python

   # Load calcite structure
   system = AtomicSystem.from_file("calcite_104.data")
   print(system)

.. code-block:: text

    <AtomicSystem with 2520 atoms>

    Box
    a (Å)  b (Å)  c (Å)  α (°)  β (°)  γ (°)
    34.86  24.36  60.89  90.00  90.00  90.00

    Atoms
    type  number     %     mass  charge
    C     504 20.00 12.01078     4.0
    Ca     504 20.00 40.07840     2.0
    O    1512 60.00 15.99943    -2.0

    Total charge: 0.000e
    Volume: 51.69 nm3
    Density: 1.62 g/cm3

Step 2: Create Topology Rules for Calcite
=========================================

Now we create topology rules using the known bond distances.

.. code-block:: python

   # Carbon atom bonded to exactly 3 oxygen atoms at 1.4 Å
   carbonate_topo_rule = TopologyRule(
       center="type C",
       neighbors=NeighborCriterion("type O", 1.4, 3),
       bonds=True,
       angles=True
   )

   print(system)

.. code-block:: text

    <AtomicSystem with 2520 atoms, 1512 bonds, 1512 angles>

    Box
    a (Å)  b (Å)  c (Å)  α (°)  β (°)  γ (°)
    34.86  24.36  60.89  90.00  90.00  90.00

    Atoms
    type  number     %     mass  charge
       C     504 20.00 12.01078     4.0
      Ca     504 20.00 40.07840     2.0
       O    1512 60.00 15.99943    -2.0

    Bonds
    type  number
     C-O    1512

    Angles
    type  number
    O-C-O   1512

    Total charge: 0.000e
    Volume: 51.69 nm3
    Density: 1.62 g/cm3

Step 3: Apply Rules and Inspect Results
=======================================

Apply each rule to the system and inspect the results.


