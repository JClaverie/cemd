.. currentmodule:: cemd.core.atomic_system

AtomicSystem
============

.. autoclass:: AtomicSystem
   :members: atoms, bonds, angles, dihedrals, impropers, velocities, pair_params, bond_params, angle_params, dihedral_params, improper_params

Properties
----------

.. autosummary::
   :nosignatures:

   AtomicSystem.box
   AtomicSystem.volume
   AtomicSystem.density
   AtomicSystem.total_charge
   AtomicSystem.total_mass
   AtomicSystem.density
   AtomicSystem.masses
   AtomicSystem.charges
   AtomicSystem.elements
   AtomicSystem.num_atoms
   AtomicSystem.num_bonds
   AtomicSystem.num_angles
   AtomicSystem.num_dihedrals
   AtomicSystem.num_impropers
   AtomicSystem.atom_types
   AtomicSystem.bond_types
   AtomicSystem.angle_types
   AtomicSystem.dihedral_types
   AtomicSystem.improper_types

Input/Output
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AtomicSystem.from_file
   AtomicSystem.from_cod
   AtomicSystem.from_pubchem
   AtomicSystem.from_mda
   AtomicSystem.from_pmg
   AtomicSystem.to_mda
   AtomicSystem.to_pmg
   AtomicSystem.write

Topology
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AtomicSystem.set_types
   AtomicSystem.set_type2atoms
   AtomicSystem.reset_types
   AtomicSystem.reset_types
   AtomicSystem.set_topology
   AtomicSystem.reset_topology
   AtomicSystem.remove_connection_types
   AtomicSystem.keep_connection_types
   AtomicSystem.add_bond
   AtomicSystem.add_angle
   AtomicSystem.add_dihedral
   AtomicSystem.add_improper

Force Field
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AtomicSystem.set_masses
   AtomicSystem.set_charges
   AtomicSystem.set_ff_from_database
   AtomicSystem.explore_ff_database
   AtomicSystem.set_ff_pair_param
   AtomicSystem.set_ff_bond_param
   AtomicSystem.set_ff_angle_param
   AtomicSystem.apply_pair_mixing_rules

Modifiers
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AtomicSystem.add_atom
   AtomicSystem.add_atoms
   AtomicSystem.protonate_atom
   AtomicSystem.protonate_atoms
   AtomicSystem.remove_atom
   AtomicSystem.remove_atoms
   AtomicSystem.set_atom_position
   AtomicSystem.set_box
   AtomicSystem.orthogonalize
   AtomicSystem.unskew
   AtomicSystem.replicate
   AtomicSystem.wrap
   AtomicSystem.center_on_com

