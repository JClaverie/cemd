Analyzing MD Trajectories
=========================

The **analysis** module provides tools to extract physical properties from molecular dynamics simulations. It is designed to work seamlessly with `MDAnalysis` universes, allowing you to compute structural, dynamical, and electrostatic properties.

Structural Analysis
-------------------
To understand the organization of atoms in your system, you can compute distribution functions.

### Radial Distribution Function (RDF)
The RDF, $g(r)$, describes the probability of finding a particle at a distance $r$ from another. 

.. code-block:: python

   from cemd.analysis import compute_rdf

   # Compute RDF between 'Si' and 'O' atoms
   rdf_df = compute_rdf(universe, type1="Si", type2="O", cutoff=10.0)

   # The resulting DataFrame contains g(r), G(r), and the coordination number n(r)
   print(rdf_df.head())

Dynamical Analysis
------------------
Diffusion properties are key to characterizing the mobility of ions or water molecules in confined environments.

### Mean Squared Displacement (MSD)
The MSD is used to calculate the diffusion coefficient ($D$) via Einstein's relation. You can compute it for bulk solutions or as a profile at an interface.

.. code-block:: python

   from cemd.analysis import msd, diffusion_coefficient

   # Calculate MSD for 'Ow' (water oxygen) atoms
   msd_data = msd(universe, atom_type="Ow", dt=100.0)
   
   # Extract the diffusion coefficient
   diff_coeff = diffusion_coefficient(msd_data)

Interface and Electrostatics
----------------------------
For interfacial systems (like C-S-H/water), you can analyze the local density and the resulting electrostatic potential.

### Density Profiles
To analyze the density variation along an axis (e.g., perpendicular to a surface):

.. code-block:: python

   from cemd.analysis import density_profile

   # Calculate the density profile along the Z axis
   profile = density_profile(universe, atom_types=["Ow", "Ca"], axis="z")

### Electrostatic Potential
By combining density profiles with atomic charges, you can estimate the electrostatic potential across the interface.

.. code-block:: python

   from cemd.analysis import electrostatic_potential

   # Provide the charges corresponding to the atom types in your profile
   potential = electrostatic_potential(profile, list_charges=[0.0, 2.0])