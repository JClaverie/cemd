#
# This file is part of the CEMD distribution
# Copyright (c) 2024-2026 Jérôme Claverie.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import annotations

from typing import Sequence, Self, TYPE_CHECKING

import numpy as np
import pandas as pd

from .._utils import lattice2lammps, lattice2vectors, vectors2lattice
from .._constants import MASSES_DICT

if TYPE_CHECKING:
    from atomic_system import AtomicSystem

class EditMixin:
    """Mixin class containing editing and modification methods 
    for the AtomicSystem class.
    """

    def set_box(self, new_box: Sequence[float] | np.ndarray) -> None:
        """Assign a new box to the system.

        Parameters
        ----------
            new_box: list of float
                list of the new box parameters to assign
        """

        if isinstance(new_box, list):
            new_box = np.array(new_box)
            
        self._box = new_box
        self._lmp_box = lattice2lammps(new_box)
        self._box_vectors = lattice2vectors(self._box)

    def set_coordinates(self: AtomicSystem, 
                        index: int, 
                        position: Sequence[float]) -> AtomicSystem:
        """
        Modify the coordinates of a single atom using a vector (x, y, z).
        """
        if len(position) != 3:
            raise ValueError("Position must be an iterable of 3 floats (x, y, z)")

        if index not in self.atoms.index:
            print(f"Warning: Index {index} not found.")
            return self

        x, y, z = position
        
        self.atoms.at[index, 'x'] = float(x)
        self.atoms.at[index, 'y'] = float(y)
        self.atoms.at[index, 'z'] = float(z)

        return self

    def replicate(self, factors: Sequence[int]) -> Self:
        """
        Replicate the system by integer factors along the lattice vectors (a, b, c).
        
        Parameters
        ----------
        factors :
            Number of copies along each lattice vector (nx, ny, nz).
        """
        nx, ny, nz = factors
        if nx == 1 and ny == 1 and nz == 1:
            return self

        # Prepare the basis vectors
        # lattice2vectors returns a (3, 3) matrix where each row is a vector v1, v2, v3
        vecs = np.array(lattice2vectors(self.box))
        v1, v2, v3 = vecs[0], vecs[1], vecs[2]

        original_atoms = self.atoms.copy()
        original_num_atoms = len(original_atoms)
        
        all_atoms = []
        
        # Dictionaries to store new interactions
        new_interactions = {
            'bonds': [], 'angles': [], 'dihedrals': [], 'impropers': []
        }

        # Replication loop
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if i == 0 and j == 0 and k == 0:
                        all_atoms.append(original_atoms)
                        continue
                    
                    # Calculate the shift for this cell
                    shift = i * v1 + j * v2 + k * v3
                    
                    # Copy atoms
                    new_atoms = original_atoms.copy()
                    new_atoms['x'] += shift[0]
                    new_atoms['y'] += shift[1]
                    new_atoms['z'] += shift[2]
                    
                    # Updating indices (IDs) to avoid duplicates
                    # shift the indexes by (number_atoms_orig *unique_multiplier)
                    offset = (i * ny * nz + j * nz + k) * original_num_atoms
                    new_atoms.index += offset
                    all_atoms.append(new_atoms)

                    # Replicate the interactions (Bonds, Angles, etc.)
                    for name in ['bonds', 'angles', 'dihedrals', 'impropers']:
                        df = getattr(self, name)
                        if df is not None and not df.empty:
                            new_df = df.copy()
                            # shift the index type of the interaction itself
                            new_df.index += (i * ny * nz + j * nz + k) * len(df)
                            
                            # shift the columns atom_1, atom_2, etc.
                            atom_cols = [c for c in df.columns if c.startswith('atom_')]
                            for col in atom_cols:
                                new_df[col] += offset
                            
                            new_interactions[name].append(new_df)

        # Merge Dataframes
        self.atoms = pd.concat(all_atoms)
        
        if self.velocities is not None:
            self.velocities = None

        for name in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if new_interactions[name]:
                full_list = [getattr(self, name)] + new_interactions[name]
                setattr(self, name, pd.concat(full_list))

        new_box = self.box.copy()
        new_box[0] *= nx
        new_box[1] *= ny
        new_box[2] *= nz
        self.set_box(new_box)

        return self

    def wrap(self) -> None:
            """
            Wraps all atoms back into the simulation box [0, L] using Periodic Boundary Conditions.
            Handles both orthogonal and triclinic boxes.
            """
            if self.atoms.empty:
                return self

            # Get the pass matrix (H_matrix)
            H = np.array(self._box_vectors)
            inv_H = np.linalg.inv(H)

            # Extract current coordinates (N, 3)
            coords = self.atoms[['x', 'y', 'z']].values

            # Conversion to fractional coordinates (0 to 1)
            # s = r . inv(H)
            frac_coords = np.dot(coords, inv_H)

            # Application of modulo 1.0
            # This brings everything into the range [0, 1[
            frac_coords %= 1.0

            # Return to Cartesian coordinates
            # r_new = s_new . H
            new_coords = np.dot(frac_coords, H)

            self.atoms[['x', 'y', 'z']] = new_coords
            
            return self

    def orthogonalize(self, tolerance: float=0.1, max_replica: int=10) -> bool:
        """
        Attempt to transform a triclinic (skewed) box into an orthogonal one.

        This method searches for new lattice vectors that align with the Cartesian 
        axes (X, Y, Z) by exploring linear combinations of the current basis 
        vectors. It then re-populates the new box by shifting and wrapping 
        existing atom coordinates.

        Parameters
        ----------
        tolerance
            The maximum deviation allowed from the Cartesian axes (in Å) for 
            a candidate vector to be considered orthogonal.
        max_replica
            The search range for linear combinations (m, n, o) of the original 
            lattice vectors. Higher values increase the chance of finding an 
            orthogonal cell but significantly increase computation time.

        Returns
        -------
        bool
            True if an orthogonal cell was successfully found and populated, 
            False otherwise. In case of failure, the box is 'unskewed' 
            (tilt factors removed) to maintain consistency.

        Notes
        -----
        - This operation will change the total number of atoms in the system.
        - Velocities and interactions (bonds, etc.) are typically lost or 
          invalidated during this specific geometric transformation.
        - The resulting box will have tilt factors (xy, xz, yz) set to zero.
        """

        H = np.array(lattice2vectors(self.box))
        new_vectors = []

        for i in range(3):
            best_v = None
            best_len = float('inf')
            
            for m in range(-max_replica, max_replica + 1):
                for n in range(-max_replica, max_replica + 1):
                    for o in range(-max_replica, max_replica + 1):
                        if m == 0 and n == 0 and o == 0: continue
                        
                        v_cand = m*H[0] + n*H[1] + o*H[2]
                        # check the alignment with the i axis
                        others = [v_cand[j] for j in range(3) if j != i]
                        
                        if all(abs(comp) < tolerance for comp in others):
                            v_len = abs(v_cand[i]) # We want the length on the axis
                            if 0.1 < v_len < best_len: # 0.1 to avoid zero vector
                                best_len = v_len
                                best_v = v_cand
            
            if best_v is None:
                print(f"Error: Unable to find an orthogonal vector for axis {i}")
                print(f"Unskew the box anyway")
                self.unskew() 
                return False
            new_vectors.append(best_v)

        diag_dim = [abs(new_vectors[0][0]), abs(new_vectors[1][1]), abs(new_vectors[2][2])]
        new_H = np.diag(diag_dim)
        
        search_range = np.arange(-max_replica, max_replica + 1)
        m_grid, n_grid, o_grid = np.meshgrid(search_range, search_range, search_range)
        translations = np.vstack([m_grid.ravel(), n_grid.ravel(), o_grid.ravel()]).T
        translation_vectors = translations @ H

        all_replicas = []
        eps = 1e-5

        for vec in translation_vectors:
  
            temp_df = self.atoms.copy()
            temp_df['x'] += vec[0]
            temp_df['y'] += vec[1]
            temp_df['z'] += vec[2]
            
            mask = (
                (temp_df['x'] >= -eps) & (temp_df['x'] < diag_dim[0] - eps) &
                (temp_df['y'] >= -eps) & (temp_df['y'] < diag_dim[1] - eps) &
                (temp_df['z'] >= -eps) & (temp_df['z'] < diag_dim[2] - eps)
            )
            if mask.any():
                all_replicas.append(temp_df[mask])
            
        if not all_replicas:
            return False

        self.atoms = pd.concat(all_replicas, ignore_index=True)
        
        self.atoms.index = range(1, len(self.atoms) + 1)
        if 'id' in self.atoms.columns:
            self.atoms['id'] = range(1, len(self.atoms) + 1)

        self.set_box(vectors2lattice(tuple(new_H)))
        
        return True

    def unskew(self) -> None:
        """
        Unskew the box if the non-diagonal parameters of the box matrix 
        :math:`|H_{i,j}| > H_{j,j}` for :math:`i \\neq j` by adding 
        :math:`\\pm H_{j,:}` to :math:`H_{i,:}` while :math:`|H_{i,j}| > H_{j,j}`.
        """

        boxm = np.array( lattice2vectors(self.box) )

        for i in range(3):
            for j in range(3):
                if i != j:
                    while boxm[i,j] > 0.5 * boxm[i,i]:
                        boxm[i] -= boxm[j]
                    while boxm[i,j] < - 0.5 * boxm[i,i]:
                        boxm[i] += boxm[j]

        self.set_box( vectors2lattice( (boxm[0], boxm[1], boxm[2]) ) )
        self.wrap()

    def center_on_com(self) -> None:
        """
        Translates the system so its center of mass is at the center of the box, correctly handling Periodic Boundary Conditions (PBC).
        """
        import numpy as np

        # Prepare dimensions and weights
        box_dims = self.box[:3]  # [Lx, Ly, Lz]
        atom_masses = self.atoms['type'].map(lambda t: self._masses_storage.get(t, MASSES_DICT.get(t, 1.0)))
        total_mass = atom_masses.sum()
        
        if total_mass <= 0:
            return

        # Calculate the Periodic Center of Mass (Bai and Breen method)
        # transform coordinates to periodic angles: theta = 2 * pi * x / L
        com_coords = []
        for i, axis in enumerate(['x', 'y', 'z']):
            L = box_dims[i]
            theta = (self.atoms[axis] / L) * 2 * np.pi
            
            # Average of sine and cosine weighted by mass
            avg_sin = (np.sin(theta) * atom_masses).sum() / total_mass
            avg_cos = (np.cos(theta) * atom_masses).sum() / total_mass
            
            # Back to average angle, then back to coordinate
            avg_theta = np.arctan2(-avg_sin, -avg_cos) + np.pi
            com_coords.append((avg_theta / (2 * np.pi)) * L)
        
        com_pbc = np.array(com_coords)

        # Translate the system
        # Target: put the COM at the center of the box (box_dims / 2)
        target = box_dims / 2
        shift = target - com_pbc
        
        self.atoms[['x', 'y', 'z']] += shift
        
        # Wrap everything back into the box [0, L]
        self.wrap()

