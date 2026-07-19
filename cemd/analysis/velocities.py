#
# This file is part of the CEMD distribution
# Copyright (c) 2022-2026 Jérôme Claverie.
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

from tqdm import tqdm
import numpy as np
import pandas as pd
import MDAnalysis as mda

def velocity_profile(universe: mda.Universe, 
                     atom_types: list[str], 
                     bin_size = 0.1):

    sel = universe.select_atoms(f"type {atom_types}")

    box = universe.dimensions

    positions = []
    velocities = []

    for ts in tqdm(universe.trajectory):

        # collect positions w.r.t the center of mass to remove the drift
        pos = sel.positions
        vel = sel.velocities

        positions.append(pos[:,2])
        velocities.append(abs(vel))

    positions_array = np.array(positions).flatten()
    velocities_array = np.array(velocities).flatten()

    drange = np.arange(np.min(positions_array), np.max(positions_array), bin_size)
    binned_pos = (drange[:-1] + drange[1:]) / 2

    velocities_mean = np.array([np.mean(velocities_array[np.where((positions_array > low) & (positions_array <= high))]) for low, high in zip(drange[:-1], drange[1:])])

    return pd.Series(velocities_mean, binned_pos)