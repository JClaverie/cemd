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

import os
from typing import Sequence

import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm import tqdm
from MDAnalysis import transformations

def write_dcd(universe: mda.Universe, 
              output_path: str, 
              selection: str = "all", 
              start: int = 0, 
              end: int = None, 
              step: int = 1) -> None:
    """Write trajectory frames from an MDAnalysis Universe to a new DCD file.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe object containing topology and trajectory data.
    output_path : str
        Path where the new DCD file will be saved.
    selection : str, optional
        Selection string (MDAnalysis syntax) specifying which atoms to write.
    start : int, optional
        Starting frame index (0-indexed).
    end : int, optional
        Ending frame index. If None, writes up to the last available frame.
    step : int, optional
        Step size (stride) for writing frames.
    """
    # check on file extension
    _, ext = os.path.splitext(output_path)
    if ext.lower() != '.dcd':
        raise TypeError("The output file extension must be '.dcd'.")

    # apply the atom selection
    selected_atoms = universe.select_atoms(selection)
    if len(selected_atoms) == 0:
        raise ValueError(f"Selection string '{selection}' matched 0 atoms.")

    print(f"Preparing to write DCD file: {output_path}")
    print(f"Selected {len(selected_atoms)} out of {len(universe.atoms)} atoms.")

    # handle the trajectory slice definition
    trajectory_slice = universe.trajectory[start:end:step]
    total_frames_to_write = len(trajectory_slice)
    
    print(f"Writing {total_frames_to_write} frames (start={start}, end={end}, step={step})...")

    # open the writer and stream the frames
    # the Writer requires the output filename and the exact number of atoms being written
    with mda.Writer(output_path, selected_atoms.n_atoms) as W:
        for ts in trajectory_slice:
            W.write(selected_atoms)

    print("DCD trajectory written successfully!")

def shift2com(universe: mda.Universe, 
              atom_types: list[str | int], 
              output_trajectory: str = 'recentered_traj.dcd') -> None:
    """Recenter all atoms relative to the center of mass (COM) of a selection.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis universe object.
    atom_types : list
        List of atom types used to define the reference center of mass.
    output_trajectory : str, optional
        Path to the output recentered DCD trajectory file.
    """
    
    selection_string = f"type {' '.join(atom_types)}"
    ref_atoms = universe.select_atoms(selection_string)
    
    if len(ref_atoms) == 0:
        raise ValueError(f"No atoms found for the type(s): {atom_types}")
        
    all_atoms = universe.atoms
    
    with mda.Writer(output_trajectory, all_atoms.n_atoms) as W:
        
        for ts in universe.trajectory:
            com = ref_atoms.center_of_mass()
            
            all_atoms.positions -= com
            
            W.write(all_atoms)

# def shift2com(itraj, psf, atypes, otraj='recentered_traj.dcd'):
#     """Recenter all atoms w.r.t to the COM of a selection based on atom types

#     Parameters
#     ----------
#         itraj: str
#             An input DCD trajectory
#         psf: str 
#             A PSF file
#         atypes: list of str
#             List of the atom types in the reference selection
#         otraj: str
#             An output DCD trajectory
    
#     """

#     sel_str = " ".join(map(str, atypes))

#     subprocess.run(['vmd', '-dispdev', 'text', '-e', shift2com_tcl, '-args', itraj, psf, sel_str, otraj], stdout=subprocess.DEVNULL, check=True)            

def minmax_position(universe: mda.Universe, 
                    atom_types: list[str | int],
                    axis: str ='z',
                    bounds: Sequence[float] = None, 
                    start: int = 0, 
                    end: int = -1) -> tuple[float, float]:
    """Calculate the mean of the minimum and maximum coordinates along an axis.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis Universe object containing topology and trajectory.
    atom_types : list
        List of atom type strings to select.
    axis : str, optional
        Axis along which to calculate the coordinates ('x', 'y', or 'z').
    bounds : Sequence[float], optional
        Optional spatial limits [min, max] in Angstroms to filter atoms.
    start : int, optional
        Starting frame index for the trajectory slice.
    end : int, optional
        Ending frame index for the trajectory slice.

    Returns
    -------
    tuple
        A tuple containing (mean_min, mean_max) coordinates.
    """

    typestr = " ".join(map(str, atom_types))

    if bounds is None:
        selstr = "type {}".format(typestr)
        print(f"Compute the min and max coordinate along {axis} for {typestr} atoms...")
    else:
        selstr = f"type {typestr} and prop {axis} >= {bounds[0]} and prop {axis} < {bounds[1]}"
        print(f"Compute the min and max coordinate along {axis} for {typestr} atoms between {axis}={bounds[0]} angströms and {axis}={bounds[1]}...")

    sel = universe.select_atoms(selstr)

    if axis == 'x': axid = 0
    if axis == 'y': axid = 1
    if axis == 'z': axid = 2

    mins, maxs = [], []

    for ts in tqdm(universe.trajectory[start:end]):

        mins.append(sel.positions[:,axid].min())
        maxs.append(sel.positions[:,axid].max())
    
    return np.mean(np.array(mins)), np.mean(np.array(maxs))

def mean_pos(universe) -> pd.DataFrame:
    """Return a DataFrame containing the mean position of each atom over the trajectory.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis universe object.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'type', 'x', 'y', and 'z'.
    """

    ag = universe.atoms
    transform = transformations.unwrap(ag)
    universe.trajectory.add_transformations(transform)

    mean_pos = np.zeros((len(ag), 3))
    for ts in tqdm(universe.trajectory):
        mean_pos += ag.positions

    mean_pos /= len(universe.trajectory)

    combined = np.column_stack((universe.atoms.types, mean_pos))

    df = pd.DataFrame(combined, columns=['type', 'x', 'y', 'z'])

    return df

def com(universe: mda.Universe, atom_types: list[str | int]) -> float:
    """Calculate the mean position of the center of mass of a selection.

    Parameters
    ----------
    universe : mda.Universe
        MDAnalysis universe object.
    atom_types : list
        List of atom type strings to select.

    Returns
    -------
    float
        The mean position of the center of mass over the trajectory.
    """

    sel = universe.select_atoms("type {}".format(' '.join(atom_types)))

    comlist = []

    for ts in tqdm(universe.trajectory):
        comlist.append(sel.center_of_mass())

    return np.mean(comlist, axis = 0)