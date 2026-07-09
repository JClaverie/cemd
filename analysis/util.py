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

    Iterates through the specified frame range and step of a universe trajectory
    and writes the coordinates of the selected atom group to disk.

    Parameters
    ----------
    universe
        MDAnalysis Universe object containing topology and trajectory data.
    output_path
        Path where the new DCD file will be saved.
    selection
        Selection string (MDAnalysis syntax) specifying which atoms to write.
    start
        Starting frame index (0-indexed).
    end
        Ending frame index. If None, writes up to the last available frame.
    step
        Step size (stride) for writing frames (e.g., step=2 writes every second frame).
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
    """Recenter all atoms w.r.t to the COM of a selection based on atom types.

    Parameters
    ----------
    universe
        MDAnalysis universe
    atom_types
        List of the atom types used to define the reference center of mass.
    otraj
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
    """Return the mean of the minimum and maximum coordinates of a group of atoms along an axis.

    Iterates through the trajectory frames and finds the spatial extrema (min and max)
    for a selection of atom types. If bounds are provided, only atoms within that 
    spatial range at each frame are considered.

    Parameters
    ----------
    universe
        MDAnalysis Universe object containing topology and trajectory.
    atom_types
        List of atom type strings to select (e.g., ['Si', 'O']).
    axis
        Axis along which to calculate the coordinates ('x', 'y', or 'z').
    bounds
        Optional spatial limits [min, max] in Angstroms to filter atoms along the axis.
    start
        Starting frame index for the trajectory slice.
    end
        Ending frame index for the trajectory slice. If None, goes to the last frame.
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
    """Return a DataFrame containing the mean position of each atoms over the trajectory.

    Parameters
    ----------
        universe
            MDAnalysis universe

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
    """Return the mean position of the center of mass of the selection given by types over the trajectory.

    Parameters
    ----------
        universe
            MDAnalysis universe
        atom_types
            List of atom type strings to select (e.g., ['Si', 'O']).
    
    """

    sel = universe.select_atoms("type {}".format(' '.join(atom_types)))

    comlist = []

    for ts in tqdm(universe.trajectory):
        comlist.append(sel.center_of_mass())

    return np.mean(comlist, axis = 0)

# def dump2dcd(dumptraj, dcdtraj=None) -> None:
#     """Convert a LAMMPS dump trajectory to a DCD trajectory.

#     Parameters
#     ----------
#         dumptraj: str 
#             An input LAMMPS dump trajectory
#         dcdtraj: str
#             An output DCD trajectory
    
#     """

#     name, ext = os.path.splitext(dumptraj)

#     if ext != '.lammpsdump':
#         newdumpname = name + '.lammpsdump'
#         os.rename(dumptraj, newdumpname)
#         u = mda.Universe(newdumpname)
#         os.rename(newdumpname, name + ext)
#     else:
#         u = mda.Universe(dumptraj)
    
#     if dcdtraj is None:
#         dcdtraj = name + '.dcd'

#     with mda.Writer(dcdtraj, len(u.atoms)) as W:
#         for ts in u.trajectory:
#             W.write(u)

# def slice_trajectory(universe: mda.Universe, 
#                      start: int, 
#                      end: int, 
#                      output_traj: str=None) -> None:
#     """Extract a slice of a trajectory and save it to a new DCD file.

#     Iterates through the specified frame range of an existing MDAnalysis Universe
#     and writes the coordinates into a new trajectory file.

#     Parameters
#     ----------
#     universe
#         MDAnalysis Universe object containing the topology and trajectory.
#     start
#         Starting frame index (0-indexed).
#     end
#         Ending frame index (inclusive).
#     output_traj
#         Path to the output DCD file. If None, automatically generated 
#         based on the input file name and frame range.
#     """

#     if output_traj is None:
#         input_file = universe.trajectory.filename
#         name, _ = os.path.splitext(input_file)
#         output_traj = f'{name}_{start}-{end}.dcd'
#     else:
#         _, ext = os.path.splitext(output_traj)
#         if ext != '.dcd':
#             raise TypeError("Please change the extension of the output trajectory to .dcd.")

#     with mda.Writer(output_traj, len(universe.atoms)) as W:
#         for ts in universe.trajectory[start:end+1]:
#             W.write(universe)

# def merge_trajectories(topology_file: str, 
#                        trajectory_list: list[str], 
#                        output_traj: str = None) -> None:
#     """Merge a list of sequential trajectories into a single DCD file.

#     Loads a topology along with multiple trajectory files using MDAnalysis, 
#     and concatenates them frame by frame into a unique output trajectory.

#     Parameters
#     ----------
#     topology_file
#         Path to the topology file (e.g., .psf, .pdb) matching the trajectories.
#     trajectory_list
#         A list of paths to the trajectory files to be merged in order.
#     output_traj
#         Path to the output merged DCD file. If None, default is 'merged_traj.dcd'.
#     """
#     if not trajectory_list:
#         raise ValueError("The list of trajectories provided is empty.")

#     if output_traj is None:
#         output_traj = 'merged_traj.dcd'
#     else:
#         _, ext = os.path.splitext(output_traj)
#         if ext.lower() != '.dcd':
#             raise TypeError("L'extension du fichier de sortie doit être '.dcd'.")

#     print(f"Loading the topology and {len(trajectory_list)} trajectories...")
    
#     u = mda.Universe(topology_file, trajectory_list)

#     all_atoms = u.atoms

#     print(f"Merging to file: {output_traj} ({len(u.trajectory)} frames in total)")

#     with mda.Writer(output_traj, all_atoms.n_atoms) as W:
#         for ts in u.trajectory:
#             W.write(all_atoms)

#     print("Merge completed successfully!")

# def lmp2psf(lmpdata_file):
#     """Generate a PSF file from a LAMMPS data file to read with DCD trajectory.

#     Parameters
#     ----------
#         lmpdata_file: str 
#             Input LAMMPS data file

#     """

#     fname = os.path.splitext(lmpdata_file)[0]

#     output_file = fname + ".psf"

#     subprocess.run(['vmd', '-dispdev', 'text', '-e', makepsf_tcl, '-args', lmpdata_file, output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

#     print(f"{output_file} generated from LAMMPS data with VMD!")
