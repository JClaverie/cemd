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

import os
import subprocess
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOW_TCL = os.path.join(CURRENT_DIR, "view.tcl")

def view(target, trajectory=None) -> None:
    """Opens VMD to visualize an atomic system object or simulation files.

    This function automatically detects whether the input target is an in-memory 
    Python object or a file path. If an object is provided, it writes a 
    temporary LAMMPS data file to disk before launching VMD. It can also 
    overlay a trajectory file onto the topology.

    Args:
        target (AtomicSystem or LAMMPSData or str): The Python molecular object 
            to visualize, or the string path to a LAMMPS data topology file (.data).
        trajectory (str, optional): Path to a molecular trajectory file (e.g., 
            .dcd or .lammpstrj) to be projected onto the topology. Defaults to None.

    Raises:
        FileNotFoundError: If the topology file or the trajectory file does not exist.
        ValueError: If VMD terminates with an error or a non-zero exit code.
        TypeError: If the target argument is neither a supported Python object 
            nor a string path.
    """
    
    if hasattr(target, "write"):
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            temp_data = os.path.join(tmp, 'tmp.data')
            target.write(temp_data)
            
            args = [temp_data]
            if trajectory:
                if not os.path.exists(trajectory):
                    raise FileNotFoundError(f"Trajectory file '{trajectory}' not found.")
                args.append(trajectory)
                
            try:
                subprocess.run(['vmd', '-e', SHOW_TCL, '-args'] + args, check=True)
            except subprocess.CalledProcessError as error:
                raise ValueError("VMD stopped with an error.") from error
        return

    if isinstance(target, str):
        if not os.path.exists(target):
            raise FileNotFoundError(f"Topology file '{target}' does not exist.")
            
        args = [target]
        if trajectory:
            if not os.path.exists(trajectory):
                raise FileNotFoundError(f"Trajectory file '{trajectory}' does not exist.")
            args.append(trajectory)

        try:
            subprocess.run(['vmd', '-e', SHOW_TCL, '-args'] + args, check=True)
        except subprocess.CalledProcessError as error:
            raise ValueError("VMD stopped with an error.") from error
            
    else:
        raise TypeError("Target must be an AtomicSystem/LAMMPSData object or a file path (str).")