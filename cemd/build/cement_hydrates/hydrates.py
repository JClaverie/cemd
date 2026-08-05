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
import subprocess
import tempfile
import shutil
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Union, TYPE_CHECKING

import numpy as np

from ..base import BaseBuilder
from ..._utils import require_program
from ..._paths import STRUCTURES_DIR, PYCSH_DIR
from ...core.atomic_system import AtomicSystem
from ...analysis import analyze_silicates
from ._silicate_helpers import substitute_si_by_al

if TYPE_CHECKING:
    pass


# =============================================================================
# CSHBuilder
# =============================================================================

@dataclass
class CSHBuilder(BaseBuilder):
    """
    Builder for creating C-S-H (Calcium-Silicate-Hydrate) structures.
    
    This builder creates C-S-H models using either:
    - build()          : Classic tobermorite-based approach with interlayer filling
    - build_pycsh()    : Full atomistic model using pyCSH
    
    Parameters
    ----------
    cs_ratio : float
        Calcium/Silicate ratio.
    ws_ratio : float
        Water/Silicate ratio.
    progress_callback : Callable, optional
        Progress callback function.
    
    Examples
    --------
    >>> # Create C-S-H with classic method (tobermorite)
    >>> builder = CSHBuilder(cs_ratio=1.5, ws_ratio=1.0)
    >>> system = builder.build(
    ...     supercell=[4,1,1],
    ...     model='tob11a_merlino.cif'
    ... )
    >>> 
    >>> # Create C-S-H with pyCSH
    >>> builder = CSHBuilder(cs_ratio=1.2, ws_ratio=0.8)
    >>> systems = builder.build_pycsh(
    ...     supercell=[3,5,2],
    ...     nsamples=3,
    ...     seed=42
    ... )
    """
    
    # Required parameters
    cs_ratio: float = field(default=None)
    ws_ratio: float = field(default=None)
    
    # Optional parameter (commun)
    progress_callback: Optional[Callable[[int, str], None]] = None
    
    # Internal state
    _system: Optional[AtomicSystem] = field(default=None, init=False, repr=False)
    _analysis: Optional[dict] = field(default=None, init=False, repr=False)
    _is_built: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize and validate the builder."""
        super().__init__(None)
        if self.cs_ratio is not None and self.cs_ratio <= 0:
            raise ValueError(f"cs_ratio must be positive, got {self.cs_ratio}")
        if self.ws_ratio is not None and self.ws_ratio < 0:
            raise ValueError(f"ws_ratio must be >= 0, got {self.ws_ratio}")
    
    def __repr__(self) -> str:
        """Return a styled string representation."""
        lines = ["<CSHBuilder>"]
        lines.append("")
        lines.append("┌─ Configuration")
        lines.append(f"│   Ca/Si ratio  : {self.cs_ratio:.3f}" if self.cs_ratio is not None else "│   Ca/Si ratio  : None")
        lines.append(f"│   H2O/Si ratio : {self.ws_ratio:.3f}" if self.ws_ratio is not None else "│   H2O/Si ratio : None")
        
        if self._system is not None:
            lines.append(f"│   System        : {self._system.num_atoms} atoms")
            
            # Analyse automatique si elle n'existe pas encore
            if self._analysis is None:
                try:
                    self._analysis = analyze_silicates(self._system)
                except Exception as e:
                    self._analysis = {'error': str(e)}
            
            if self._analysis is not None and 'error' not in self._analysis:
                lines.append("│")
                lines.append("├─ Analysis")
                lines.append(f"│   Ca/(Si+Al)  : {self._analysis['Ca/(Si+Al)']:.3f}")
                lines.append(f"│   Al/Si        : {self._analysis['Al/Si']:.3f}")
                lines.append(f"│   H2O/(Si+Al)  : {self._analysis['H2O/(Si+Al)']:.3f}")
                mcl = self._analysis['MCL']
                mcl_str = f"{mcl:.2f}" if mcl != float('inf') else "∞"
                lines.append(f"│   MCL          : {mcl_str}")
                qn = self._analysis['Qn_distribution']
                lines.append(f"│   Qⁿ dist      : Q0={qn[0]:.1f}% Q1={qn[1]:.1f}% Q2={qn[2]:.1f}% Q3={qn[3]:.1f}% Q4={qn[4]:.1f}%")
            elif self._analysis is not None and 'error' in self._analysis:
                lines.append("│")
                lines.append("├─ Analysis")
                lines.append(f"│   ⚠️ {self._analysis['error']}")
        else:
            lines.append("│   Status        : no system")
        
        lines.append("└─")
        return "\n".join(lines)

    @classmethod
    def from_system(cls, system: AtomicSystem) -> "CSHBuilder":
        """
        Create a builder from an existing AtomicSystem.
        
        This is useful for analyzing or modifying existing C-S-H structures.
        
        Parameters
        ----------
        system : AtomicSystem
            The existing C-S-H system.
        
        Returns
        -------
        CSHBuilder
            A builder instance configured with the given system.
        
        Examples
        --------
        >>> system = AtomicSystem.from_file("csh.lmp")
        >>> builder = CSHBuilder.from_system(system)
        >>> analysis = builder.analyze()
        >>> cash_system, ratio = builder.to_cash(as_ratio=0.1)
        """
        builder = cls()
        builder._system = system
        try:
            builder._analysis = analyze_silicates(system)
        except Exception as e:
            builder._analysis = {'error': str(e)}
        return builder

    def analyze(self, types_map: Optional[dict] = None, cutoff: float = 1.85) -> dict:
        """
        Analyze the silicate network of the current system.
        
        This method uses `analyze_silicates` to compute:
        - Stoichiometric ratios (Ca/(Si+Al), Al/Si, H2O/(Si+Al))
        - Q^n distribution (polymerization)
        - Mean Chain Length (MCL)
        
        Parameters
        ----------
        types_map : dict, optional
            Custom mapping of atom types for Si, O, Al, and Ca.
            Defaults to TYPES_PRESET.
        cutoff : float, default=1.85
            Distance cutoff for neighbor selection in Å.
        
        Returns
        -------
        dict
            Analysis results containing 'Ca/(Si+Al)', 'Al/Si', 'H2O/(Si+Al)',
            'MCL', and 'Qn_distribution'.
        
        Examples
        --------
        >>> builder = CSHBuilder.from_system(system)
        >>> analysis = builder.analyze()
        >>> print(f"MCL: {analysis['MCL']:.2f}")
        >>> print(f"Qn distribution: {analysis['Qn_distribution']}")
        """
        if self._system is None:
            raise RuntimeError("No system available. Build one first or use from_system().")
        
        system = self._system or (self._result if isinstance(self._result, AtomicSystem) else self._result[0])
        
        self._analysis = analyze_silicates(system, types_map=types_map, cutoff=cutoff)
        return self._analysis

    def _update_analysis(self) -> None:
        """Update analysis for the current system."""
        if self._system is not None:
            try:
                self._analysis = analyze_silicates(self._system)
            except Exception as e:
                self._analysis = {'error': str(e)}
    
    # ------------------------------------------------------------------
    # Classic method (tobermorite)
    # ------------------------------------------------------------------
    
    def build(self,
              supercell: Optional[Sequence[int]] = None,
              model: str = 'tob11a_merlino.cif',
              min_mcl: float = 3.0,
              symmetry: bool = True,
              _progress_callback: Optional[Callable[[int, str], None]] = None) -> AtomicSystem:
        """
        Build C-S-H using the classic tobermorite-based approach.
        
        Parameters
        ----------
        supercell : Sequence[int], optional
            Supercell dimensions [a, b, c].
            Defaults: [3,5,1] for 'tob11a_hamid.cif', [4,1,1] for 'tob11a_merlino.cif'
        model : str, default='tob11a_merlino.cif'
            Tobermorite model ('tob11a_hamid.cif' or 'tob11a_merlino.cif').
        min_mcl : float, default=3.0
            Minimum mean chain length.
        symmetry : bool, default=True
            Remove bridging silicates symmetrically.
        
        Returns
        -------
        AtomicSystem
            The built C-S-H structure.
        
        Examples
        --------
        >>> builder = CSHBuilder(cs_ratio=1.5, ws_ratio=1.0)
        >>> system = builder.build(
        ...     supercell=[4,1,1],
        ...     model='tob11a_merlino.cif'
        ... )
        """

        from ._silicate_helpers import (
            remove_bridging_silicates, 
            calculate_csh_modifiers,
            neutralize_csh_charge
        )
        from ._interlayer_helpers import (
            distribute_species_in_layers, 
            fill_csh_interlayers
)

        require_program('packmol')
        
        # Validate classic-specific parameters
        if model not in ['tob11a_hamid.cif', 'tob11a_merlino.cif']:
            raise ValueError(f"Unsupported model: {model}")
        if min_mcl < 2:
            raise ValueError(f"min_mcl must be >= 2, got {min_mcl}")
        
        system = AtomicSystem.from_file(os.path.join(STRUCTURES_DIR, model))
        
        # Apply supercell
        if model == "tob11a_hamid.cif":
            supercell = supercell or [3, 5, 1]
            system.replicate(supercell)
            system.unskew()
        elif model == "tob11a_merlino.cif":
            system.orthogonalize()
            supercell = supercell or [4, 1, 1]
            system.replicate(supercell)
        
        univ = system.to_mda()
        box = univ.dimensions
        
        # Calculate modifications
        nsi = len(univ.select_atoms("type Si"))
        nca = len(univ.select_atoms("type Ca"))
        nsi_to_remove, nca_to_add, vacancy_fraction = calculate_csh_modifiers(
            nsi, nca, self.cs_ratio, min_mcl
        )
        
        if _progress_callback:
            _progress_callback(10, "Removing bridging silicates...")
        
        univ = remove_bridging_silicates(univ, nsi_to_remove, symmetry)
        
        # Update Si count
        si_sel = univ.select_atoms("type Si")
        nsi = len(si_sel)
        
        # Calculate water count
        num_water = round(nsi * self.ws_ratio) + nca_to_add
        
        if num_water == 0:
            final_system = AtomicSystem.from_mda(univ)
            final_system.set_box(box)
            return final_system
        
        # Distribute species in layers
        si_layers_pos, nw_layers, nca_layers, nlayers = distribute_species_in_layers(
            num_water, nca_to_add, si_sel.positions[:, 2], 
            supercell[2] if supercell else 1, box[2]
        )
        
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            final_pdb = fill_csh_interlayers(
                univ, box, si_layers_pos, nw_layers, nca_layers,
                nca_to_add, nlayers, tmp, _progress_callback
            )
            
            if nca_to_add != 0:
                neutralize_csh_charge(final_pdb)
            
            final_system = AtomicSystem.from_file(final_pdb)
        
        final_system.set_box(box)
        
        if _progress_callback:
            _progress_callback(100, "Structure complete")
        
        self._system = final_system
        self._update_analysis()
        return final_system
    
    # ------------------------------------------------------------------
    # pyCSH method
    # ------------------------------------------------------------------
    
    def build_pycsh(self,
                    supercell: Sequence[int] = (3, 5, 2),
                    nsamples: int = 1,
                    name: Optional[str] = None,
                    seed: Optional[int] = None,
                    _progress_callback: Optional[Callable[[int, str], None]] = None) -> Union[AtomicSystem, list[AtomicSystem]]:
        """
        Build C-S-H using pyCSH (full atomistic model).
        
        Parameters
        ----------
        supercell : Sequence[int], default=(3,5,2)
            Supercell dimensions [a, b, c].
        nsamples : int, default=1
            Number of samples to generate.
        name : str, optional
            Prefix for output files.
        seed : int, optional
            Random seed for reproducibility.
        
        Returns
        -------
        AtomicSystem or list[AtomicSystem]
            The built C-S-H structure(s). Returns a single AtomicSystem if
            nsamples=1, otherwise a list.
        
        Examples
        --------
        >>> builder = CSHBuilder(cs_ratio=1.2, ws_ratio=0.8)
        >>> system = builder.build_pycsh(
        ...     supercell=[3,5,2],
        ...     nsamples=1,
        ...     seed=42
        ... )
        >>> 
        >>> # Generate multiple samples
        >>> systems = builder.build_pycsh(
        ...     supercell=[3,5,2],
        ...     nsamples=3
        ... )
        """
        # Validate pycsh-specific parameters
        if nsamples < 1:
            raise ValueError(f"nsamples must be >= 1, got {nsamples}")
        
        if len(supercell) != 3:
            raise ValueError(f"supercell must have 3 elements, got {len(supercell)}")
        
        output_path = os.path.join(PYCSH_DIR, "output")
        
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        seed = seed or int.from_bytes(os.urandom(4), 'big')
        name = name or f"cs{self.cs_ratio}_ws{self.ws_ratio}"
        
        os.chdir(PYCSH_DIR)
        with open(os.path.join(PYCSH_DIR, "parameters.py"), 'w') as f:
            f.write(f"seed = {seed}\n")
            f.write(f"shape = {supercell}\n")
            f.write(f"Ca_Si_ratio = {self.cs_ratio}\n")
            f.write(f"W_Si_ratio = {self.ws_ratio}\n")
            f.write(f"N_samples = {nsamples}\n")
            f.write(f"create = True\n")
            f.write(f"write_lammps = True\n")
            f.write(f"write_lammps_erica = False\n")
            f.write(f"write_vasp = False\n")
            f.write(f"write_siesta = False\n")
            f.write(f'prefix = "{name}"\n')
        
        try:
            pattern = re.compile(r"Structure\s+(\d+)\s+converged")
            
            process = subprocess.Popen(
                ['python', '-u', "main_brick.py"],
                cwd=PYCSH_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            for line in process.stdout:
                match = pattern.search(line)
                if match and _progress_callback:
                    current_id = int(match.group(1)) + 1
                    _progress_callback(current_id, f"pyCSH: structure {current_id}/{nsamples}")
            
            process.wait()
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error executing pyCSH: {e}")
        
        # Collect results
        atomic_systems = []
        for i in range(1, nsamples + 1):
            reax_name = f"{name}_reax{i}.data"
            lmp_path = os.path.join(output_path, reax_name)
            
            if not os.path.exists(lmp_path):
                raise FileNotFoundError(f"pyCSH did not generate {reax_name}")
            
            data = AtomicSystem.from_file(lmp_path)
            data.reset_types()
            atomic_systems.append(data)
        
        result = atomic_systems[0] if len(atomic_systems) == 1 else atomic_systems
        self._system = result if isinstance(result, AtomicSystem) else result[0]
        self._update_analysis()
        return result
    
    # ------------------------------------------------------------------
    # Conversion to C-A-S-H
    # ------------------------------------------------------------------
    
    def to_cash(self, 
                as_ratio: float, 
                supercell_factor: int = 1,
                analyze_before: bool = True,
                analyze_after: bool = True) -> tuple[AtomicSystem, float, Optional[dict], Optional[dict]]:
        """
        Convert C-S-H to C-A-S-H by substituting Si with Al.
        """
        if self._system is None:
            raise RuntimeError("No system available. Build one first or use from_system().")
        
        system = self._system
        
        # Analyze before
        before_analysis = None
        if analyze_before:
            before_analysis = analyze_silicates(system)
            if self.progress_callback:
                self.progress_callback(10, f"Before: MCL={before_analysis['MCL']:.2f}")
        
        # Perform Al substitution
        if self.progress_callback:
            self.progress_callback(30, f"Substituting Si with Al (target Al/Si={as_ratio})...")
        
        universe = system.to_mda()
        si_sel = universe.select_atoms("type Si")
        
        if len(si_sel) == 0:
            raise ValueError("No silicon atoms found in the system.")
        
        nsi = len(si_sel)
        nal_target = round(as_ratio * nsi)
        
        universe, substituted_ids = substitute_si_by_al(
            universe, nal_target, supercell_factor
        )
        
        system.set_type2atoms(substituted_ids, "Al")
        system.keep_connection_types(
            bond_types=[('Hsi', 'Osih'), ('Ha', 'Oah'), ('Hh', 'Oh'), ('Hw', 'Ow')],
            angle_types=[('Hw', 'Ow', 'Hw')]
        )
        
        actual_ratio = nal_target / nsi
        
        if self.progress_callback:
            self.progress_callback(70, f"Substitution complete. Actual Al/Si={actual_ratio:.3f}")
        
        # Analyze after
        after_analysis = None
        if analyze_after:
            after_analysis = analyze_silicates(system)
            if self.progress_callback:
                self.progress_callback(90, f"After: MCL={after_analysis['MCL']:.2f}")
        
        if self.progress_callback:
            self.progress_callback(100, "C-A-S-H conversion complete")
        
        self._system = system
        self._update_analysis()
        return system, actual_ratio, before_analysis, after_analysis

@dataclass
class AFBuilder(BaseBuilder):
    """
    Builder for creating AFt (Ettringite) or AFm structures.
    
    This builder creates AFt/AFm structures by adding water molecules to
    a sulfate-containing framework.
    
    Parameters
    ----------
    ws_ratio : float
        Water/Sulfate ratio.
    progress_callback : Callable, optional
        Progress callback function.
    
    Examples
    --------
    >>> # Create AFt
    >>> builder = AFBuilder(ws_ratio=10.0)
    >>> system = builder.build_aft(
    ...     supercell=[2,2,1]
    ... )
    >>> 
    >>> # Create AFm
    >>> builder = AFBuilder(ws_ratio=8.0)
    >>> system = builder.build_afm(
    ...     supercell=[3,1,1]
    ... )
    """
    
    # Required parameters
    ws_ratio: float = field(default=None)
    
    # Optional parameter (commun)
    progress_callback: Optional[Callable[[int, str], None]] = None
    
    # Internal state
    _result: Optional[AtomicSystem] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize and validate the builder."""
        super().__init__(None)
        if self.ws_ratio is None or self.ws_ratio < 0:
            raise ValueError(f"ws_ratio must be >= 0, got {self.ws_ratio}")
    
    def __repr__(self) -> str:
        """Return a styled string representation."""
        lines = ["<AFBuilder>"]
        lines.append("")
        lines.append("┌─ Configuration")
        lines.append(f"│   ws_ratio   : {self.ws_ratio:.3f}")
        lines.append("└─")
        return "\n".join(lines)
    
    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------
    
    def build_aft(self,
                  supercell: Optional[Sequence[int]] = None,
                  model_file: str = 'aft_moore.cif') -> AtomicSystem:
        """
        Build AFt (Ettringite) structure.
        
        Parameters
        ----------
        supercell : Sequence[int], optional
            Supercell dimensions [a, b, c].
        model_file : str, default='aft_moore.cif'
            CIF file for the AFt structure.
        
        Returns
        -------
        AtomicSystem
            The built AFt structure.
        
        Examples
        --------
        >>> builder = AFBuilder(ws_ratio=10.0)
        >>> system = builder.build_aft(supercell=[2,2,1])
        """
        return self._build_af('aft', supercell, model_file)
    
    def build_afm(self,
                  supercell: Optional[Sequence[int]] = None,
                  model_file: str = 'afm.cif',
                  _progress_callback: Optional[Callable[[int, str], None]] = None) -> AtomicSystem:
        """
        Build AFm structure.
        
        Parameters
        ----------
        supercell : Sequence[int], optional
            Supercell dimensions [a, b, c].
        model_file : str, default='afm.cif'
            CIF file for the AFm structure.
        
        Returns
        -------
        AtomicSystem
            The built AFm structure.
        
        Examples
        --------
        >>> builder = AFBuilder(ws_ratio=8.0)
        >>> system = builder.build_afm(supercell=[3,1,1])
        """
        return self._build_af('afm', supercell, model_file, _progress_callback)
    
    def _build_af(self,
                  structure_type: str,
                  supercell: Optional[Sequence[int]] = None,
                  model_file: str = 'aft_moore.cif',
                  _progress_callback: Optional[Callable[[int, str], None]] = None) -> AtomicSystem:
        """Internal method to build AFt/AFm structures."""
        require_program('packmol')

        from ._packmol import get_structure_path, add_packmol_structure, run_packmol
        
        if supercell is not None and len(supercell) != 3:
            raise ValueError(f"supercell must have 3 elements, got {len(supercell)}")
        
        # Load base structure
        input_system = AtomicSystem.from_file(
            os.path.join(STRUCTURES_DIR, model_file)
        )
        
        if supercell is not None:
            input_system.replicate(supercell)
        
        input_system.orthogonalize()
        
        # Get information with MDAnalysis
        univ = input_system.to_mda()
        box = univ.dimensions
        
        # Count sulfur atoms
        s_sel = univ.select_atoms("type S")
        num_s = len(s_sel)
        num_water = round(num_s * self.ws_ratio)
        
        # Get reference atom for Packmol
        sel = univ.select_atoms("all")
        idmin = np.argmin(sel.positions[:, 2]) + 1
        
        if _progress_callback:
            _progress_callback(10, f"Building {structure_type.upper()}...")
        
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            tempfile_ipdb = os.path.join(tmp, 'itmp.pdb')
            sel.write(tempfile_ipdb)
            
            h2o_pdb = get_structure_path('h2o', tmp)
            
            structures = [
                add_packmol_structure(
                    tempfile_ipdb,
                    1,
                    f"inside box 0 0 0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}",
                    f"atoms {idmin}",
                    "fixed 0 0 0 0 0 0",
                    "end atoms"
                ),
                add_packmol_structure(
                    h2o_pdb,
                    num_water,
                    f"inside box 0 0 0 {box[0]:.4f} {box[1]:.4f} {box[2]:.4f}"
                )
            ]
            
            if _progress_callback:
                _progress_callback(50, f"Adding {num_water} water molecules...")
            
            output_system = run_packmol(structures)
        
        # Set box and topology
        output_system.set_box(box)
        output_system.set_topo()
        
        if _progress_callback:
            _progress_callback(100, f"{structure_type.upper()} complete")
        
        self._result = output_system
        return self._result