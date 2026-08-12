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
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    AtomType,
    BuckinghamParams,
    Class2AngleParams,
    Class2BondAngleParams,
    Class2BondBondParams,
    Class2BondParams,
    DistanceImproperParams,
    ForceFieldModel,
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicImproperParams,
    LJParams,
)

# ===============================================================================
# Mapping keywords to files
# ===============================================================================

FILE_MAPPING = {
    # TOML force fields
    "clayff": "clayff.toml",
    "cshff": "cshff2014.toml",
    "cshff2014": "cshff2014.toml",
    "guillot": "guillot2007.toml",
    "raiteri": "raiteri2015.toml",
    "raiteri2015": "raiteri2015.toml",
    
    # GROMOS
    "gromos": "GROMOS_54A7_ATB.lt",

    # INTERFACE CVFF
    "iff-cvff": "cvff_interface_v1_5.frc",

    # INTERFACE CHARMM
    "iff-charmm": "charmm27_interface_v1_5.prm"
}


# ============================================================================
# ForceFieldDatabase
# ============================================================================

class ForceFieldDatabase:
    """
    Force field database loaded by keywords.
    
    Usage:
        db = ForceFieldDatabase("clayff")           # Load clayff
        db = ForceFieldDatabase(["clayff", "gromos"]) # Load multiple
        db = ForceFieldDatabase("all")              # Load all available
        db = ForceFieldDatabase("all-toml")             # Load all TOML files
        db = ForceFieldDatabase()                   # Empty database
    
    Attributes
    ----------
    atom : dict[str, AtomType]
        Dictionary of atom types {full_name: AtomType}
    lj : dict[str, LJParams]
        Dictionary of LJ parameters {model.pair: LJParams}
    buckingham : dict[str, BuckinghamParams]
        Dictionary of Buckingham parameters {model.pair: BuckinghamParams}
    bond : dict[str, HarmonicBondParams | Class2AngleParams]
        Dictionary of bond parameters {model.pair: BondParams}
    angle : dict[str, HarmonicAngleParams | Class2AngleParams]
        Dictionary of angle parameters {model.triple: AngleParams}
    improper : dict[str, HarmonicAngleParams]
        Dictionary of improper parameters {model.quad: AngleParams}
    bondbond: dict[str, Class2BondBondParams]
    bondangle: dict[str, Class2BondAngleParams]
    models : dict[str, ForceFieldModel]
        Dictionary of model metadata {name: ForceFieldModel}
    dihedral : dict[str, Any]
        Dictionary of dihedral parameters (format specific)
    loaded_keywords : list[str]
        List of keywords that were used to load force fields
    """

    def __init__(
        self,
        keywords: str | list[str] | None = None,
        db_dir: str | Path | None = None,
    ):
        """
        Initialize the force field database.

        Parameters
        ----------
        keywords : str or list[str], optional
            Keyword(s) to load specific force fields.
            -"all": loads all supported files
            -"all-toml": Load all TOML files only
            -"clayff": loads clayff.toml
            -"gromos": loads GROMOS_54A7_ATB.lt
            -["clayff", "gromos"]: loads multiple
            -None: creates empty database
        db_dir : str or Path, optional
            Custom directory containing force field database files.
            Defaults to standard package database directory.
        """
        # Déterminer le chemin du dossier des forcefields
        self.db_dir = str(db_dir) if db_dir is not None else self._get_forcefield_dir()
        
        self.atom: dict[str, AtomType] = {}
        self.lj: dict[str, LJParams] = {}
        self.buckingham: dict[str, BuckinghamParams] = {}
        self.bond: dict[str, HarmonicBondParams | Class2BondParams] = {}
        self.angle: dict[str, HarmonicAngleParams | Class2AngleParams] = {}
        self.improper: dict[str, HarmonicImproperParams | DistanceImproperParams] = {}
        self.bondbond: dict[str, Class2BondBondParams] = {}
        self.bondangle: dict[str, Class2BondAngleParams] = {}
        self.models: dict[str, ForceFieldModel] = {}
        self.dihedral: dict[str, Any] = {}
        self.loaded_keywords: list[str] = []

        # Load the requested forcefields
        if keywords is not None:
            self.load(keywords)

    # ===============================================================================
    # Loading Methods
    # ===============================================================================

    @staticmethod
    def _get_forcefield_dir() -> str:
        # 2. Default: package directory
        current_dir = Path(__file__).parent
        default_dir = current_dir / "db"
        if default_dir.exists():
            return str(default_dir)
        
        # 3. Try current working directory
        cwd_dir = Path.cwd() / "db"
        if cwd_dir.exists():
            return str(cwd_dir)
        
        # 4. Create default directory if it doesn't exist
        default_dir.mkdir(parents=True, exist_ok=True)
        return str(default_dir)

    def load(self, keywords: str | list[str]) -> None:
        """
        Load force fields by keyword(s).
        
        Parameters
        ----------
        keywords : str or list[str]
            Keyword(s) to load. Use "all" to load all available.
        """
        if keywords == "all":
            self._load_all()
            return

        if keywords == "all-toml":
            self._load_all_toml()
            return
        
        if isinstance(keywords, str):
            keywords = [keywords]
        
        for keyword in keywords:
            self._load_keyword(keyword.strip())

    def _load_keyword(self, keyword: str) -> None:
        """
        Load a single force field by keyword.
        
        Parameters
        ----------
        keyword : str
            Keyword identifying the force field
        """
        # Vérifier si le mot-clé existe
        if keyword not in FILE_MAPPING:
            # Try case insensitive
            keyword_lower = keyword.lower()
            for key in FILE_MAPPING:
                if key.lower() == keyword_lower:
                    keyword = key
                    break
            else:
                print(f"Warning: Unknown keyword '{keyword}'. Skipping.")
                return
        
        filename = FILE_MAPPING[keyword]
        filepath = os.path.join(self.db_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File '{filename}' not found in {self.db_dir}. Skipping.")
            return
        
        # Load according to extension
        if filename.endswith(".toml"):
            with open(filepath, "rb") as f:
                model_data = tomllib.load(f)
            model_name = os.path.splitext(filename)[0]
            self._load_model_from_dict(model_name, model_data)
            self.loaded_keywords.append(keyword)
            
        elif filename == "GROMOS_54A7_ATB.lt":
            self._load_gromos_lt(filepath, keyword)
            self.loaded_keywords.append(keyword)
            
        elif filename.endswith("charmm27_interface_v1_5.prm"):
            self._load_charmm_prm(filepath, keyword)
            self.loaded_keywords.append(keyword)

        elif filename.endswith("cvff_interface_v1_5.frc"):
            self._load_cvff_frc(filepath, keyword)
            self.loaded_keywords.append(keyword)
        
        else:
            print(f"Warning: Unsupported file format for '{filename}'. Skipping.")

    def _load_all(self) -> None:
        """Load all supported force field files."""
        # Global metadata
        metadata_path = os.path.join(self.db_dir, "_metadata.toml")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = tomllib.load(f)
                for name, data in metadata.get("models", {}).items():
                    self.models[name] = ForceFieldModel(
                        name=data.get("name", name),
                        description=data.get("description", ""),
                        ref=data.get("ref", ""),
                        tags=data.get("tags", []),
                    )

        # Browse all files in folder
        for filename in sorted(os.listdir(self.db_dir)):
            if filename == "_metadata.toml":
                continue
            
            filepath = os.path.join(self.db_dir, filename)
            
            # Toml
            if filename.endswith(".toml"):
                with open(filepath, "rb") as f:
                    model_data = tomllib.load(f)
                model_name = os.path.splitext(filename)[0]
                self._load_model_from_dict(model_name, model_data)
                self.loaded_keywords.append(model_name)
            
            # GROMOS .lt
            elif filename.endswith(".lt"):
                model_name = os.path.splitext(filename)[0]
                self._load_gromos_lt(filepath, model_name)
                self.loaded_keywords.append(model_name)
            
            # CHARMM .par
            elif filename.endswith(".par"):
                model_name = os.path.splitext(filename)[0]
                self._load_charmm_prm(filepath, model_name)
                self.loaded_keywords.append(model_name)
            
            # AMBER .frcmod
            elif filename.endswith(".frcmod"):
                model_name = os.path.splitext(filename)[0]
                self._load_amber_frcmod(filepath, model_name)
                self.loaded_keywords.append(model_name)
            
            # AMBER .parm7
            elif filename.endswith(".parm7"):
                model_name = os.path.splitext(filename)[0]
                self._load_amber_parm(filepath, model_name)
                self.loaded_keywords.append(model_name)

    def _load_all_toml(self) -> None:
        """Load all TOML force field files from the database directory."""
        # Global metadata
        metadata_path = os.path.join(self.db_dir, "_metadata.toml")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                metadata = tomllib.load(f)
                for name, data in metadata.get("models", {}).items():
                    self.models[name] = ForceFieldModel(
                        name=data.get("name", name),
                        description=data.get("description", ""),
                        ref=data.get("ref", ""),
                        tags=data.get("tags", []),
                    )

        # Browse all TOML files
        for filename in sorted(os.listdir(self.db_dir)):
            if filename == "_metadata.toml":
                continue
            
            if not filename.endswith(".toml"):
                continue
            
            filepath = os.path.join(self.db_dir, filename)
            with open(filepath, "rb") as f:
                model_data = tomllib.load(f)
            
            model_name = os.path.splitext(filename)[0]
            self._load_model_from_dict(model_name, model_data)
            self.loaded_keywords.append(f"{model_name} (toml)")


    def _load_model_from_dict(self, model_name: str, model_data: dict) -> None:
        """Load a single model from its data dictionary."""
        if model_name not in self.models:
            self.models[model_name] = ForceFieldModel(
                name=model_name,
                description=model_data.get("description", ""),
                ref=model_data.get("ref", ""),
                tags=model_data.get("tags", []),
            )

        # Atom types
        for key, params in model_data.get("atom", {}).items():
            full_key = f"{model_name}.{key}"
            self.atom[full_key] = AtomType(
                element=params["element"],
                charge=params.get("charge", 0.0),
                environment=params.get("environment", ""),
                ref=params.get("ref", ""),
                mass=params.get("mass"),
                model=model_name,
            )

        # Lj
        for key, params in model_data.get("lj", {}).items():
            self.lj[f"{model_name}.{key}"] = LJParams(
                epsilon=params["epsilon"],
                sigma=params["sigma"],
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Buckingham
        for key, params in model_data.get("buckingham", {}).items():
            self.buckingham[f"{model_name}.{key}"] = BuckinghamParams(
                a=params["A"],
                rho=params["rho"],
                c=params.get("C", 0.0),
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Bond
        bond_data = model_data.get("bond", {})
        for key, params in bond_data.get("harmonic", {}).items():
            self.bond[f"{model_name}.{key}"] = HarmonicBondParams(
                k=params["k"],
                r0=params["r0"],
                ref=params.get("ref", ""),
                model=model_name,
            )

        for key, params in bond_data.get("class2", {}).items():
            self.bond[f"{model_name}.{key}"] = Class2BondParams(
                r0=params["r0"],
                k2=params["k2"],
                k3=params.get("k3", 0.0),
                k4=params.get("k4", 0.0),
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Angle
        angle_data = model_data.get("angle", {})
        for key, params in angle_data.get("harmonic", {}).items():
            self.angle[f"{model_name}.{key}"] = HarmonicAngleParams(
                k=params["k"],
                theta0=params["theta0"],
                ref=params.get("ref", ""),
                model=model_name,
            )

        for key, params in angle_data.get("class2", {}).items():
            self.angle[f"{model_name}.{key}"] = Class2AngleParams(
                theta0=params["theta0"],
                k2=params["k2"],
                k3=params.get("k3", 0.0),
                k4=params.get("k4", 0.0),
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Improper
        improper_data = model_data.get("improper", {})
        for key, params in improper_data.get("distance", {}).items():
            self.improper[f"{model_name}.{key}"] = DistanceImproperParams(
                k2=params["k2"],
                k4=params.get("k4", 0.0),
                ref=params.get("ref", ""),
                model=model_name,
            )

        for key, params in improper_data.get("harmonic", {}).items():
            self.improper[f"{model_name}.{key}"] = HarmonicImproperParams(
                k=params["k"],
                chi0=params.get("chi0", 0.0),
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Bondbond
        bondbond_data = model_data.get("bondbond", {})
        for key, params in bondbond_data.get("class2", {}).items():
            self.bondbond[f"{model_name}.{key}"] = Class2BondBondParams(
                m=params["m"],
                r1=params["r1"],
                r2=params["r2"],
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Bondangle
        bondangle_data = model_data.get("bondangle", {})
        for key, params in bondangle_data.get("class2", {}).items():
            self.bondangle[f"{model_name}.{key}"] = Class2BondAngleParams(
                n1=params["n1"],
                n2=params["n2"],
                r1=params["r1"],
                r2=params["r2"],
                ref=params.get("ref", ""),
                model=model_name,
            )

        # Dihedral
        if "dihedral" in model_data:
            for key, params in model_data["dihedral"].items():
                self.dihedral[f"{model_name}.{key}"] = params

    def _load_gromos_lt(self, filepath: str, model_name: str) -> None:
        """Load a GROMOS force field from a moltemplate .lt file."""
        from ._parsers._gromos import GromosForceFieldLoader
        loader = GromosForceFieldLoader(self)
        loader.load_from_file(filepath, model_name)

    def _load_charmm_prm(self, filepath: str, model_name: str) -> None:
        """Load a CHARMM force field from a .prm file."""
        from ._parsers._iff_charmm import CHARMMInterfaceLoader
        loader = CHARMMInterfaceLoader(self)
        loader.load_from_file(filepath, model_name)

    def _load_cvff_frc(self, filepath: str, model_name: str) -> None:
        """Load a CVFF force field from a .frc file."""
        from ._parsers._iff_cvff import CVFFInterfaceLoader
        loader = CVFFInterfaceLoader(self)
        loader.load_from_file(filepath, model_name)

    def _load_amber_frcmod(self, filepath: str, model_name: str) -> None:
        """Load an AMBER force field from a .frcmod file."""
        from ._parsers._amber import AmberForceFieldLoader
        loader = AmberForceFieldLoader(self)
        loader.load_from_file(filepath, model_name)

    def _load_amber_parm(self, filepath: str, model_name: str) -> None:
        """Load an AMBER force field from a .parm7 file."""
        from ._parsers._amber import AmberParmLoader
        loader = AmberParmLoader(self)
        loader.load_from_file(filepath, model_name)

    # ===============================================================================
    # Research methods
    # ===============================================================================

    def _extract_short_name(self, name: str) -> tuple[str | None, str]:
        """Extract model and short name from a full name."""
        if "." in name:
            model, short = name.split(".", 1)
            return model, short
        return None, name

    def get_atom_type(self, name: str) -> AtomType | None:
        """Get atom type by full name (model.type) or short name."""
        if name in self.atom:
            return self.atom[name]

        _, short = self._extract_short_name(name)
        for full_key, params in self.atom.items():
            if full_key.endswith(f".{short}"):
                return params
        return None

    def _get_pair_params(self, dict_obj: dict, type1: str, type2: str, model: str = None):
        """Generic method to search for pair parameters."""
        _, short1 = self._extract_short_name(type1)
        _, short2 = self._extract_short_name(type2)

        if model is None:
            m1, _ = self._extract_short_name(type1)
            if m1 is not None:
                model = m1

        key = f"{short1}-{short2}"
        key_rev = f"{short2}-{short1}"

        if model:
            for k in [f"{model}.{key}", f"{model}.{key_rev}"]:
                if k in dict_obj:
                    return dict_obj[k]
            return None

        for full_key, params in dict_obj.items():
            if "." in full_key:
                pair_key = full_key.split(".", 1)[1]
                if pair_key == key or pair_key == key_rev:
                    return params
        return None

    def _get_triple_params(self, dict_obj: dict, type1: str, type2: str, type3: str, model: str = None):
        """Generic method to search for triple parameters."""
        _, short1 = self._extract_short_name(type1)
        _, short2 = self._extract_short_name(type2)
        _, short3 = self._extract_short_name(type3)

        if model is None:
            m1, _ = self._extract_short_name(type1)
            if m1 is not None:
                model = m1

        key = f"{short1}-{short2}-{short3}"
        key_rev = f"{short3}-{short2}-{short1}"

        if model:
            for k in [f"{model}.{key}", f"{model}.{key_rev}"]:
                if k in dict_obj:
                    return dict_obj[k]
            return None

        for full_key, params in dict_obj.items():
            if "." in full_key:
                triple_key = full_key.split(".", 1)[1]
                if triple_key == key or triple_key == key_rev:
                    return params
        return None

    def _get_quadruple_params(self, dict_obj: dict, type1: str, type2: str, type3: str, type4: str, model: str = None):
        """Generic method to search for quadruple parameters."""
        _, short1 = self._extract_short_name(type1)
        _, short2 = self._extract_short_name(type2)
        _, short3 = self._extract_short_name(type3)
        _, short4 = self._extract_short_name(type4)

        if model is None:
            m1, _ = self._extract_short_name(type1)
            if m1 is not None:
                model = m1

        key = f"{short1}-{short2}-{short3}-{short4}"
        key_rev = f"{short4}-{short3}-{short2}-{short1}"

        if model:
            for k in [f"{model}.{key}", f"{model}.{key_rev}"]:
                if k in dict_obj:
                    return dict_obj[k]
            return None

        for full_key, params in dict_obj.items():
            if "." in full_key:
                quad_key = full_key.split(".", 1)[1]
                if quad_key == key or quad_key == key_rev:
                    return params
        return None

    def get_lj(self, type1: str, type2: str, model: str = None) -> LJParams | None:
        """Get Lennard-Jones 12-6 parameters for a pair."""
        return self._get_pair_params(self.lj, type1, type2, model)

    def get_buckingham(self, type1: str, type2: str, model: str = None) -> BuckinghamParams | None:
        """Get Buckingham potential parameters for a pair."""
        return self._get_pair_params(self.buckingham, type1, type2, model)

    def get_bond(self, type1: str, type2: str, model: str = None) -> HarmonicBondParams | Class2BondParams | None:
        """Get bond parameters for a pair."""
        return self._get_pair_params(self.bond, type1, type2, model)

    def get_angle(self, type1: str, type2: str, type3: str, model: str = None) -> HarmonicAngleParams | Class2AngleParams | None:
        """Get angle parameters for a triple of atom types."""
        return self._get_triple_params(self.angle, type1, type2, type3, model)

    def get_bondbond(self, type1: str, type2: str, type3: str, model: str = None) -> Class2BondBondParams | None:
        """Get bondbond parameters for a triple of atom types."""
        return self._get_triple_params(self.bondbond, type1, type2, type3, model)

    def get_bondangle(self, type1: str, type2: str, type3: str, model: str = None) -> Class2BondAngleParams | None:
        """Get bondangle parameters for a triple of atom types."""
        return self._get_triple_params(self.bondangle, type1, type2, type3, model)

    def get_dihedral(self, type1: str, type2: str, type3: str, type4: str, model: str = None) -> Any | None:
        """Get dihedral parameters for a quadruple of atom types."""
        return self._get_quadruple_params(self.dihedral, type1, type2, type3, type4, model)

    def get_improper(self, type1: str, type2: str, type3: str, type4: str, model: str = None) -> HarmonicImproperParams | DistanceImproperParams | None:
        """Get improper parameters for a quadruple of atom types."""
        return self._get_quadruple_params(self.improper, type1, type2, type3, type4, model)

    # ===============================================================================
    # Utilities 
    # ===============================================================================

    def get_model_names(self) -> list[str]:
        """Get list of available model names."""
        return list(self.models.keys())

    def get_model(self, name: str) -> ForceFieldModel | None:
        """Get model metadata by name."""
        return self.models.get(name)

    def get_atom_types_for_model(self, model: str) -> list[str]:
        """Get all atom type names for a specific model."""
        prefix = f"{model}."
        return [key for key in self.atom.keys() if key.startswith(prefix)]

    def get_loaded_keywords(self) -> list[str]:
        """Get list of keywords that were used to load force fields."""
        return self.loaded_keywords

    def clear(self) -> None:
        """Clear all loaded data."""
        self.atom.clear()
        self.lj.clear()
        self.buckingham.clear()
        self.bond.clear()
        self.angle.clear()
        self.improper.clear()
        self.bondbond.clear()
        self.bondangle.clear()
        self.models.clear()
        self.dihedral.clear()
        self.loaded_keywords.clear()

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Convert to pandas DataFrames for compatibility."""
        import pandas as pd

        dfs = {}

        # Atom types
        records = []
        for key, params in self.atom.items():
            short_type = key.split(".")[-1] if "." in key else key
            records.append({
                "type": short_type,
                "full_type": key,
                "element": params.element,
                "charge": params.charge,
                "model": params.model,
                "environment": params.environment,
                "ref": params.ref,
                "mass": params.mass,
            })
        dfs["atoms"] = pd.DataFrame(records)

        # LJ parameters
        records = []
        for key, params in self.lj.items():
            pair = key.split(".")[-1] if "." in key else key
            records.append({
                "pair": pair,
                "full_key": key,
                "epsilon": params.epsilon,
                "sigma": params.sigma,
                "model": params.model,
                "ref": params.ref,
            })
        dfs["lj"] = pd.DataFrame(records)

        # Bonds
        records = []
        for key, params in self.bond.items():
            bond_id = key.split(".")[-1] if "." in key else key
            if hasattr(params, 'k') and hasattr(params, 'r0'):
                records.append({
                    "bond": bond_id,
                    "full_key": key,
                    "type": "harmonic",
                    "k": params.k,
                    "r0": params.r0,
                    "model": params.model,
                    "ref": params.ref,
                })
            elif hasattr(params, 'k2') and hasattr(params, 'r0'):
                records.append({
                    "bond": bond_id,
                    "full_key": key,
                    "type": "class2",
                    "k2": params.k2,
                    "k3": params.k3,
                    "k4": params.k4,
                    "r0": params.r0,
                    "model": params.model,
                    "ref": params.ref,
                })
        dfs["bonds"] = pd.DataFrame(records)

        # Buckingham
        records = []
        for key, params in self.buckingham.items():
            pair = key.split(".")[-1] if "." in key else key
            records.append({
                "pair": pair,
                "full_key": key,
                "a": getattr(params, "a", None),
                "rho": getattr(params, "rho", None),
                "c": getattr(params, "c", None),
                "model": params.model,
                "ref": params.ref,
            })
        dfs["buckingham"] = pd.DataFrame(records)

        # Angles
        records = []
        for key, params in self.angle.items():
            angle_id = key.split(".")[-1] if "." in key else key
            rec = {"angle": angle_id, "full_key": key, "model": params.model, "ref": params.ref}
            if hasattr(params, "k") and hasattr(params, "theta0"):
                rec.update({"type": "harmonic", "k": params.k, "theta0": params.theta0})
            elif hasattr(params, "k2") and hasattr(params, "theta0"):
                rec.update({
                    "type": "class2",
                    "theta0": params.theta0,
                    "k2": params.k2,
                    "k3": getattr(params, "k3", 0.0),
                    "k4": getattr(params, "k4", 0.0),
                })
            records.append(rec)
        dfs["angles"] = pd.DataFrame(records)

        # Dihedrals
        records = []
        for key, params in self.dihedral.items():
            dih_id = key.split(".")[-1] if "." in key else key
            records.append({"dihedral": dih_id, "full_key": key, "params": str(params)})
        dfs["dihedrals"] = pd.DataFrame(records)

        # Impropers
        records = []
        for key, params in self.improper.items():
            imp_id = key.split(".")[-1] if "." in key else key
            rec = {"improper": imp_id, "full_key": key, "model": getattr(params, "model", "")}
            if hasattr(params, "k"):
                rec.update({"k": getattr(params, "k"), "chi0": getattr(params, "chi0", 0.0)})
            elif hasattr(params, "k2"):
                rec.update({"k2": getattr(params, "k2"), "k4": getattr(params, "k4", 0.0)})
            records.append(rec)
        dfs["impropers"] = pd.DataFrame(records)

        return dfs

    def __repr__(self) -> str:
        return f"ForceFieldDatabase(models={list(self.models.keys())}, loaded={self.loaded_keywords})"