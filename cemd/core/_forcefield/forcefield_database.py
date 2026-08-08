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


class ForceFieldDatabase:
    """
    Force field database loaded from TOML files.

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
    """

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.atom: dict[str, AtomType] = {}
        self.lj: dict[str, LJParams] = {}
        self.buckingham: dict[str, BuckinghamParams] = {}
        self.bond: dict[str, HarmonicBondParams | Class2AngleParams] = {}
        self.angle: dict[str, HarmonicAngleParams | Class2AngleParams] = {}
        self.improper: dict[str, HarmonicAngleParams] = {}
        self.bondbond: dict[str, Class2BondBondParams] = {}
        self.bondangle: dict[str, Class2BondAngleParams] = {}
        self.models: dict[str, ForceFieldModel] = {}

        self._load_all()

    def _load_all(self) -> None:
        """Load all TOML files from the database directory."""
        # Métadonnées
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

        for filename in sorted(os.listdir(self.db_dir)):
            if filename == "_metadata.toml":
                continue

            if not filename.endswith(".toml"):
                continue

            filepath = os.path.join(self.db_dir, filename)
            with open(filepath, "rb") as f:
                model_data = tomllib.load(f)

            model_name = os.path.splitext(filename)[0]
            self._load_model(model_name, model_data)

    def _load_model(self, model_name: str, model_data: dict) -> None:
        """Load a single model from its data."""
        # Atom types
        for key, params in model_data.get("atom", {}).items():
            full_key = f"{model_name}.{key}"
            self.atom[full_key] = AtomType(
                element=params["element"],
                charge=params["charge"],
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
                A=params["A"],
                rho=params["rho"],
                C=params.get("C", 0.0),
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
                k3=params["k3"],
                k4=params["k4"],
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
                k3=params["k3"],
                k4=params["k4"],
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

    # ===============================================================================
    # Research methods
    # ===============================================================================

    def _extract_short_name(self, name: str) -> tuple[str | None, str]:
        """
        Extracts the model and short name of a type.

        Settings
        ----------
        name : str
            Nom complet (model.type) ou nom court (type)

        Returns
        -------
        tuple[Optional[str], str]
            (model, short_name)
        """
        if "." in name:
            model, short = name.split(".", 1)
            return model, short
        return None, name

    def get_atom_type(self, name: str) -> AtomType | None:
        """
        Get atom type by full name (model.type) or short name.
        """
        # Try exact match first
        if name in self.atom:
            return self.atom[name]

        # Try to find by short name
        _, short = self._extract_short_name(name)
        for full_key, params in self.atom.items():
            if full_key.endswith(f".{short}"):
                return params

        return None

    def _get_pair_params(
        self, dict_obj: dict, type1: str, type2: str, model: str = None
    ):
        """
        Generic method to search for pair parameters.
        """
        # Extract short names
        _, short1 = self._extract_short_name(type1)
        _, short2 = self._extract_short_name(type2)

        # If a template is provided, use it
        if model is None:
            # Try to extract the model of the first type
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

        # Search all models
        for full_key, params in dict_obj.items():
            if "." in full_key:
                pair_key = full_key.split(".", 1)[1]
                if pair_key == key or pair_key == key_rev:
                    return params
        return None

    def get_lj(self, type1: str, type2: str, model: str = None) -> LJParams | None:
        """Get Lennard-Jones 12-6 parameters for a pair."""
        return self._get_pair_params(self.lj, type1, type2, model)

    def get_buckingham(
        self, type1: str, type2: str, model: str = None
    ) -> BuckinghamParams | None:
        """Get Buckingham potential parameters for a pair."""
        return self._get_pair_params(self.buckingham, type1, type2, model)

    def get_bond(
        self, type1: str, type2: str, model: str = None
    ) -> HarmonicBondParams | None:
        """Get bond parameters for a pair."""
        return self._get_pair_params(self.bond, type1, type2, model)

    def get_angle(
        self, type1: str, type2: str, type3: str, model: str = None
    ) -> HarmonicAngleParams | Class2AngleParams | None:
        """
        Get angle parameters for a triple of atom types.
        """
        # Extraire les noms courts
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
                if k in self.angle:
                    return self.angle[k]
            return None

        for full_key, params in self.angle.items():
            if "." in full_key:
                triple_key = full_key.split(".", 1)[1]
                if triple_key == key or triple_key == key_rev:
                    return params
        return None

    def get_bondbond(
        self, type1: str, type2: str, type3: str, model: str = None
    ) -> Class2BondBondParams | None:
        """
        Get bondbond parameters for a triple of atom types.
        """
        # Extraire les noms courts
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
                if k in self.bondbond:
                    return self.bondbond[k]
            return None

        for full_key, params in self.bondbond.items():
            if "." in full_key:
                triple_key = full_key.split(".", 1)[1]
                if triple_key == key or triple_key == key_rev:
                    return params
        return None

    def get_bondangle(
        self, type1: str, type2: str, type3: str, model: str = None
    ) -> Class2BondAngleParams | None:
        """
        Get bondangle parameters for a triple of atom types.
        """
        # Extraire les noms courts
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
                if k in self.bondangle:
                    return self.bondangle[k]
            return None

        for full_key, params in self.bondangle.items():
            if "." in full_key:
                triple_key = full_key.split(".", 1)[1]
                if triple_key == key or triple_key == key_rev:
                    return params
        return None

    # ============================================================
    # Conversion
    # ============================================================

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Convert to pandas DataFrames for compatibility."""
        import pandas as pd

        dfs = {}

        # Atom types
        records = []
        for key, params in self.atom.items():
            short_type = key.split(".")[-1] if "." in key else key
            records.append(
                {
                    "type": short_type,
                    "full_type": key,
                    "element": params.element,
                    "charge": params.charge,
                    "model": params.model,
                    "environment": params.environment,
                    "ref": params.ref,
                    "mass": params.mass,
                }
            )
        dfs["list"] = pd.DataFrame(records)

        # # Lj
        # records = []
        # for key, params in self.lj.items():
        #     parts = key.split(".")
        #     model = parts[0] if len(parts) > 1 else ""
        #     pair = parts[1] if len(parts) > 1 else key
        #     types = pair.split("-")
        #     records.append({
        #         "type 1": types[0] if len(types) > 0 else "",
        #         "type 2": types[1] if len(types) > 1 else "",
        #         "epsilon (kcal/mol)": params.epsilon,
        #         "sigma (A)": params.sigma,
        #         "model": model,
        #         "ref": params.ref,
        #     })
        # dfs["lj_12-6"] = pd.DataFrame(records)

        # # Buckingham
        # records = []
        # for key, params in self.buckingham.items():
        #     parts = key.split(".")
        #     model = parts[0] if len(parts) > 1 else ""
        #     pair = parts[1] if len(parts) > 1 else key
        #     types = pair.split("-")
        #     records.append({
        #         "type 1": types[0] if len(types) > 0 else "",
        #         "type 2": types[1] if len(types) > 1 else "",
        #         "A (kcal/mol)": params.A,
        #         "rho (A)": params.rho,
        #         "C (kcal/mol.A6)": params.C,
        #         "model": model,
        #         "ref": params.ref,
        #     })
        # dfs["buckingham"] = pd.DataFrame(records)

        # # Bond
        # records = []
        # for key, params in self.bond.items():
        #     parts = key.split(".")
        #     model = parts[0] if len(parts) > 1 else ""
        #     pair = parts[1] if len(parts) > 1 else key
        #     types = pair.split("-")
        #     records.append({
        #         "type 1": types[0] if len(types) > 0 else "",
        #         "type 2": types[1] if len(types) > 1 else "",
        #         "k (kcal/(mol.A2)": params.k,
        #         "r (A)": params.r0,
        #         "model": model,
        #         "ref": params.ref,
        #     })
        # dfs["bond"] = pd.DataFrame(records)

        # # Angle
        # records = []
        # for key, params in self.angle.items():
        #     parts = key.split(".")
        #     model = parts[0] if len(parts) > 1 else ""
        #     triple = parts[1] if len(parts) > 1 else key
        #     types = triple.split("-")
        #     records.append({
        #         "type 1": types[0] if len(types) > 0 else "",
        #         "type 2": types[1] if len(types) > 1 else "",
        #         "type 3": types[2] if len(types) > 2 else "",
        #         "k (kcal/(mol.rad2))": params.k,
        #         "theta (deg)": params.theta0,
        #         "model": model,
        #         "ref": params.ref,
        #     })
        # dfs["angle"] = pd.DataFrame(records)

        return dfs
