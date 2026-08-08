# scripts/toml_to_json.py
"""
Convert TOML files to JSON for the documentation.
"""

import json
import tomllib
from pathlib import Path

from cemd._data.config import get_ff_db_path

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DOCS_DIR.parent

TOML_DIR = Path(get_ff_db_path())
OUTPUT_PATH = DOCS_DIR / "_static" / "ff_data.json"


def toml_to_json():
    """Convert all TOML files to a single JSON for the viewer."""

    data = {
        "metadata": {},
        "models": {},
        "list": [],
        "lj_12-6": [],
        "buckingham": [],
        "bond": [],
        "angle": [],
        "improper": [],
    }

    # 1. Lire les métadonnées
    metadata_path = TOML_DIR / "_metadata.toml"
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            data["metadata"] = tomllib.load(f)
            if "models" in data["metadata"]:
                data["models"] = data["metadata"]["models"]

    # 2. Parcourir tous les fichiers TOML
    for toml_path in sorted(TOML_DIR.glob("*.toml")):
        if toml_path.name == "_metadata.toml":
            continue

        with open(toml_path, "rb") as f:
            model_data = tomllib.load(f)

        model_name = toml_path.stem

        # Atom types → list
        for key, params in model_data.get("atom_types", {}).items():
            short_type = key.split(".")[-1] if "." in key else key

            data["list"].append(
                {
                    "type": short_type,
                    "full_type": f"{model_name}.{key}",
                    "element": params.get("element", ""),
                    "charge": params.get("charge", 0.0),
                    "model": model_name,
                    "environment": params.get("environment", ""),
                    "ref": params.get("ref", ""),  # Référence explicite
                    "mass": params.get("mass"),
                }
            )

        # LJ
        for key, params in model_data.get("lj", {}).items():
            parts = key.split("-")
            data["lj_12-6"].append(
                {
                    "type 1": parts[0] if len(parts) > 0 else "",
                    "type 2": parts[1] if len(parts) > 1 else "",
                    "epsilon (kcal/mol)": params.get("epsilon", 0.0),
                    "sigma (A)": params.get("sigma", 0.0),
                    "model": model_name,
                    "ref": params.get("ref", ""),  # Référence explicite
                }
            )

        # Buckingham
        for key, params in model_data.get("buckingham", {}).items():
            parts = key.split("-")
            data["buckingham"].append(
                {
                    "type 1": parts[0] if len(parts) > 0 else "",
                    "type 2": parts[1] if len(parts) > 1 else "",
                    "A (kcal/mol)": params.get("A", 0.0),
                    "rho (A)": params.get("rho", 0.0),
                    "C (kcal/mol.A6)": params.get("C", 0.0),
                    "model": model_name,
                    "ref": params.get("ref", ""),  # Référence explicite
                }
            )

        # Bond
        for key, params in model_data.get("bond", {}).items():
            parts = key.split("-")
            data["bond"].append(
                {
                    "type 1": parts[0] if len(parts) > 0 else "",
                    "type 2": parts[1] if len(parts) > 1 else "",
                    "k (kcal/(mol.A2)": params.get("k", 0.0),
                    "r (A)": params.get("r0", 0.0),
                    "model": model_name,
                    "ref": params.get("ref", ""),  # Référence explicite
                }
            )

        # Angle
        for key, params in model_data.get("angle", {}).items():
            parts = key.split("-")
            data["angle"].append(
                {
                    "type 1": parts[0] if len(parts) > 0 else "",
                    "type 2": parts[1] if len(parts) > 1 else "",
                    "type 3": parts[2] if len(parts) > 2 else "",
                    "k (kcal/(mol.rad2))": params.get("k", 0.0),
                    "theta (deg)": params.get("theta0", 0.0),
                    "model": model_name,
                    "ref": params.get("ref", ""),  # Référence explicite
                }
            )

        # Improper
        for key, params in model_data.get("improper", {}).items():
            parts = key.split("-")
            data["improper"].append(
                {
                    "type 1": parts[0] if len(parts) > 0 else "",
                    "type 2": parts[1] if len(parts) > 1 else "",
                    "type 3": parts[2] if len(parts) > 2 else "",
                    "type 4": parts[3] if len(parts) > 3 else "",
                    "k (kcal/(mol,rad2))": params.get("k", 0.0),
                    "theta (deg)": params.get("theta0", 0.0),
                    "model": model_name,
                    "ref": params.get("ref", ""),  # Référence explicite
                }
            )

    # 3. Écrire le JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Generated: {OUTPUT_PATH}")
    print(f"   Models: {len(data['models'])}")
    print(f"   Atom types: {len(data['list'])}")
    print(f"   LJ pairs: {len(data['lj_12-6'])}")
    print(f"   Buckingham pairs: {len(data['buckingham'])}")
    print(f"   Bond pairs: {len(data['bond'])}")
    print(f"   Angle triples: {len(data['angle'])}")
    print(f"   Improper quads: {len(data['improper'])}")


if __name__ == "__main__":
    toml_to_json()
