# scripts/toml_to_json.py
"""
Export all ForceFieldDatabase objects into the structured JSON format.
"""

import json
from pathlib import Path

from cemd.core._forcefield.forcefield_database import ForceFieldDatabase

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent
OUTPUT_PATH = DOCS_DIR / "_static" / "ff_data.json"


def db_to_json() -> None:
    """Export ForceFieldDatabase('all') to JSON matching the target schema."""

    # 1. Instanciation de la base de données globale
    db = ForceFieldDatabase("all")

    # Initialisation du dictionnaire avec la structure JSON attendue
    data = {
        "metadata": {"models": {}},
        "models": {},
        "list": [],
        "lj_12-6": [],
        "buckingham": [],
        "bond": [],
        "angle": [],
        "improper": [],
        "dihedral": [],
    }

    # 2. Remplissage des métadonnées et modèles
    for model_key, model_obj in db.models.items():
        model_info = {
            "name": getattr(model_obj, "name", model_key),
            "reference": getattr(model_obj, "ref", ""),
            "year": getattr(model_obj, "year", 0),
            "authors": getattr(model_obj, "authors", []),
            "description": getattr(model_obj, "description", ""),
            "tags": getattr(model_obj, "tags", []),
        }
        # Les sections metadata.models et models partagent le même format
        data["metadata"]["models"][model_key] = model_info
        data["models"][model_key] = model_info

    # 3. Remplissage de la liste des atomes ("list")
    for full_key, atom_obj in db.atom.items():
        short_type = full_key.split(".")[-1] if "." in full_key else full_key

        data["list"].append(
            {
                "type": short_type,
                "full_type": full_key,
                "element": atom_obj.element,
                "charge": atom_obj.charge,
                "model": atom_obj.model or "",
                "environment": getattr(atom_obj, "environment", ""),
                "ref": atom_obj.ref,
                "mass": getattr(atom_obj, "mass", None),
            }
        )

    # 4. Lennard-Jones (lj_12-6)
    for full_key, lj_obj in db.lj.items():
        pair_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = pair_str.split("-")

        data["lj_12-6"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "epsilon (kcal/mol)": getattr(lj_obj, "epsilon", 0.0),
                "sigma (A)": getattr(lj_obj, "sigma", 0.0),
                "model": lj_obj.model or "",
                "ref": lj_obj.ref,
            }
        )

    # 5. Buckingham
    for full_key, buck_obj in db.buckingham.items():
        pair_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = pair_str.split("-")

        data["buckingham"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "A (kcal/mol)": getattr(buck_obj, "a", 0.0),
                "rho (A)": getattr(buck_obj, "rho", 0.0),
                "C (kcal/mol.A6)": getattr(buck_obj, "c", 0.0),
                "model": buck_obj.model or "",
                "ref": buck_obj.ref,
            }
        )

    # 6. Liaisons (bond)
    for full_key, bond_obj in db.bond.items():
        pair_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = pair_str.split("-")

        k_val = getattr(bond_obj, "k", getattr(bond_obj, "k2", 0.0))
        r_val = getattr(bond_obj, "r0", 0.0)

        data["bond"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "k (kcal/(mol.A2)": k_val,
                "r (A)": r_val,
                "model": bond_obj.model or "",
                "ref": bond_obj.ref,
            }
        )

    # 7. Angles (angle)
    for full_key, angle_obj in db.angle.items():
        triple_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = triple_str.split("-")

        k_val = getattr(angle_obj, "k", getattr(angle_obj, "k2", 0.0))
        theta_val = getattr(angle_obj, "theta0", 0.0)

        data["angle"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "type 3": parts[2] if len(parts) > 2 else "",
                "k (kcal/(mol.rad2))": k_val,
                "theta (deg)": theta_val,
                "model": angle_obj.model or "",
                "ref": angle_obj.ref,
            }
        )

    # 8. Impropres (improper)
    for full_key, imp_obj in db.improper.items():
        quad_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = quad_str.split("-")

        k_val = getattr(imp_obj, "k", getattr(imp_obj, "k2", 0.0))
        theta_val = getattr(imp_obj, "chi0", 0.0)

        data["improper"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "type 3": parts[2] if len(parts) > 2 else "",
                "type 4": parts[3] if len(parts) > 3 else "",
                "k (kcal/(mol,rad2))": k_val,
                "theta (deg)": theta_val,
                "model": getattr(imp_obj, "model", "") or "",
                "ref": getattr(imp_obj, "ref", ""),
            }
        )

    # 9. Dièdres (dihedral)
    for full_key, dih_obj in db.dihedral.items():
        quad_str = full_key.split(".")[-1] if "." in full_key else full_key
        parts = quad_str.split("-")

        data["dihedral"].append(
            {
                "type 1": parts[0] if len(parts) > 0 else "",
                "type 2": parts[1] if len(parts) > 1 else "",
                "type 3": parts[2] if len(parts) > 2 else "",
                "type 4": parts[3] if len(parts) > 3 else "",
                "model": getattr(dih_obj, "model", "") or "",
                "ref": getattr(dih_obj, "ref", ""),
            }
        )

    # 10. Écriture du fichier JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exportation réussie dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    db_to_json()
