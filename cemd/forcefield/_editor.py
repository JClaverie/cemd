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

import dataclasses
import tomllib

import pandas as pd
import streamlit as st
import tomli_w
from models import (
    AtomType,
    BuckinghamParams,
    CHARMMDihedralParams,
    Class2AngleAngleParams,
    Class2AngleAngleTorsionParams,
    Class2AngleParams,
    Class2BondAngleParams,
    Class2BondBondParams,
    Class2BondParams,
    Class2DihedralParams,
    Class2ImproperParams,
    CVFFImproperParams,
    DistanceImproperParams,
    HarmonicAngleParams,
    HarmonicBondParams,
    HarmonicDihedralParams,
    HarmonicImproperParams,
    LJParams,
    MorseBondParams,
)

from cemd.forcefield._config import get_user_config_path, reset_user_forcefields

FF_DIR = get_user_config_path()


def load_ff(name: str) -> dict:
    with open(FF_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def save_ff(name: str, data: dict) -> None:
    with open(FF_DIR / f"{name}.toml", "wb") as f:
        tomli_w.dump(data, f)


def get_all_ff() -> list[str]:
    return sorted([p.stem for p in FF_DIR.glob("*.toml") if p.name != "_metadata.toml"])


def dict_to_df(data: dict, default_cols: list[str] = None) -> pd.DataFrame:
    """Converts a dictionary {key: {params}} to a DataFrame with fallback columns."""
    if not data:
        if default_cols:
            return pd.DataFrame(columns=default_cols)
        return pd.DataFrame()
    return pd.DataFrame([{"key": k, **v} for k, v in data.items()])


def df_to_dict(df: pd.DataFrame) -> dict:
    result = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("key")) and row.get("key"):
            key = str(row["key"])
            result[key] = {k: v for k, v in row.items() if k != "key" and pd.notna(v)}
    return result


st.set_page_config(page_title="CEMD Forcefield Editor", layout="wide")
st.title("CEMD Forcefield Database Editor")

# Sidebar — file selection and management
with st.sidebar:
    st.header("Forcefields")
    ff_names = get_all_ff()
    if not ff_names:
        st.warning("No TOML files found in the db directory.")
        st.stop()

    selected = st.radio("Select", ff_names, key="ff_selector")

    st.divider()

    # --- Create a new forcefield ---
    with st.expander("➕ New forcefield"):
        new_name = st.text_input("Name (e.g. 'reaxff')", key="new_name")
        new_display = st.text_input("Display name (e.g. 'ReaxFF')", key="new_display")
        new_ref = st.text_input("Reference DOI", key="new_ref")
        new_desc = st.text_area("Description", key="new_desc")

        if st.button("Create", type="primary", key="create_ff") and new_name:
            if (FF_DIR / f"{new_name}.toml").exists():
                st.error(f"'{new_name}' already exists.")
            else:
                template = {
                    "model": {
                        "name": new_display or new_name,
                        "reference": new_ref,
                        "description": new_desc,
                    },
                    "atom": {},
                    "lj": {},
                    "buckingham": {},
                    "bond": {"harmonic": {}, "class2": {}},
                    "angle": {"harmonic": {}, "class2": {}},
                    "bondbond": {"class2": {}},
                    "bondangle": {"class2": {}},
                    "improper": {"distance": {}, "harmonic": {}},
                    "dihedral": {},
                }
                save_ff(new_name, template)
                st.success(f"Created: {new_name}.toml")
                st.rerun()

    # --- Duplicate ---
    with st.expander("📋 Duplicate"):
        dup_name = st.text_input("New name", key="dup_name")
        if st.button("Duplicate", key="dup_ff") and dup_name:
            if (FF_DIR / f"{dup_name}.toml").exists():
                st.error(f"'{dup_name}' already exists.")
            else:
                data_dup = load_ff(selected)
                save_ff(dup_name, data_dup)
                st.success(f"Duplicated to {dup_name}.toml")
                st.rerun()

    # --- Delete ---
    with st.expander("🗑️ Delete", expanded=False):
        st.warning(f"Permanently delete '{selected}'?")
        if st.button("Confirm deletion", type="primary", key="del_ff"):
            (FF_DIR / f"{selected}.toml").unlink()
            st.rerun()

    with st.expander("🔄 Reset to defaults", expanded=False):
        st.warning(
            "This will overwrite all your custom changes with the original default forcefields."
        )
        if st.button("Confirm Reset", type="primary", key="reset_ff"):
            reset_user_forcefields()
            st.success("Forcefields successfully reset to defaults!")
            st.rerun()

data = load_ff(selected)

st.header(f"{data.get('model', {}).get('name', selected)}")
st.caption(f"Reference: {data.get('model', {}).get('reference', 'N/A')}")
st.write(f"Description: {data.get('model', {}).get('description', '')}")

st.divider()


def generate_category_config(
    dc_class, main_key: str, sub_key: str | None = None
) -> dict:
    """
    Extrait automatiquement les colonnes et les types d'une dataclass
    pour configurer le tableau Streamlit.
    """
    fields = dataclasses.fields(dc_class)

    # The 'key' column always comes first
    cols = ["key"] + [f.name for f in fields]

    config = {"key": st.column_config.TextColumn("Key", help="Unique identifier")}

    for f in fields:
        type_str = str(f.type)
        if "float" in type_str:
            config[f.name] = st.column_config.NumberColumn(
                f.name.capitalize(), format="%.4f", step=0.01
            )
        elif "int" in type_str:
            config[f.name] = st.column_config.NumberColumn(
                f.name.capitalize(), format="%d", step=1
            )
        else:
            config[f.name] = st.column_config.TextColumn(f.name.capitalize())

    return {"key": (main_key, sub_key), "cols": cols, "config": config}


# Category dictionary generated automatically from the dataclasses!
categories = {
    "Atom Types": generate_category_config(AtomType, "atom"),
    "Lennard-Jones (LJ)": generate_category_config(LJParams, "lj"),
    "Buckingham": generate_category_config(BuckinghamParams, "buckingham"),
    "Bonds - Harmonic": generate_category_config(
        HarmonicBondParams, "bond", "harmonic"
    ),
    "Bonds - Morse": generate_category_config(MorseBondParams, "bond", "morse"),
    "Bonds - Class2": generate_category_config(Class2BondParams, "bond", "class2"),
    "Angles - Harmonic": generate_category_config(
        HarmonicAngleParams, "angle", "harmonic"
    ),
    "Angles - Class2": generate_category_config(Class2AngleParams, "angle", "class2"),
    "Bond-Bond - Class2": generate_category_config(
        Class2BondBondParams, "bondbond", "class2"
    ),
    "Bond-Angle - Class2": generate_category_config(
        Class2BondAngleParams, "bondangle", "class2"
    ),
    "Dihedrals - Harmonic": generate_category_config(
        HarmonicDihedralParams, "dihedral", "harmonic"
    ),
    "Dihedrals - Class2": generate_category_config(
        Class2DihedralParams, "dihedral", "class2"
    ),
    "Dihedrals - CHARMM": generate_category_config(
        CHARMMDihedralParams, "dihedral", "charmm"
    ),
    "Dihedrals - Angle-Angle-Torsion": generate_category_config(
        Class2AngleAngleTorsionParams, "dihedral", "angle_angle_torsion"
    ),
    "Impropers - Harmonic": generate_category_config(
        HarmonicImproperParams, "improper", "harmonic"
    ),
    "Impropers - Class2": generate_category_config(
        Class2ImproperParams, "improper", "class2"
    ),
    "Impropers - CVFF": generate_category_config(
        CVFFImproperParams, "improper", "cvff"
    ),
    "Impropers - Angle-Angle": generate_category_config(
        Class2AngleAngleParams, "improper", "angle_angle"
    ),
    "Impropers - Distance": generate_category_config(
        DistanceImproperParams, "improper", "distance"
    ),
}

selected_category = st.selectbox(
    "Select parameter category to edit:", list(categories.keys())
)

cat_info = categories[selected_category]
main_key, sub_key = cat_info["key"]
default_cols = cat_info["cols"]

if sub_key:
    section_data = data.get(main_key, {}).get(sub_key, {})
else:
    section_data = data.get(main_key, {})

st.subheader(selected_category)

df_section = dict_to_df(section_data, default_cols=default_cols)
edited_df = st.data_editor(
    df_section,
    num_rows="dynamic",
    use_container_width=True,
    key=f"editor_{main_key}_{sub_key}",
)

if st.button(f"Save {selected_category}"):
    if main_key not in data:
        data[main_key] = {}

    converted_dict = df_to_dict(edited_df)

    if sub_key:
        if main_key not in data or not isinstance(data[main_key], dict):
            data[main_key] = {}
        data[main_key][sub_key] = converted_dict
    else:
        data[main_key] = converted_dict

    save_ff(selected, data)
    st.success(f"Changes for '{selected_category}' saved successfully!")
