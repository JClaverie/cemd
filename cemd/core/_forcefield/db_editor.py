import tomllib
from pathlib import Path

import pandas as pd
import streamlit as st
import tomli_w

FF_DIR = Path(__file__).parent / "db"

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def load_ff(name: str) -> dict:
    with open(FF_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def save_ff(name: str, data: dict) -> None:
    with open(FF_DIR / f"{name}.toml", "wb") as f:
        tomli_w.dump(data, f)


def get_all_ff() -> list[str]:
    """Returns all .toml files in the forcefields folder, except _metadata.toml."""
    return sorted([p.stem for p in FF_DIR.glob("*.toml") if p.name != "_metadata.toml"])


def dict_to_df(data: dict) -> pd.DataFrame:
    """Converts a dictionary {key: {params}} to a DataFrame."""
    if not data:
        return pd.DataFrame()
    return pd.DataFrame([{"key": k, **v} for k, v in data.items()])


def df_to_dict(df: pd.DataFrame) -> dict:
    """Converts a DataFrame with column 'key' to dictionary {key: {params}}."""
    result = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("key")) and row.get("key"):
            key = row["key"]
            result[key] = {k: v for k, v in row.items() if k != "key" and pd.notna(v)}
    return result


# ----------------------------------------------------------------
# App
# ----------------------------------------------------------------

st.set_page_config(page_title="CEMD Forcefield Editor", layout="wide")
st.title("CEMD Forcefield Database")

# Sidebar — file selection
with st.sidebar:
    st.header("Forcefields")
    ff_names = get_all_ff()
    selected = st.radio("Select", ff_names, key="ff_selector")

    st.divider()

    # ---Create a new forcefield ---
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
                    "dihedral": {"charmm": {}, "harmonic": {}},
                }
                save_ff(new_name, template)
                st.success(f"Created {new_name}.toml")
                st.rerun()

    # ---Duplicate an existing forcefield ---
    with st.expander("📋 Duplicate"):
        dup_name = st.text_input("New name", key="dup_name")
        if st.button("Duplicate", key="dup_ff") and dup_name:
            data_dup = load_ff(selected)
            save_ff(dup_name, data_dup)
            st.success(f"Duplicated to {dup_name}.toml")
            st.rerun()

    # ---DELETE ---
    with st.expander("🗑️ Delete", expanded=False):
        st.warning(f"Delete '{selected}' permanently?")
        if st.button("Confirm delete", type="primary", key="del_ff"):
            (FF_DIR / f"{selected}.toml").unlink()
            st.rerun()

# Loading the selected forcefield
data = load_ff(selected)

st.header(f"{data['model']['name']}")
st.caption(f"Reference: {data['model'].get('reference', 'N/A')}")

# Tabs by section
(
    tab_atoms,
    tab_lj,
    tab_buck,
    tab_bonds,
    tab_angles,
    tab_bondbond,
    tab_bondangle,
    tab_impropers,
    tab_dihedrals,
) = st.tabs(
    [
        "Atoms",
        "LJ 12-6",
        "Buckingham",
        "Bonds",
        "Angles",
        "BondBond",
        "BondAngle",
        "Impropers",
        "Dihedrals",
    ]
)

# ================================================================
# ATOM TYPES
# ================================================================
with tab_atoms:
    st.subheader("Atom Types")

    atoms_data = data.get("atom", {})
    df_atoms = dict_to_df(atoms_data)

    if df_atoms.empty:
        df_atoms = pd.DataFrame(
            columns=["key", "element", "charge", "environment", "ref"]
        )

    edited_atoms = st.data_editor(
        df_atoms,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_atoms",
        column_config={
            "key": st.column_config.TextColumn("🔑 Key", required=True, width="medium"),
            "element": st.column_config.TextColumn(
                "⚛️ Element", required=True, width="small"
            ),
            "charge": st.column_config.NumberColumn(
                "🔋 Charge", required=True, width="small"
            ),
            "environment": st.column_config.TextColumn("🌍 Environment", width="large"),
            "ref": st.column_config.TextColumn("📚 Reference", width="medium"),
        },
    )

    if st.button("💾 Save Atoms", key="save_atoms", type="primary"):
        data["atom"] = df_to_dict(edited_atoms)
        save_ff(selected, data)
        st.success(f"✅ Saved {len(data['atom'])} atom types!")
        st.rerun()

# ================================================================
# LJ 12-6
# ================================================================
with tab_lj:
    st.subheader("Lennard-Jones 12-6 Parameters")

    lj_data = data.get("lj", {})
    df_lj = dict_to_df(lj_data)

    if df_lj.empty:
        df_lj = pd.DataFrame(columns=["key", "epsilon", "sigma", "ref"])

    edited_lj = st.data_editor(
        df_lj,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_lj",
        column_config={
            "key": st.column_config.TextColumn(
                "🔑 Pair", required=True, width="medium"
            ),
            "epsilon": st.column_config.NumberColumn(
                "ε (kcal/mol)", required=True, width="medium"
            ),
            "sigma": st.column_config.NumberColumn(
                "σ (Å)", required=True, width="medium"
            ),
            "ref": st.column_config.TextColumn("📚 Reference", width="medium"),
        },
    )

    if st.button("💾 Save LJ", key="save_lj", type="primary"):
        data["lj"] = df_to_dict(edited_lj)
        save_ff(selected, data)
        st.success(f"✅ Saved {len(data['lj'])} LJ parameters!")
        st.rerun()

# ================================================================
# BUCKINGHAM
# ================================================================
with tab_buck:
    st.subheader("Buckingham Parameters")

    buck_data = data.get("buckingham", {})
    df_buck = dict_to_df(buck_data)

    if df_buck.empty:
        df_buck = pd.DataFrame(columns=["key", "A", "rho", "C", "ref"])

    edited_buck = st.data_editor(
        df_buck,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_buck",
        column_config={
            "key": st.column_config.TextColumn(
                "🔑 Pair", required=True, width="medium"
            ),
            "A": st.column_config.NumberColumn(
                "A (kcal/mol)", required=True, width="medium"
            ),
            "rho": st.column_config.NumberColumn(
                "ρ (Å)", required=True, width="medium"
            ),
            "C": st.column_config.NumberColumn(
                "C (kcal/mol·Å⁶)", required=True, width="medium"
            ),
            "ref": st.column_config.TextColumn("📚 Reference", width="medium"),
        },
    )

    if st.button("💾 Save Buckingham", key="save_buck", type="primary"):
        data["buckingham"] = df_to_dict(edited_buck)
        save_ff(selected, data)
        st.success(f"✅ Saved {len(data['buckingham'])} Buckingham parameters!")
        st.rerun()

# ================================================================
# BONDS (harmonic + class2)
# ================================================================
with tab_bonds:
    st.subheader("Bond Parameters")

    bonds_data = data.get("bond", {})
    bond_styles = list(bonds_data.keys())

    if not bond_styles:
        st.info("No bond styles defined. Add 'harmonic' or 'class2'.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add harmonic", key="add_bond_harmonic"):
                data["bond"]["harmonic"] = {}
                save_ff(selected, data)
                st.rerun()
        with col2:
            if st.button("Add class2", key="add_bond_class2"):
                data["bond"]["class2"] = {}
                save_ff(selected, data)
                st.rerun()
    else:
        for style in bond_styles:
            with st.expander(f"🔗 {style}", expanded=False):
                params = bonds_data.get(style, {})
                df_bond = dict_to_df(params)

                if df_bond.empty:
                    # Specific columns according to style
                    if style == "harmonic":
                        df_bond = pd.DataFrame(columns=["key", "k", "r0", "ref"])
                    else:  # Class2
                        df_bond = pd.DataFrame(
                            columns=["key", "r0", "k2", "k3", "k4", "ref"]
                        )

                # Config columns according to style
                col_config = {
                    "key": st.column_config.TextColumn(
                        "🔑 Bond", required=True, width="medium"
                    ),
                }
                if style == "harmonic":
                    col_config["k"] = st.column_config.NumberColumn(
                        "k (kcal/mol·Å²)", required=True, width="medium"
                    )
                    col_config["r0"] = st.column_config.NumberColumn(
                        "r0 (Å)", required=True, width="medium"
                    )
                else:  # Class2
                    col_config["r0"] = st.column_config.NumberColumn(
                        "r0 (Å)", required=True, width="medium"
                    )
                    col_config["k2"] = st.column_config.NumberColumn(
                        "k2", required=True, width="medium"
                    )
                    col_config["k3"] = st.column_config.NumberColumn(
                        "k3", width="medium"
                    )
                    col_config["k4"] = st.column_config.NumberColumn(
                        "k4", width="medium"
                    )
                col_config["ref"] = st.column_config.TextColumn(
                    "📚 Reference", width="medium"
                )

                edited_bond = st.data_editor(
                    df_bond,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"editor_bonds_{style}",
                    column_config=col_config,
                )

                if st.button(
                    f"💾 Save {style}", key=f"save_bonds_{style}", type="primary"
                ):
                    data["bond"][style] = df_to_dict(edited_bond)
                    save_ff(selected, data)
                    st.success(f"✅ Saved {len(data['bond'][style])} bonds!")
                    st.rerun()

# ================================================================
# ANGLES (harmonic + class2)
# ================================================================
with tab_angles:
    st.subheader("Angle Parameters")

    angles_data = data.get("angle", {})
    angle_styles = list(angles_data.keys())

    if not angle_styles:
        st.info("No angle styles defined. Add 'harmonic' or 'class2'.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add harmonic", key="add_angle_harmonic"):
                data["angle"]["harmonic"] = {}
                save_ff(selected, data)
                st.rerun()
        with col2:
            if st.button("Add class2", key="add_angle_class2"):
                data["angle"]["class2"] = {}
                save_ff(selected, data)
                st.rerun()
    else:
        for style in angle_styles:
            with st.expander(f"📐 {style}", expanded=False):
                params = angles_data.get(style, {})
                df_angle = dict_to_df(params)

                if df_angle.empty:
                    if style == "harmonic":
                        df_angle = pd.DataFrame(columns=["key", "k", "theta0", "ref"])
                    else:  # Class2
                        df_angle = pd.DataFrame(
                            columns=["key", "theta0", "k2", "k3", "k4", "ref"]
                        )

                col_config = {
                    "key": st.column_config.TextColumn(
                        "🔑 Angle", required=True, width="medium"
                    ),
                }
                if style == "harmonic":
                    col_config["k"] = st.column_config.NumberColumn(
                        "k (kcal/mol·rad²)", required=True, width="medium"
                    )
                    col_config["theta0"] = st.column_config.NumberColumn(
                        "θ₀ (°)", required=True, width="medium"
                    )
                else:  # Class2
                    col_config["theta0"] = st.column_config.NumberColumn(
                        "θ₀ (°)", required=True, width="medium"
                    )
                    col_config["k2"] = st.column_config.NumberColumn(
                        "k2", required=True, width="medium"
                    )
                    col_config["k3"] = st.column_config.NumberColumn(
                        "k3", width="medium"
                    )
                    col_config["k4"] = st.column_config.NumberColumn(
                        "k4", width="medium"
                    )
                col_config["ref"] = st.column_config.TextColumn(
                    "📚 Reference", width="medium"
                )

                edited_angle = st.data_editor(
                    df_angle,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"editor_angles_{style}",
                    column_config=col_config,
                )

                if st.button(
                    f"💾 Save {style}", key=f"save_angles_{style}", type="primary"
                ):
                    data["angle"][style] = df_to_dict(edited_angle)
                    save_ff(selected, data)
                    st.success(f"✅ Saved {len(data['angle'][style])} angles!")
                    st.rerun()

# =================================================================
# BONDBOND (class2)
# =================================================================
with tab_bondbond:
    st.subheader("BondBond Parameters (class2)")
    st.caption("Term E_bb = M * (r_ij - r1) * (r_jk - r2)")

    bondbond_data = data.get("bondbond", {}).get("class2", {})
    df_bondbond = dict_to_df(bondbond_data)

    if df_bondbond.empty:
        df_bondbond = pd.DataFrame(columns=["key", "m", "r1", "r2", "ref"])

    edited_bondbond = st.data_editor(
        df_bondbond,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_bondbond",
        column_config={
            "key": st.column_config.TextColumn(
                "🔑 Angle", required=True, width="medium"
            ),
            "m": st.column_config.NumberColumn(
                "M (kcal/mol·Å²)", required=True, width="medium"
            ),
            "r1": st.column_config.NumberColumn(
                "r1 (Å)", required=True, width="medium"
            ),
            "r2": st.column_config.NumberColumn(
                "r2 (Å)", required=True, width="medium"
            ),
            "ref": st.column_config.TextColumn("📚 Reference", width="medium"),
        },
    )

    if st.button("💾 Save BondBond", key="save_bondbond", type="primary"):
        if "bondbond" not in data:
            data["bondbond"] = {}
        data["bondbond"]["class2"] = df_to_dict(edited_bondbond)
        save_ff(selected, data)
        st.success(f"✅ Saved {len(data['bondbond']['class2'])} bondbond parameters!")
        st.rerun()

# ================================================================
# BOND ANGLE (class 2)
# ================================================================
with tab_bondangle:
    st.subheader("BondAngle Parameters (class2)")
    st.caption("Term E_ba = N1*(r_ij - r1)*(θ-θ0) + N2*(r_jk - r2)*(θ-θ0)")

    bondangle_data = data.get("bondangle", {}).get("class2", {})
    df_bondangle = dict_to_df(bondangle_data)

    if df_bondangle.empty:
        df_bondangle = pd.DataFrame(columns=["key", "n1", "n2", "r1", "r2", "ref"])

    edited_bondangle = st.data_editor(
        df_bondangle,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_bondangle",
        column_config={
            "key": st.column_config.TextColumn(
                "🔑 Angle", required=True, width="medium"
            ),
            "n1": st.column_config.NumberColumn(
                "N1 (kcal/mol·Å)", required=True, width="medium"
            ),
            "n2": st.column_config.NumberColumn(
                "N2 (kcal/mol·Å)", required=True, width="medium"
            ),
            "r1": st.column_config.NumberColumn(
                "r1 (Å)", required=True, width="medium"
            ),
            "r2": st.column_config.NumberColumn(
                "r2 (Å)", required=True, width="medium"
            ),
            "ref": st.column_config.TextColumn("📚 Reference", width="medium"),
        },
    )

    if st.button("💾 Save BondAngle", key="save_bondangle", type="primary"):
        if "bondangle" not in data:
            data["bondangle"] = {}
        data["bondangle"]["class2"] = df_to_dict(edited_bondangle)
        save_ff(selected, data)
        st.success(f"✅ Saved {len(data['bondangle']['class2'])} bondangle parameters!")
        st.rerun()

# ================================================================
# IMPROPERS (distance + harmonic)
# ================================================================
with tab_impropers:
    st.subheader("Improper Parameters")

    impropers_data = data.get("improper", {})
    improper_styles = list(impropers_data.keys())

    if not improper_styles:
        st.info("No improper styles defined.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add distance", key="add_improper_distance"):
                data["improper"]["distance"] = {}
                save_ff(selected, data)
                st.rerun()
        with col2:
            if st.button("Add harmonic", key="add_improper_harmonic"):
                data["improper"]["harmonic"] = {}
                save_ff(selected, data)
                st.rerun()
    else:
        for style in improper_styles:
            with st.expander(f"⚠️ {style}", expanded=False):
                params = impropers_data.get(style, {})
                df_improper = dict_to_df(params)

                if df_improper.empty:
                    if style == "distance":
                        df_improper = pd.DataFrame(columns=["key", "k2", "k4", "ref"])
                    else:  # Harmonic
                        df_improper = pd.DataFrame(columns=["key", "k", "chi0", "ref"])

                col_config = {
                    "key": st.column_config.TextColumn(
                        "🔑 Improper", required=True, width="medium"
                    ),
                }
                if style == "distance":
                    col_config["k2"] = st.column_config.NumberColumn(
                        "k2", required=True, width="medium"
                    )
                    col_config["k4"] = st.column_config.NumberColumn(
                        "k4", width="medium"
                    )
                else:
                    col_config["k"] = st.column_config.NumberColumn(
                        "k (kcal/mol·rad²)", required=True, width="medium"
                    )
                    col_config["chi0"] = st.column_config.NumberColumn(
                        "χ₀ (°)", width="medium"
                    )
                col_config["ref"] = st.column_config.TextColumn(
                    "📚 Reference", width="medium"
                )

                edited_improper = st.data_editor(
                    df_improper,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"editor_impropers_{style}",
                    column_config=col_config,
                )

                if st.button(
                    f"💾 Save {style}", key=f"save_impropers_{style}", type="primary"
                ):
                    data["improper"][style] = df_to_dict(edited_improper)
                    save_ff(selected, data)
                    st.success(f"✅ Saved {len(data['improper'][style])} impropers!")
                    st.rerun()

# ================================================================
# DIHEDRALS (charmm + harmonic)
# ================================================================
with tab_dihedrals:
    st.subheader("Dihedral Parameters")

    dihedrals_data = data.get("dihedral", {})
    dihedral_styles = list(dihedrals_data.keys())

    if not dihedral_styles:
        st.info("No dihedral styles defined.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add charmm", key="add_dihedral_charmm"):
                data["dihedral"]["charmm"] = {}
                save_ff(selected, data)
                st.rerun()
        with col2:
            if st.button("Add harmonic", key="add_dihedral_harmonic"):
                data["dihedral"]["harmonic"] = {}
                save_ff(selected, data)
                st.rerun()
    else:
        for style in dihedral_styles:
            with st.expander(f"🌀 {style}", expanded=False):
                params = dihedrals_data.get(style, {})
                df_dihedral = dict_to_df(params)

                if df_dihedral.empty:
                    if style == "harmonic":
                        df_dihedral = pd.DataFrame(
                            columns=["key", "k", "d", "n", "ref"]
                        )
                    else:  # Charm
                        df_dihedral = pd.DataFrame(
                            columns=["key", "k", "n", "d", "phi0", "ref"]
                        )

                col_config = {
                    "key": st.column_config.TextColumn(
                        "🔑 Dihedral", required=True, width="medium"
                    ),
                }
                if style == "harmonic":
                    col_config["k"] = st.column_config.NumberColumn(
                        "k (kcal/mol)", required=True, width="medium"
                    )
                    col_config["d"] = st.column_config.NumberColumn(
                        "d (+1/-1)", required=True, width="small"
                    )
                    col_config["n"] = st.column_config.NumberColumn(
                        "n", required=True, width="small"
                    )
                else:  # Charm
                    col_config["k"] = st.column_config.NumberColumn(
                        "k (kcal/mol)", required=True, width="medium"
                    )
                    col_config["n"] = st.column_config.NumberColumn(
                        "n", required=True, width="small"
                    )
                    col_config["d"] = st.column_config.NumberColumn(
                        "d", required=True, width="small"
                    )
                    col_config["phi0"] = st.column_config.NumberColumn(
                        "φ₀ (°)", required=True, width="medium"
                    )
                col_config["ref"] = st.column_config.TextColumn(
                    "📚 Reference", width="medium"
                )

                edited_dihedral = st.data_editor(
                    df_dihedral,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"editor_dihedrals_{style}",
                    column_config=col_config,
                )

                if st.button(
                    f"💾 Save {style}", key=f"save_dihedrals_{style}", type="primary"
                ):
                    data["dihedral"][style] = df_to_dict(edited_dihedral)
                    save_ff(selected, data)
                    st.success(f"✅ Saved {len(data['dihedral'][style])} dihedrals!")
                    st.rerun()

# ================================================================
# COMPARISON OF FORCEFIELDS
# ================================================================
st.divider()
with st.expander("🔍 Compare atom types across forcefields"):
    all_atoms = {}
    for name in ff_names:
        ff = load_ff(name)
        for atype, params in ff.get("atom", {}).items():
            if atype not in all_atoms:
                all_atoms[atype] = {}
            all_atoms[atype][name] = params.get("charge", "—")

    if all_atoms:
        df_compare = pd.DataFrame(all_atoms).T
        st.dataframe(df_compare, use_container_width=True)
    else:
        st.info("No atom types to compare.")
