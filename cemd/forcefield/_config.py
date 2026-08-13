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

import shutil
import sys
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================


def get_current_dir() -> Path:
    """Returns the directory of this file."""
    return Path(__file__).resolve().parent


def get_default_config_path() -> Path:
    """Returns the path of the default config file (same folder)."""
    return get_current_dir() / "db"


def get_user_config_path() -> Path:
    """Returns the path of the user config file."""
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".config"

    return base / "cemd" / "forcefields"


# ============================================================================
# Force fields (private in package, public in .config)
# ============================================================================


def get_ff_db_path() -> Path:
    """
    Get the force field database path (priority: user > default).

    Returns
    -------
    Path
        Path to the force field directory containing TOML files.
    """
    user_path = get_user_config_path()
    if user_path.exists():
        try:
            has_files = any(
                f.endswith(".toml") for f in user_path.iterdir() if f.is_file()
            )
            if has_files:
                return user_path
        except OSError:
            pass
    return get_default_config_path()


# ============================================================================
# Initialization
# ============================================================================


def init_user_forcefields_database(force: bool = False) -> None:
    """Copy all default force field files to user directory."""
    user_dir = get_user_config_path()
    default_dir = get_default_config_path()

    # Vérifier si le dossier source existe
    if not default_dir.exists():
        print(f"Default force field directory not found: {default_dir}")
        return

    # Lister les fichiers TOML
    toml_files = [
        f for f in default_dir.iterdir() if f.is_file() and f.suffix == ".toml"
    ]
    if not toml_files:
        print(f"No TOML files found in default directory: {default_dir}")
        return

    # Créer le dossier utilisateur si nécessaire
    user_dir.mkdir(parents=True, exist_ok=True)

    # Copier les fichiers
    copied = 0
    for src in toml_files:
        dst = user_dir / src.name

        # Copier si force=True ou si le fichier n'existe pas
        if force or not dst.exists():
            shutil.copy2(src, dst)
            copied += 1


def reset_user_forcefields() -> None:
    """
    Reset all user force field files by overwriting them with default ones.
    """
    init_user_forcefields_database(force=True)
    print("Force fields have been reset to default values.")


init_user_forcefields_database()
