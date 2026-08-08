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
import shutil
import sys

# ============================================================================
# Paths
# ============================================================================


def get_current_dir() -> str:
    """Returns the directory of this file."""
    return os.path.dirname(os.path.abspath(__file__))


def get_default_config_path() -> str:
    """Returns the path of the default config file (same folder)."""
    return os.path.join(get_current_dir(), "forcefields")


def get_user_config_path() -> str:
    """Returns the path of the user config file."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.path.expanduser("~/.config")

    return os.path.join(base, "cemd", "forcefields")


# ============================================================================
# Force fields (private in package, public in .config)
# ============================================================================


def get_ff_db_path() -> str:
    """
    Get the force field database path (priority: user > default).

    Returns
    -------
    str
        Path to the force field directory containing TOML files.
    """
    user_path = get_user_config_path()
    if os.path.exists(user_path):
        try:
            has_files = any(f.endswith(".toml") for f in os.listdir(user_path))
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
    if not os.path.exists(default_dir):
        print(f"Default force field directory not found: {default_dir}")
        return

    # Lister les fichiers TOML
    toml_files = [f for f in os.listdir(default_dir) if f.endswith(".toml")]
    if not toml_files:
        print(f"No TOML files found in default directory: {default_dir}")
        return

    # Créer le dossier utilisateur si nécessaire
    os.makedirs(user_dir, exist_ok=True)

    # Copier les fichiers
    copied = 0
    for filename in toml_files:
        src = os.path.join(default_dir, filename)
        dst = os.path.join(user_dir, filename)

        # Copier si force=True ou si le fichier n'existe pas
        if force or not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1


init_user_forcefields_database()
