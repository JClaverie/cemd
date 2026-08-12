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
import tomllib
from typing import Any


def get_current_dir() -> str:
    """Returns the directory of this file."""
    return os.path.dirname(os.path.abspath(__file__))


def get_default_config_path() -> str:
    """Returns the path of the default config file (same folder)."""
    return os.path.join(get_current_dir(), "default_vmd_config.toml")


def get_user_config_path() -> str:
    """Returns the path of the user config file."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.path.expanduser("~/.config")

    return os.path.join(base, "cemd", "vmd_config.toml")


def load_toml(path: str) -> dict:
    """Load a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config() -> dict[str, Any]:
    """
    Load the configuration:
    1. The default config from the package
    2. The user config if it exists (overload)
    """
    default_path = get_default_config_path()

    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default config not found: {default_path}")

    config = load_toml(default_path)

    # Overload with user config if it exists
    user_path = get_user_config_path()
    if os.path.exists(user_path):
        try:
            user_config = load_toml(user_path)
            config = deep_merge(config, user_config)
        except Exception as e:
            print(f"Error loading user config: {e}")

    return config


def deep_merge(base, override):
    """Recursive merge of two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def init_user_config() -> None:
    """
    Copy the default file to the user folder
    if the user file does not exist.
    """
    user_path = get_user_config_path()

    if os.path.exists(user_path):
        return

    user_dir = os.path.dirname(user_path)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    shutil.copy2(get_default_config_path(), user_path)
    print(f"Created user config: {user_path}")
    print("Edit this file to customize VMD settings:")
    print(f"{user_path}")


init_user_config()

_CONFIG = load_config()

VMD_RESOLUTION = _CONFIG.get("resolution", 12)
VMD_MATERIAL = _CONFIG.get("material", "AOEdgy")
VMD_BACKGROUND = _CONFIG.get("background", "white")
VMD_ELEMENT_COLORS = _CONFIG.get("element_colors", {})
VMD_ELEMENT_RADII = _CONFIG.get("element_radii", {})
VMD_BOND_CUTOFFS = _CONFIG.get("bond_cutoffs", {})
VMD_MATERIAL_SETTINGS = _CONFIG.get("material_settings", {})
VMD_MATERIAL_OPTIONS = _CONFIG.get("material_options", {}).get("options", [])
