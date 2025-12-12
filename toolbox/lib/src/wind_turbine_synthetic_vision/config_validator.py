# Wind Turbine Synthetic Vision
# Copyright (C) 2025 Arash Shahirpour, Jakob Gebler, Manuel Sanders
# Institute of Automatic Control - RWTH Aachen University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import yaml
from typing import Dict, Any
from copy import deepcopy


def deep_merge_dicts(default: Dict[Any, Any], user: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merge two dictionaries, with user values taking precedence over defaults.
    
    Args:
        default: Default configuration dictionary
        user: User-provided configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = deepcopy(default)
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def load_default_config(default_config_path: str = None) -> Dict[str, Any]:
    """
    Load the default configuration file.
    
    Args:
        default_config_path: Path to default config file. If None, uses the config.yaml
                           in the same directory as this module.
                           
    Returns:
        Default configuration dictionary
        
    Raises:
        FileNotFoundError: If default config file is not found
        yaml.YAMLError: If default config file contains invalid YAML
    """
    if default_config_path is None:
        default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")
    
    try:
        with open(default_config_path, "r") as file:
            default_config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing default configuration file: {e}")
    
    if default_config is None:
        default_config = {}
    
    return default_config


def validate_config(user_config_path: str, default_config_path: str = None) -> Dict[str, Any]:
    """
    Validate a user configuration file against defaults and fill in missing values.
        
    Args:
        user_config_path: Path to user's configuration file
        default_config_path: Path to default config file. If None, uses the config.yaml
                           in the same directory as this module.
                           
    Returns:
        Validated and complete configuration dictionary
        
    Raises:
        FileNotFoundError: If user config file is not found
        yaml.YAMLError: If config files contain invalid YAML
    """
    # Load default configuration
    default_config = load_default_config(default_config_path)
    
    # Load user configuration
    if not os.path.exists(user_config_path):
        raise FileNotFoundError(f"User configuration file not found: {user_config_path}")
    
    try:
        with open(user_config_path, "r") as file:
            user_config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing user configuration file: {e}")
    
    if user_config is None:
        user_config = {}
    
    # Merge configurations with user values taking precedence
    validated_config = deep_merge_dicts(default_config, user_config)
    
    return validated_config
