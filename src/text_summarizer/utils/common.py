import os 
from box.exceptions import BoxValueError 
import yaml
from text_summarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the yaml file.
        
    Returns:
        ConfigBox: A ConfigBox object containing the yaml data.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
        return ConfigBox(content)
    except BoxValueError:
        raise BoxValueError("YAML file is empty or has invalid content.")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    """Creates directories if they don't exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of directories.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def get_size(path: Path, unit: str = "kb") -> float:
    """Returns the size of a file in the specified unit.
    
    Args:
        path (Path): Path to the file.
        unit (str): Unit for size ('kb', 'mb', 'gb').
        
    Returns:
        float: Size of the file in the specified unit.
    """
    size_in_bytes = os.path.getsize(path)
    if unit == "kb":
        return size_in_bytes / 1024
    elif unit == "mb":
        return size_in_bytes / (1024 ** 2)
    elif unit == "gb":
        return size_in_bytes / (1024 ** 3)
    else:
        raise ValueError("Invalid unit. Choose from 'kb', 'mb', or 'gb'.")
        