from pathlib import Path
import yaml


def load_yaml_file(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML data as dictionary.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)