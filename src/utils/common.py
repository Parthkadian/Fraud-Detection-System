from pathlib import Path
import json
import joblib


def create_directories(paths: list[str]) -> None:
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_object(file_path: str, obj) -> None:
    """
    Save Python object using joblib.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str):
    """
    Load Python object using joblib.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Object file not found: {file_path}")
    return joblib.load(file_path)


def save_json(file_path: str, data: dict) -> None:
    """
    Save dictionary as JSON.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(file_path: str) -> dict:
    """
    Load dictionary from JSON file.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)