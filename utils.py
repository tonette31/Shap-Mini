from __future__ import annotations
import json, os
from typing import List

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_columns(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_columns(path: str, cols: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(cols), f, indent=2)
