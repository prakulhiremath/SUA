"""I/O utilities for saving results, tables, and figures."""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_latex_table(df: pd.DataFrame, path: str | Path, caption: str = "",
                     label: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=False)
    if caption or label:
        lines = latex.split("\n")
        insert = []
        if caption:
            insert.append(f"\\caption{{{caption}}}")
        if label:
            insert.append(f"\\label{{{label}}}")
        lines.insert(1, "\n".join(insert))
        latex = "\n".join(lines)
    with open(path, "w") as f:
        f.write(latex)
