from __future__ import annotations

import ast
import re

import pandas as pd
import yaml

from src.utils.data_loader import normalize_cell

# new: src/utils/data_loader_safe.py (or place near your plotting code)
from ast import literal_eval
from typing import List
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
# New: robust normalizer that never evaluates list-like NA in a boolean context
from ast import literal_eval
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

def safe_literal_eval(value):
    try:
        # Evaluate the value if it seems like a list
        return ast.literal_eval(value.strip()) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return value


def decode_one_hot(df, prefix="_"):
    yaml_file = "./data/info_yamls/category_descriptions.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        cols_to_convert = yaml.safe_load(f)

    for col in cols_to_convert.keys():
        try:
            df[col] = df[col].apply(safe_literal_eval)
        except TypeError:
            pass
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x])
        unique_categories = set(cat for cats in df[col] for cat in cats)
        for category in unique_categories:
            df[f"{col}{prefix}{category}"] = df[col].apply(lambda x: category in x)
        df = df.drop(columns=[col])

    return df


def is_horizontal_or_vertical(x1, y1, x2, y2):
    return (x1 == x2) or (y1 == y2)


def convert_to_horizontal_and_vertical_connections(element):
    """
    Converts a diagonal connection into one horizontal and one vertical connection.

    Parameters:
        element (tuple): A tuple containing two points, where each point is a tuple (x, y).

    Returns:
        list: A list of horizontal and vertical connections as tuples.
              Each tuple represents a connection (start_point, end_point).
    """
    new_connections = []

    # Unpack the points
    (x1, y1), (x2, y2) = element

    # Define the intermediate point
    intermediate_point = (x2, y1)
    intermediate_point2 = (x1, y2)

    # Create horizontal connection (from (x1, y1) to intermediate point)
    new_connections.append(((x1, y1), intermediate_point))
    new_connections.append(((x1, y1), intermediate_point2))

    # Create vertical connection (from intermediate point to (x2, y2))
    new_connections.append((intermediate_point, (x2, y2)))
    new_connections.append((intermediate_point2, (x2, y2)))

    # Return the list of connections
    return new_connections


def get_scenario_or_manipulation(coord, axis, scenario_order, manipulation_order, row_pos, col_pos):
    if axis == 'y':  # For y-coordinate (scenarios)
        return scenario_order[list(row_pos).index(coord)]
    elif axis == 'x':  # For x-coordinate (manipulations)
        return manipulation_order[list(col_pos).index(coord)]
    return None


def is_dark_color(hex_color):
    """Return True if color is dark, based on luminance."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    # Perceived luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 128


def export_all_category_counts(df):
    df = decode_one_hot(df, "_")

    with open("./data/info_yamls/categories.yaml", 'r', encoding='utf-8') as f:
        category_definitions = yaml.safe_load(f)

    base_categories = list(category_definitions.keys())

    category_counts = {}
    for cat in base_categories:
        cols = [col for col in df.columns if col.startswith(f"{cat}_")]
        if cols:
            counts = df[cols].sum().rename(lambda x: x.replace(f"{cat}_", ""))
            category_counts[cat] = counts

    # Flatten into DataFrame
    export_df = pd.concat(category_counts, axis=1).fillna(0).astype(int)
    return export_df


def ensure_list(val):
    if isinstance(val, list):
        return val
    elif pd.isnull(val):
        return []
    elif isinstance(val, str):
        # Try to eval strings that look like lists
        import ast
        try:
            result = ast.literal_eval(val)
            if isinstance(result, list):
                return result
            else:
                return [result]
        except:
            return [val]
    else:
        return [val]


def _norm_list(val):
    vals = normalize_cell(val)
    if vals is None:
        return []
    return vals

def _hex_to_rgb(hx: str):
    hx = (hx or "").lstrip("#")
    if len(hx) == 3:  # e.g. #abc
        hx = "".join(ch * 2 for ch in hx)
    try:
        return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def _rgba(hx: str, a: float) -> str:
    r, g, b = _hex_to_rgb(hx)
    return f"rgba({r},{g},{b},{a})"


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(s)).strip("-").lower()



def _norm_list_safe(val) -> list:
    """
    Robustly normalize a dataframe cell into a list, handling:
      - Strings like "[EEG, EDA]" or "['EEG','EDA']"
      - Plain scalars/strings
      - List-like objects (list/tuple/set/np.array/Series/Index)
    Never evaluates array-like NA in a boolean context.
    """
    def _isna_scalar(x) -> bool:
        try:
            res = pd.isna(x)
        except Exception:
            return False
        return bool(res) if isinstance(res, (bool, np.bool_)) else False

    def _dedupe_preserve_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # --- Strings ---
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return []
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            # 1) Try Python literal first (handles "['EEG','EDA']")
            try:
                parsed = literal_eval(s)
                if is_list_like(parsed) and not isinstance(parsed, (str, bytes)):
                    seq = list(parsed)
                else:
                    seq = [parsed]
            except Exception:
                # 2) Fallback for unquoted items: "[EEG, EDA]" -> ["EEG","EDA"]
                #    Split on commas not inside nested brackets (rare here)
                parts = [p.strip() for p in inner.split(",")]
                # strip any stray quotes
                parts = [re.sub(r"""^['"]|['"]$""", "", p) for p in parts]
                # remove empties and NA-like
                seq = [p for p in parts if p and p.lower() not in ("nan", "none")]
            # filter scalar-NA and dedupe
            seq = [x for x in seq if not _isna_scalar(x)]
            return _dedupe_preserve_order(seq)
        # plain scalar string
        return [] if _isna_scalar(s) else [s]

    # --- List-like but not strings/bytes ---
    if is_list_like(val) and not isinstance(val, (str, bytes)):
        try:
            seq = list(val)
        except Exception:
            seq = [val]
        seq = [x for x in seq if not _isna_scalar(x)]
        return _dedupe_preserve_order(seq)

    # --- Scalar (incl. None/NaT) ---
    return [] if _isna_scalar(val) else [val]

def _norm_list_safe_str(val) -> list[str]:
    return [str(x) for x in _norm_list_safe(val)]