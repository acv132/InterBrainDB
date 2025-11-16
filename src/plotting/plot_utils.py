from __future__ import annotations

import ast
import re
# new: src/utils/data_loader_safe.py (or place near your plotting code)
# New: robust normalizer that never evaluates list-like NA in a boolean context
from ast import literal_eval

import numpy as np
import pandas as pd
import yaml
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


def get_scenario_or_medium(coord, axis, scenario_order, medium_order, row_pos, col_pos):
    if axis == 'y':  # For y-coordinate (scenarios)
        return scenario_order[list(row_pos).index(coord)]
    elif axis == 'x':  # For x-coordinate (mediums)
        return medium_order[list(col_pos).index(coord)]
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
    """
    Normalize any cell into a list, safely handling:
    - scalars and strings
    - strings that look like lists, e.g. "[EEG, EDA]"
    - list-like objects (list/tuple/set/np.array/Series/Index)
    """
    return _norm_list_safe(val)


def _norm_list(val):
    """
    Backwards-compatible wrapper to normalize a cell into a list,
    using the robust _norm_list_safe implementation.
    """
    return _norm_list_safe(val)


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
        """
        Deduplicate while preserving order.
        Works even if items are unhashable (e.g. lists) by using a string key.
        """
        seen = set()
        out = []
        for x in seq:
            try:
                # use x directly if hashable
                key = x
                hash(key)  # will raise TypeError for unhashable types
            except TypeError:
                # fall back to a stable string representation
                key = repr(x)

            if key not in seen:
                seen.add(key)
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


def wrap_label(text, width=12):
    """Insert line breaks every `width` characters for axis label wrapping."""
    if not isinstance(text, str):
        text = str(text)
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def to_list_cell(x):
    """Cell parser: robustly turn any representation into a list of strings"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if str(i).strip()]
        except Exception:
            pass
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
    elif "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [s]
    return [p for p in parts if p]


def conditions_from_row(row):
    """
    Return a list of (scenario, medium) condition tuples for a single row,
    applying ordered pairing semantics:

    - If len(IS) == len(IM) > 0: pair in order: (IS[i], IM[i]).
    - If one side has length 1 and the other > 1: broadcast the single value.
    - Else (both > 1 but unequal): zip up to min length (conservative).
    """
    scenarios = ensure_list(row["interaction scenario"])
    mediums = ensure_list(row["interaction medium"])

    if not scenarios or not mediums:
        return []

    n_s = len(scenarios)
    n_m = len(mediums)

    conditions = []
    if n_s == n_m:
        # your explicit rule: [a,b,c] × [d,e,f] -> (a,d), (b,e), (c,f)
        for s, m in zip(scenarios, mediums):
            conditions.append((s, m))
    elif n_s == 1 and n_m > 1:
        # single IS, multiple IM
        for m in mediums:
            conditions.append((scenarios[0], m))
    elif n_m == 1 and n_s > 1:
        # multiple IS, single IM
        for s in scenarios:
            conditions.append((s, mediums[0]))
    else:
        # unequal lengths > 1 on both sides: fall back to ordered pairs up to min length
        for s, m in zip(scenarios, mediums):
            conditions.append((s, m))

    return conditions
