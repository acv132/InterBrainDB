import ast

import pandas as pd
import yaml


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
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Perceived luminance formula
    luminance = 0.299*r + 0.587*g + 0.114*b
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
