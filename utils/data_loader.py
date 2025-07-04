import ast
import io
import os
import re
from ast import literal_eval

import pandas as pd
import requests
import streamlit as st
import tabulate


def create_article_handle(row):
    author_snippet = row['author'].replace(",", "").split(" ")
    if len(author_snippet) == 1:
        author = author_snippet[0]
    elif len(author_snippet) == 2:
        author = author_snippet[0] + "& " + author_snippet[1]
    else:
        author = author_snippet[0] + " et al.,"
    handle = f"{author} {row['year']}"
    return handle


@st.cache_data
def load_database(data_dir, file):
    # load data
    if file.endswith(".csv"):
        return pd.read_csv(os.path.join(data_dir, file), sep=";", keep_default_na=False)
    elif file.endswith(".xlsx"):
        return pd.read_excel(os.path.join(data_dir, file), keep_default_na=False)
    else:
        raise ValueError("Unsupported file format. Please use either .csv or .xlsx.")


def generate_bibtex_content(df: pd.DataFrame) -> str:
    output = io.StringIO()
    for _, row in df.iterrows():
        output.write(f"@{row['BibTexID']},\n")
        for field in row.index:
            if field not in ['ENTRYTYPE', 'BibTexID'] and not pd.isnull(row[field]):
                value = str(row[field]).replace('{', '{{').replace('}', '}}')  # Escape braces
                output.write(f"  {field.lower()} = {{{value}}},\n")
        output.write("}\n\n")
    return output.getvalue()


def extract_unique_tags(series):
    """Split comma-separated strings into a flat unique list."""
    tags = set()
    for entry in series.dropna():
        if isinstance(entry, str):
            if entry.startswith("[") and entry.endswith("]"):
                # Handle lists stored as strings like "['EEG', 'ECG']"
                entry = ast.literal_eval(entry)
            else:
                entry = [tag.strip() for tag in entry.split(",")]
            tags.update(entry)
    return sorted(tags)


def matches_any_tag(row_val, selected):
    """Returns True if any selected tag is in the row's list of tags."""
    if pd.isna(row_val):
        return False
    if isinstance(row_val, str):
        if row_val.startswith("[") and row_val.endswith("]"):
            tags = ast.literal_eval(row_val)
        else:
            tags = [t.strip() for t in row_val.split(",")]
    else:
        tags = [row_val]
    return any(tag in tags for tag in selected)


def generate_apa7_latex_table(df):
    return tabulate.tabulate(df, headers="keys", tablefmt="latex_booktabs", showindex=False)


def generate_excel_table(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='References')
    output.seek(0)
    return output


def normalize_cell(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        try:
            return literal_eval(val) if val.startswith("[") and val.endswith("]") else [val]
        except:
            return [val]
    if pd.isna(val):
        return []
    return [val]


def validate_doi(doi):
    '''
    Tests if the provided DOI is valid.
    '''

    def is_valid_doi(doi):
        pattern = r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$"
        return bool(re.match(pattern, doi, re.IGNORECASE))

    def doi_exists(doi):
        url = f"https://doi.org/{doi}"
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200

    return is_valid_doi(doi) and doi_exists(doi)


def flatten_cell(x):
    # If actual list of length 1, take its element
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    # If string that looks like a Python list, try parsing
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and len(parsed) == 1:
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return x


def hedges_g_between(groups, apply_correction=True):
    """
    Compute Hedges' g from any number of groups.

    Parameters:
    - groups: list of tuples (mean, std, n)
    - apply_correction: whether to apply small sample correction (default: True)

    Returns:
    - g: Hedges' g (standardized mean difference between first two groups)
    - s_pooled: pooled standard deviation
    """

    if len(groups) < 2:
        raise ValueError("At least two groups are needed to compute Hedges' g.")

    # Extract values
    means = [m for m, s, n in groups]
    variances = [(n - 1) * (s ** 2) for m, s, n in groups]
    total_df = sum(n - 1 for _, _, n in groups)

    # Compute pooled standard deviation
    s_pooled = np.sqrt(sum(variances) / total_df)

    # Difference between first two means
    mean_diff = means[0] - means[1]

    # Raw Hedges' g
    g = mean_diff / s_pooled

    if apply_correction:
        N_total = sum(n for _, _, n in groups)
        correction = 1 - (3 / (4 * N_total - 9)) if N_total > 9 else 1
        g *= correction

    return g, s_pooled


def hedges_g_within(subject_diffs, apply_correction=True):
    """
    Computes Hedges' g for within-subjects (paired) designs.

    Parameters:
    - subject_diffs: array-like of difference scores (Condition A - Condition B)
    - apply_correction: apply small-sample correction (default True)

    Returns:
    - g_z: bias-corrected standardized mean difference
    - d_z: raw effect size (Cohen's d_z)
    """
    diffs = np.asarray(subject_diffs)
    M_D = np.mean(diffs)
    SD_D = np.std(diffs, ddof=1)
    d_z = M_D / SD_D

    n = len(diffs)
    if apply_correction:
        correction = 1 - (3 / (4 * n - 1))
        g_z = d_z * correction
    else:
        g_z = d_z

    return g_z, d_z
