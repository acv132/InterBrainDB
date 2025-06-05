import ast
import io
import os
from ast import literal_eval

import tabulate
import pandas as pd
import streamlit as st


def create_article_handle(row):
    author_snippet = row['author'].replace(",","").split(" ")
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
        return pd.read_csv(os.path.join(data_dir, file), sep=";")
    elif file.endswith(".xlsx"):
        return pd.read_excel(os.path.join(data_dir, file))
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
