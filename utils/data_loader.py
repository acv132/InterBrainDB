from __future__ import annotations
import ast
import io
import os
import re
from ast import literal_eval

import pandas as pd
import requests
import streamlit as st
import tabulate



import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd

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
        output.write(f"@{list(row['BibTexID'])[0]},\n")
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


def generate_csv_table(df: pd.DataFrame, encoding: str = "utf-8-sig") -> bytes:
    """
    Serialize a DataFrame to CSV bytes.
    Uses 'utf-8-sig' so Excel detects UTF-8 correctly on Windows.
    """
    csv_str = df.to_csv(index=False, sep=";")
    return csv_str.encode(encoding)


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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"
            }
        response = requests.get(url, allow_redirects=True, headers=headers)
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


def create_tab_header(df, display_df):
    st.markdown(f"Total studies in database: N = {len(df)}")
    st.markdown(f"Currently included studies: N = {len(display_df)}")
    avg_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').mean()
    sd_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').std()
    sem_sample_size = round(sd_sample_size / (len(display_df) ** 0.5) if len(display_df) > 0 else 0, 2)
    min_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').min()
    max_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').max()
    st.markdown(
        f"Descriptives of sample size: Mean = {avg_sample_size:.1f} ± {sd_sample_size:.1f} (SEM = "
        f"{sem_sample_size}), "
        f"min: {min_sample_size}, max: {max_sample_size}"
        )


def generate_bibtexid(
    df: pd.DataFrame,
    *,
    author_col: str = "author",
    year_col: str = "year",
    title_col: str = "title",
    out_col: str = "BibTexID",
    min_title_len: int = 12,
    max_title_len: int = 40,
    prefer_ascii: bool = True,
    inplace: bool = True,
) -> pd.DataFrame | pd.Series:
    """
    Create unique BibTeX-like IDs from author, year, and title.

    Pattern: <first-author-surname><year><cleaned-title-fragment>

    First author surname is derived from the first-listed author in author_col.

    Year is coerced to a 4-digit string when possible; otherwise 'nd' (no date).

    Title fragment is cleaned (lowercase, punctuation removed) and truncated.

    If duplicates remain, disambiguate with alphabetic suffixes: a, b, c, ..., aa, ab, ...

    Args:
    df: DataFrame containing the metadata.
    author_col: Column with author(s). Accepts strings (e.g. "Smith, John and Doe, Jane")
    or list-like. First author's surname is used. (default: "author")
    year_col: Column with the publication year (int/str). Non-4-digit values become 'nd'. (default: "year")
    title_col: Column with article title. (default: "title")
    out_col: Name of the column to write IDs to. (default: "BibTexID")
    min_title_len: Initial number of title characters to include. (default: 12)
    max_title_len: Max number of title characters to try before adding suffixes. (default: 40)
    prefer_ascii: If True, transliterate accented chars to ASCII (e.g., Müller -> Muller). (default: True)
    inplace: If True, writes the IDs to df[out_col] and returns the DataFrame; if False, returns a Series. (default: True)

    Returns:
    DataFrame with out_col added (if inplace=True) or a Series of IDs (if inplace=False).

    Raises:
    TypeError: If df is not a pandas DataFrame.
    ValueError: If required columns are missing, or if min_title_len/max_title_len are invalid.

    Examples:
    >> import pandas as pd
    >> df = pd.DataFrame({
    ... "author": ["Müller, Anna and Li, Bo", "Smith, John", ["Doe, Jane", "Roe, Pat"]],
    ... "year": [2021, "2019", None],
    ... "title": ["Neural coupling in conversation", "A study of brains", "Untitled note"],
    ... })
    >> # 1) In-place: add 'BibTexID' column to df
    >> df = generate_bibtexid(df)
    >> df[["author", "year", "title", "BibTexID"]].head()
    >>
    >> # 2) Get the IDs as a Series without modifying df
    >> ids = generate_bibtexid(df, inplace=False, min_title_len=10, max_title_len=30)
    >> ids.head()
    >>
    >> # 3) Preserve diacritics (do not transliterate to ASCII)
    >> df = generate_bibtexid(df, prefer_ascii=False)
    """

    # --------- validation ---------
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing = [c for c in (author_col, year_col, title_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not (isinstance(min_title_len, int) and isinstance(max_title_len, int)):
        raise ValueError("min_title_len and max_title_len must be integers.")
    if min_title_len <= 0 or max_title_len <= 0 or max_title_len < min_title_len:
        raise ValueError("Ensure 0 < min_title_len <= max_title_len.")

    work = df if inplace else df.copy(deep=False)  # shallow copy is fine; we only create new series

    # --------- helpers ---------
    def _to_ascii(s: str) -> str:
        # Normalize unicode to NFKD and strip non-ASCII; keep alphanumerics only later.
        if not isinstance(s, str):
            s = "" if pd.isna(s) else str(s)
        if not prefer_ascii:
            return s
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    def _first_author_surname(x) -> str:
        """
        Extract first author's surname from diverse formats:
        - "Smith, John and Doe, Jane"
        - "John Smith; Jane Doe"
        - ["Smith, John", "Doe, Jane"]
        - "Smith et al."
        """
        if pd.isna(x):
            return "anon"

        # If list-like (e.g., list of authors), take the first element as string
        if isinstance(x, (list, tuple)):
            x = str(x[0]) if x else ""

        s = str(x)
        # Split on common separators for multiple authors
        parts = re.split(r"\s+and\s+|;|,?\s*&\s*|\s*·\s*|\s*,\s*et\s*al\.?", s, flags=re.I)
        first = parts[0].strip() if parts else s.strip()

        # If "Last, First" -> take before comma
        if "," in first:
            last = first.split(",", 1)[0].strip()
        else:
            # "First Middle Last" -> take last token
            tokens = first.split()
            last = tokens[-1] if tokens else "anon"

        last = _to_ascii(last).lower()
        last = re.sub(r"[^a-z0-9]+", "", last) or "anon"
        return last

    def _clean_title(t) -> str:
        if pd.isna(t):
            t = ""
        t = _to_ascii(str(t)).lower()
        # remove punctuation, spaces, hyphens, parentheses, colons, quotes, etc.
        t = re.sub(r"[^a-z0-9]+", "", t)
        return t or "untitled"

    def _year_to_str(y) -> str:
        if pd.isna(y):
            return "nd"
        s = str(int(y)) if isinstance(y, (int, float)) and not pd.isna(y) else str(y)
        # Extract a 4-digit year if present
        m = re.search(r"(1[5-9]\d{2}|20\d{2}|21\d{2})", s)  # generous range 1500–2199
        return m.group(1) if m else "nd"

    def _suffix_from_index(idx: int) -> str:
        """
        Convert 1->'a', 2->'b', ... 26->'z', 27->'aa', etc.
        idx is the duplicate index within a group (0 means no suffix).
        """
        if idx <= 0:
            return ""
        # Convert to 1-based for alphabet run
        n = idx
        letters = []
        while n > 0:
            n -= 1
            letters.append(chr(ord('a') + (n % 26)))
            n //= 26
        return "".join(reversed(letters))

    # --------- base fields ---------
    authors = work[author_col].map(_first_author_surname)
    years = work[year_col].map(_year_to_str)
    titles_clean = work[title_col].map(_clean_title)

    # --------- iterative length increase to avoid collisions ---------
    title_len = min_title_len
    while True:
        bases = authors + years + titles_clean.str.slice(0, title_len)
        if bases.nunique(dropna=False) == len(bases) or title_len >= max_title_len:
            break
        title_len += 1

    # If still not unique (even at max length), add alphabetical suffixes
    if bases.nunique(dropna=False) != len(bases):
        # For each duplicate group, assign a cumcount and map to a/b/c...
        dup_idx = bases.groupby(bases).cumcount()
        suffixes = dup_idx.map(_suffix_from_index)
        ids = bases + suffixes
    else:
        ids = bases

    # Final sanity clean (should be no effect, but keep it safe)
    ids = ids.fillna("").astype(str)
    ids = ids.str.slice(0, 255)  # safety cap for very long IDs

    if inplace:
        work[out_col] = ids
        return work
    else:
        return ids


def custom_column_picker(available_cols) -> list:
    custom_key = "custom_column_selection"
    # Preselect previously chosen columns or default to all
    preselected = st.session_state.get(custom_key, available_cols)

    # Small helper row
    col_1, col_2 = st.columns([1,19,], vertical_alignment="center")
    with col_1:
        if st.button("",icon=":material/checklist_rtl:"):
            st.session_state[custom_key] = available_cols
            custom_cols = available_cols
    with col_2:
        custom_cols = st.multiselect(
            "Choose columns to display (order is preserved by selection):",
            options=available_cols,
            default=[c for c in preselected if c in available_cols],
            key=custom_key,
            placeholder="Select columns…",
            )
    # Fallback if user clears everything
    if not custom_cols:
        st.info("No columns selected. Showing all columns for now.")
        column_order = available_cols
    else:
        column_order = custom_cols
    return column_order
