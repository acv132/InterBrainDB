import os
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yaml

from data.config import data_dir, file
from utils.data_loader import load_database

# === Load Database ===
df = load_database(data_dir, file)

# === Load Tooltips ===
with open("./data/info_yamls/categories.yaml", "r") as f:
    label_tooltips = yaml.safe_load(f)
with open("./data/info_yamls/category_descriptions.yaml", 'r') as f:
    category_tooltips = yaml.safe_load(f)

# === File Path to Save Submissions ===
submission_file = os.path.join(data_dir, "submitted_articles.xlsx")

# === Load or Initialize Submission Data ===
if os.path.exists(submission_file):
    submitted_df = load_database(data_dir, "submitted_articles.xlsx")
else:
    submitted_df = pd.DataFrame()

st.title("🆕 Submission of Article Suggestion")
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 16pt !important;
    }
    </style>
    """, unsafe_allow_html=True
    )

st.markdown("Fill in the fields below to suggest a new article for inclusion in the database.")

# === Mandatory Fields ===
st.subheader("🔒 Required Information")
doi = st.text_input("DOI", placeholder="10.1234/example.doi")
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

if not validate_doi(doi) and doi != "":
    st.error("Invalid DOI. Please ensure it is in the correct format and exists.")

year = st.number_input("Year", min_value=1900, max_value=datetime.now().year, value=datetime.now().year, step=1)
authors = st.text_area("Authors", placeholder="Lastname, Firstname;\nLastname2, Firstname2;\nLastname3, Firstname3")
title = st.text_area("Title")
abstract = st.text_area("Abstract")

# === Optional Categorizations ===
st.subheader("🏷️ Optional Categorization")
optional_inputs = {}

for category, label_dict in label_tooltips.items():
    if not isinstance(label_dict, dict):
        continue

    options = sorted(label_dict.keys())
    helptext = category_tooltips.get(category, "")
    label_info = "\n".join(
        [f"- **{label}**: {label_dict[label]}" for label in options if label in label_dict]
    )
    full_help = f"{helptext}\n\n{label_info}".strip()

    selected = st.multiselect(
        f"{category.capitalize()}",
        options=options,
        help=full_help,
        placeholder=f"Select {category.lower()} (optional)"
    )
    optional_inputs[category] = selected

# === Submission Button ===
if st.button("📤 Submit Suggestion"):
    missing_fields = []
    if not doi:
        missing_fields.append("DOI")
    if not title:
        missing_fields.append("Title")
    if not authors:
        missing_fields.append("Authors")
    if not abstract:
        missing_fields.append("Abstract")

    if missing_fields:
        st.error(f"Please fill in the required fields: {', '.join(missing_fields)}")
    else:
        new_entry = {
            "doi": doi,
            "year": year,
            "authors": authors,
            "title": title,
            "abstract": abstract
        }
        # Include optional labels
        new_entry.update(optional_inputs)

        # Normalize for comparison
        new_doi = new_entry["doi"].strip().lower()

        # === Check if DOI exists in main database or submitted suggestions ===
        existing_dois = set()
        if "doi" in df.columns:
            existing_dois.update(df["doi"].astype(str).str.lower())
            existing_dois.update(submitted_df["doi"].astype(str).str.lower())

        if new_doi in existing_dois:
            st.info("ℹ️ This article has already been submitted or is in the database.")
        else:
            new_row = pd.DataFrame([new_entry])
            submitted_df = pd.concat([submitted_df, new_row], ignore_index=True)
            submitted_df.to_excel(submission_file, index=False)

            st.success("✅ Article suggestion submitted successfully!")
            st.balloons()