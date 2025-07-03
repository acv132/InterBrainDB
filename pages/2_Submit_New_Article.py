import os
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

from data.config import data_dir, file
from utils.data_loader import load_database, validate_doi

# ========================
# 💅 UI Configuration
# ========================
st.set_page_config(page_title="Living Literature Review",
                   page_icon='assets/favicon.ico',
                   layout="wide")
st.title("🆕 Submission of New Article")
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 16pt !important;
    }
    </style>
    """, unsafe_allow_html=True
    )

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

col1,col2 = st.columns([1,1])
with col1:
    # === Mandatory Fields ===
    st.subheader("🔒 Required Information")
    st.markdown("Fill in the fields below to suggest a new article for inclusion in the database.")

    doi = st.text_input("DOI", placeholder="10.1234/example.doi")

    if not validate_doi(doi) and doi != "":
        st.error("Invalid DOI. Please ensure it is in the correct format and exists.")

    year = st.number_input("Year", min_value=1900, max_value=datetime.now().year, value=datetime.now().year, step=1)
    authors = st.text_area("Authors", placeholder="Lastname, Firstname;\nLastname2, Firstname2;\nLastname3, Firstname3")
    title = st.text_area("Title")
    abstract = st.text_area("Abstract")

    # === Optional Categorizations ===
    st.subheader("🏷️ Optional Categorization")

    st.markdown("Optionally you may already categorize the article based on the categories below. If"
                " you are unsure, you can leave these fields empty and the reviewers will categorize it for you.")
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

with col2:
    # === Submission Button ===
    st.markdown("#### 📨 Submit Your Article Suggestion")
    st.markdown("Once you have filled in the required information, you can submit your article suggestion for review. "
                "The reviewers will then check the article and categorize it if necessary. "
                "If you have any questions, do not hesitate to contact us.")
    if doi and title and authors and abstract:
        st.info("The required information is filled in. Click the button below to submit your article suggestion.")
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