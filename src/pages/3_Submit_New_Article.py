# ========================
# 📦 Imports & Setup
# ========================
import os
from datetime import datetime

import bibtexparser
import pandas as pd
import streamlit as st
import yaml

from src.utils.app_utils import footer, set_mypage_config
from src.utils.config import data_dir, file
from src.utils.data_loader import load_database, validate_doi, generate_bibtexid

# ========================
# 💅 UI Configuration
# ========================
set_mypage_config()
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

# Initialize session state defaults
for key, default in {
    'doi': '', 'year': datetime.now().year, 'authors': '', 'title': '',
    }.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ========================
# 📥 Load & Prepare Data
# ========================
# === Load Database ===
df = load_database(data_dir, file)
with open("./data/info_yamls/categories.yaml", "r") as f:
    label_tooltips = yaml.safe_load(f)
with open("./data/info_yamls/category_descriptions.yaml", 'r') as f:
    category_tooltips = yaml.safe_load(f)

# === Load or Initialize Submission Data ===
submission_file_name = "submitted_articles.csv"
if "submitted_df" not in st.session_state:
    try:
        st.session_state.submitted_df = load_database(data_dir, submission_file_name)
    except FileNotFoundError:
        st.session_state.submitted_df = pd.DataFrame()

submitted_df = st.session_state.submitted_df

# ========================
# 📋 BibTeX Parser
# ========================
col1, col2 = st.columns([1, 1], gap="medium")
with col2:
    # === Paste BibTeX Entry ===
    st.markdown("#### 📋 Paste BibTeX Entry")
    st.markdown("Paste a BibTeX entry here to auto-fill the required fields.")
    bib_input = st.text_area(label="BibTeX", value='', height=150, )
    if st.button("Parse BibTeX"):
        try:
            bib_db = bibtexparser.loads(bib_input)
            if bib_db.entries:
                entry = bib_db.entries[0]
                st.session_state.doi = entry.get('doi', st.session_state.doi)
                entry['title'] = entry.get('title', '').replace('{', '').replace('}', '')  # Clean title
                st.session_state.title = entry.get('title', st.session_state.title)
                # Convert BibTeX authors ("A and B and C") to lines
                authors = entry.get('author', '')
                if authors:
                    st.session_state.authors = authors.replace(' and ', ';\n')
                # Year
                year = entry.get('year')
                if year and year.isdigit():
                    st.session_state.year = int(year)
                st.success("✅ Parsed BibTeX and updated fields!")
            else:
                st.error("No entries found in BibTeX input.")
        except Exception as e:
            st.error(f"Error parsing BibTeX: {e}")

# ========================
# 🆕 Article Submission Form
# ========================
with col1:
    # === Mandatory Fields ===
    st.subheader("🔒 Required Information")
    st.markdown("Fill in the fields below to suggest a new article for inclusion in the database.")

    doi = st.text_input(
        "DOI", value=st.session_state.doi, placeholder="10.1234/example.doi", key='doi'
        )

    if not validate_doi(doi) and doi != "":
        st.error("Invalid DOI. Please ensure it is in the correct format and exists.")

    year = st.number_input(
        "Year", min_value=1900, max_value=datetime.now().year, step=1, key='year'
        )
    authors = st.text_area(
        "Authors",
        value=st.session_state.authors,
        placeholder="Lastname, Firstname;\nLastname2, Firstname2;\nLastname3, Firstname3",
        key='authors'
        )
    title = st.text_area(
        "Title", value=st.session_state.title, key='title'
        )

    # === Optional Categorizations ===
    st.subheader("🏷️ Optional Categorization")

    st.markdown(
        "Optionally you may already categorize the article based on the categories below. If"
        " you are unsure, you can leave these fields empty and the reviewers will categorize it for you."
        )
    optional_inputs = {}

    for category, label_dict in label_tooltips.items():
        if not isinstance(label_dict, dict):
            continue

        if category == "sample":
            optional_inputs['sample_size'] = st.number_input(
                "Enter sample size (optional)",
                min_value=2,
                value=None,
                help="Number of participants analysed in the study.",
                key='sample_size'
                )

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
        if category == "measurement modality":
            if "fNIRS" in selected:
                optional_inputs["fNIRS_channel_number"] = st.number_input(
                    "fNIRS channel number",
                    min_value=1,
                    value=None,
                    help="Number of fNIRS channels used in the study.",
                    key='fNIRS_channel_number'
                    )
            if "EEG" in selected:
                optional_inputs["EEG_electrode_number"] = st.number_input(
                    "EEG electrode number",
                    min_value=1,
                    value=None,
                    help="Number of EEG electrodes used in the study.",
                    key='EEG_electrode_number'
                    )

        # If "more" is selected in sample, ask for number of groups
        if category == "pairing configuration":
            if "more (n group = x)" in optional_inputs['pairing configuration'] and "more (n group = x)" in selected:
                n_groups = st.number_input(
                    "Number of participants in group (n group = x)",
                    min_value=5,
                    help="Specify the number of participants per group if more than a tetrad was formed.",
                    key='n_groups'
                    )
                optional_inputs['pairing configuration'] = f"more (n group = {n_groups}"

    # === Other Labels ===
    other_separator = ','
    other_labels = st.text_area(
        "Other Labels",
        key='other_labels',
        help=f"Add any other relevant labels that do not fit "
             f"into the "
             "categories above. Separate multiple labels with a comma "
             f"({other_separator}).",
        placeholder=f"label1{other_separator} label2{other_separator} ... (optional)", )

    optional_inputs['other_labels'] = [item for item in other_labels.split(other_separator) if item.strip()]
    if "fNIRS_channel_number" in optional_inputs:
        optional_inputs["other_labels"] = [f"fNIRS channel number: {optional_inputs['fNIRS_channel_number']}"] + \
                                          optional_inputs["other_labels"]
        optional_inputs.pop("fNIRS_channel_number", None)
    if "EEG_electrode_number" in optional_inputs:
        optional_inputs["other_labels"] = [f"EEG electrode number: {optional_inputs['EEG_electrode_number']}"] + \
                                          optional_inputs["other_labels"]
        optional_inputs.pop("EEG_electrode_number", None)

# ========================
# 📨 Database Check & Submission
# ========================
with col2:
    st.markdown('---')
    st.markdown("#### 🔍 Check Database for Existing DOIs")
    st.markdown(
        "You may want to check if the DOI you are submitting already exists in or has been submitted to "
        "the database before filling in the remaining fields. You can do this quickly by clicking the button below. "
        )
    if st.button("☑️ Check Database"):
        existing_dois = set()
        if "doi" in df.columns:
            existing_dois.update(df["doi"].astype(str).str.lower())
        if "doi" in submitted_df.columns:
            existing_dois.update(submitted_df["doi"].astype(str).str.lower())
        if doi:
            if doi.strip().lower() in existing_dois:
                st.warning(
                    f"⚠️ An article with the DOI {doi} has already been submitted or is part of the "
                    f"database."
                    )
            else:
                st.info("ℹ️ This DOI is not found in the database. You may proceed with submission.")

    # === Submission Button ===
    st.markdown('---')
    st.markdown("#### 📨 Submit Your Article Suggestion")
    st.markdown(
        "Once you have filled in the required information, you can submit your article suggestion for review. "
        "The reviewers will then check the article and categorize it if necessary. "
        "If you have any questions, do not hesitate to contact us."
        )
    if doi and title and authors:
        st.info("The required information is filled in. Click the button below to submit your article suggestion.")
    if st.button("📤 Submit Suggestion"):
        try:
            missing_fields = []
            if not doi:
                missing_fields.append("DOI")
            if not title:
                missing_fields.append("Title")
            if not authors:
                missing_fields.append("Authors")

            if missing_fields:
                st.error(f"❗Please fill in the required fields: {', '.join(missing_fields)}")
            else:
                authors = authors.replace('\n', ' ').replace(";", " and ").replace("  ", " ")
                # Create new entry dictionary
                new_entry = {
                    "doi": doi,
                    "authors": authors,
                    "year": year,
                    "title": title,
                    "included in paper review": "False",
                    "exclusion_reasons": "submission after publication",
                    }
                # Include optional labels
                new_entry.update(optional_inputs)
                # Replace ; in entry to avoid errors during df extension
                for key in new_entry.keys():
                    if isinstance(new_entry[key], str):
                        new_entry[key].replace(";", ",")
                # Normalize for comparison
                new_doi = new_entry["doi"].strip().lower()

                # === Check if DOI exists in main database or submitted suggestions ===
                existing_dois = set()
                if "doi" in df.columns:
                    existing_dois.update(df["doi"].astype(str).str.lower())
                if "doi" in submitted_df.columns:
                    existing_dois.update(submitted_df["doi"].astype(str).str.lower())

                if new_doi in existing_dois:
                    st.info("ℹ️ This article has already been submitted or is part of the database.")
                else:
                    # Create DataFrame for the new row (BibTexID will be set after generating)
                    new_row = pd.DataFrame([new_entry])

                    # ---- Generate a unique BibTexID using the helper on the combined data ----
                    frames = []
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        frames.append(df)
                    if isinstance(submitted_df, pd.DataFrame) and not submitted_df.empty:
                        frames.append(submitted_df)
                    frames.append(new_row)  # append last so we can grab its generated ID

                    combined = pd.concat(frames, ignore_index=True, sort=False)

                    # Use the robust generator (assumes it's defined/imported above)
                    combined = generate_bibtexid(
                        df=combined,
                        author_col="authors",
                        year_col="year",
                        title_col="title",
                        out_col="BibTexID",
                        inplace=True, )

                    # Pull the BibTexID for the newly added row (last row)
                    bib_id = combined.iloc[-1]["BibTexID"]
                    new_row["BibTexID"] = bib_id

                    submitted_df = pd.concat([submitted_df, new_row], ignore_index=True)
                    # Move BibTexID to pos after title
                    cols = submitted_df.columns.tolist()
                    cols.remove("BibTexID")
                    title_index = cols.index("title")
                    cols.insert(title_index + 1, "BibTexID")

                    submitted_df = submitted_df[cols]

                    st.session_state.submitted_df = submitted_df  # persist in session
                    submitted_df.to_csv(
                        os.path.join(data_dir, submission_file_name), index=False, sep=";", encoding="utf-8"
                        )

                    st.success("✅ Article suggestion submitted successfully!")
                    st.balloons()
                    # Reset input fields
                    st.session_state["doi_input"] = ""
                    st.session_state["title_input"] = ""
                    st.session_state["authors_input"] = ""

        except Exception as e:
            st.error(f"Error during submission: {e}")
footer()
