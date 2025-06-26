# ========================
# 📦 Imports & Setup
# ========================
import ast
import io

import streamlit as st
import yaml

import data.config
from plotting.figures import generate_interaction_figure, generate_category_counts_figure, \
    generate_interaction_figure_streamlit
from utils.data_loader import (load_database, create_article_handle, generate_bibtex_content, generate_apa7_latex_table,
                               normalize_cell, generate_excel_table)

# ========================
# 💅 UI Configuration
# ========================
st.set_page_config(page_title="Living Literature Review", layout="wide")
st.title("📖 Database")
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 16pt !important;
    }
    </style>
    """, unsafe_allow_html=True
    )

# ========================
# 🗂️ Tabs
# ========================
data_overview_tab, data_plots_tab = st.tabs(["📋 Data Overview", "📈 Plots"])

# ========================
# 📥 Load & Prepare Data
# ========================
df = load_database(data.config.data_dir, data.config.file)
df.rename(columns={"ID": "BibTexID"}, inplace=True)

# Create display-ready dataframe
display_df = df.copy().drop(
    columns=['rayyan_ID', # 'exclusion_reasons',
        # 'user_notes',
        # 'other_labels',
        "sample"]
    )
display_df["article"] = df.apply(create_article_handle, axis=1)
display_df["DOI Link"] = "https://doi.org/" + df["doi"]
display_df["sample size"] = df.apply(lambda row: f"N = {row['sample size']}", axis=1)
display_df.set_index("article", inplace=True)

# 🏷️ Load tooltips
with open("./data/info_yamls/categories.yaml", "r") as f:
    label_tooltips = yaml.safe_load(f)
with open("./data/info_yamls/category_descriptions.yaml", 'r') as f:
    category_tooltips = yaml.safe_load(f)

# ========================
# 🧰 Sidebar Filters
# ========================
with st.sidebar:
    st.title("🔍 Filters")

    # --- Paper Filter ---
    # todo at some point change to binary
    st.markdown("**Included in Paper Review**")
    paper_options = display_df['included in paper review'].unique()
    included_multiselect_key = "included_select"

    # Button to select all articles
    if st.button("Select all articles", key="included_selectall"):
        st.session_state[included_multiselect_key] = paper_options

    # Set default to empty unless session state holds selected articles
    included_defaults = st.session_state.get(included_multiselect_key, [])
    included_defaults = [a for a in included_defaults if a in paper_options]

    selected_inclusion = st.multiselect(
        "Choose articles based on paper review",
        options=paper_options,
        default=included_defaults,
        key=included_multiselect_key,
        label_visibility="collapsed",
        placeholder="Choose a label"
        )

    # --- Year Filter ---
    st.markdown("**Publication Year**")
    min_year = int(display_df["year"].min())
    max_year = int(display_df["year"].max())
    selected_years = st.slider(
        "Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="year_range"
        )

    # --- Reset Button ---
    if st.button("🔄 Reset Filters"):
        for key in list(st.session_state.keys()):
            if any(key.endswith(suffix) for suffix in ("_toggle", "_multiselect", "year_range", "_article_select")):
                del st.session_state[key]
        st.rerun()

    # --- Category Filters ---
    selected_filters = {}

    for category, label_dict in label_tooltips.items():
        if category not in display_df.columns:
            continue

        # Extract labels from the DataFrame column dynamically
        unique_labels = set()
        for val in display_df[category].dropna():
            if isinstance(val, list):
                unique_labels.update(val)
            elif isinstance(val, str) and val.startswith("[") and "]" in val:
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        unique_labels.update(parsed)
                    else:
                        unique_labels.add(parsed)
                except Exception:
                    unique_labels.add(val)
            else:
                unique_labels.add(val)

        all_labels = sorted(unique_labels)
        multi_key = f"{category}_multiselect"

        # Tooltip text from YAML if available
        tooltip_lines = [f"- **{label}**: {label_dict.get(label, '')}" for label in all_labels if label in label_dict]
        tooltip_text = category_tooltips[category]
        tooltip_text += "\n".join(tooltip_lines).strip()

        st.markdown(f"**{category.capitalize()}***️", help=tooltip_text)

        # Select all button
        if st.button("Select all", key=f"{category}_selectall"):
            st.session_state[multi_key] = all_labels

        # Set default to empty unless something is stored
        defaults = st.session_state.get(multi_key, [])
        defaults = [val for val in defaults if val in all_labels]

        selected = st.multiselect(
            label="", options=all_labels, default=defaults, key=multi_key, placeholder="Choose a label"
            )
        selected_filters[category] = selected

    # --- Article Filter ---
    st.markdown("**Select Articles**")
    article_options = display_df.index.tolist()
    article_multiselect_key = "article_select"

    # Button to select all articles
    if st.button("Select all articles", key="article_selectall"):
        st.session_state[article_multiselect_key] = article_options

    # Set default to empty unless session state holds selected articles
    article_defaults = st.session_state.get(article_multiselect_key, [])
    article_defaults = [a for a in article_defaults if a in article_options]

    selected_articles = st.multiselect(
        "Choose individual articles",
        options=article_options,
        default=article_defaults,
        key=article_multiselect_key,
        label_visibility="collapsed",
        placeholder="Choose an article"
        )

# ========================
# 🔍 Apply Filters
# ========================

# 1. Filter by year range
filtered_df = display_df[(display_df["year"] >= selected_years[0]) & (display_df["year"] <= selected_years[1])].copy()

# 2. Filter by article selection
if selected_inclusion:
    filtered_df = filtered_df[filtered_df['included in paper review'].apply(
        lambda x: any(tag in normalize_cell(x) for tag in selected_inclusion)
        )]

# 3. Filter by article selection
if selected_articles:
    filtered_df = filtered_df[filtered_df.index.isin(selected_articles)]

# 4. Apply category filters
for category, selected_labels in selected_filters.items():
    if category not in filtered_df.columns or not selected_labels:
        continue

    # OR within category: match if any label in the cell matches any selected label
    filtered_df = filtered_df[filtered_df[category].apply(
        lambda x: any(tag in normalize_cell(x) for tag in selected_labels)
        )]

# 5. Update global display_df
display_df = filtered_df

# ========================
# 📄 Data Overview Tab
# ========================
with data_overview_tab:
    st.markdown("This table provides an overview of the studies included in the analysis.")
    st.markdown(f"Total studies in database: N = {len(df)}")
    st.markdown(f"Currently included studies: N = {len(display_df)}")

    # Column view selector
    view_option = st.radio(
        "Select view mode:",
        options=["Dev Mode (temp)", "Default", "Participants", "Paradigm", "Measurement & Analysis", "All Columns"],
        horizontal=True
        )

    view_configs = {
        "Dev Mode (temp)": ["article", 'DOI Link', 'included in paper review', 'exclusion_reasons', 'user_notes',
                            'other_labels', ],
        "Default": ["article", 'DOI Link', 'included in paper review', "measurement modality", 'sample size',
                    "pairing configuration", "paradigm", 'cognitive function'],
        "Participants": ["article", 'DOI Link', 'sample size', "pairing configuration", "pairing setup",
                         'relationship pair'],
        "Paradigm": ["article", 'DOI Link', 'interaction scenario', 'interaction manipulative',
                     'transfer of information', 'type of communication', 'paradigm', 'task symmetry'],
        "Measurement & Analysis": ["article", 'DOI Link', "measurement modality", 'analysis method',
                                   'cognitive function'],
        "All Columns": list(display_df.columns)
        }
    column_order = view_configs.get(view_option, view_configs["Default"])

    # 🔧 Configure column help tooltips
    column_config = {}
    for col in column_order:
        if col == "DOI Link":
            column_config[col] = st.column_config.LinkColumn(label="DOI", display_text="🔗")
            continue
        category_desc = category_tooltips.get(col, "")
        label_defs = label_tooltips.get(col, {})
        label_lines = [f"- **{k}**: {v}" for k, v in label_defs.items()] if isinstance(label_defs, dict) else []
        full_help = f"{category_desc}\n\n" + "\n".join(label_lines) if label_lines else category_desc
        column_config[col] = st.column_config.Column(label=col, help=full_help.strip())

    # 📋 Show filtered dataframe
    st.dataframe(display_df, column_config=column_config, column_order=column_order, hide_index=False)

    # Create export dataframe
    export_columns = column_order + ["BibTexID"] if "BibTexID" in display_df.columns else column_order
    export_df = display_df.reset_index()[export_columns].copy()

    # 📤 Export: BibTeX
    bibtex_content = generate_bibtex_content(export_df)
    st.download_button(
        "📥 Download BibTeX", data=bibtex_content, file_name="BibExportReferences.bib", mime="application/x-bibtex"
        )
    # 📤 Export: APA7 LaTeX
    latex_table = generate_apa7_latex_table(export_df)
    st.download_button(
        "📥 Download APA7-Style LaTeX Table", data=latex_table, file_name="LatexExportReferences.tex", mime="text/plain"
        )
    # 📤 Export: Excel
    excel_table = generate_excel_table(export_df)
    st.download_button(
        "📥 Download Excel Table",
        data=excel_table,
        file_name="ExcelExportReferences.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ========================
# 📈 Data Plots Tab
# ========================
# todo: create plots with streamlit functions for cleaner lines etc. https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart
# todo add line plot depicting publication "year" titled "### Publication development"
with data_plots_tab:
    try:
        # ▶️ Interaction figure
        fig1 = generate_interaction_figure(display_df, data_plots_tab)
        fig1b = generate_interaction_figure_streamlit(display_df, data_plots_tab)
        buf = io.BytesIO()
        fig1.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        buf.seek(0)
        st.markdown("### Interaction Conditions")
        st.image(buf, use_container_width=True)
        st.download_button(
            "📥 Download Interaction Figure (PNG)",
            data=buf.getvalue(),
            file_name="interaction_figure.png",
            mime="image/png"
            )

        # ▶️ Category counts
        fig2 = generate_category_counts_figure(display_df, data_plots_tab)
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        buf.seek(0)
        st.markdown("### Category Counts")
        st.image(buf, use_container_width=True)
        st.download_button(
            "📥 Download Category Figure (PNG)", data=buf.getvalue(), file_name="categories_figure.png", mime="image/png"
            )

    except Exception as e:
        st.error(f"❌ Could not generate figures: {e}")
