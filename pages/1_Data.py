# ========================
# 📦 Imports & Setup
# ========================
import ast
import io

import pandas as pd
import streamlit as st
import yaml

import data.config
from plotting.figures import generate_interaction_figure, generate_2d_cluster_plot, \
    generate_category_counts_streamlit_figure
from plotting.plot_utils import export_all_category_counts
from utils.data_loader import (load_database, create_article_handle, generate_bibtex_content, generate_apa7_latex_table,
                               normalize_cell, generate_excel_table, flatten_cell)

# ========================
# 💅 UI Configuration
# ========================
st.set_page_config(
    page_title="Living Literature Review", page_icon='assets/favicon.ico', layout="wide"
    )

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
data_overview_tab, data_plots_tab, data_plots_tab2, test_tab = st.tabs(
    ["📋 Data Overview", "📈 Plots", "📈 Plots ("
                                   "Slow)", "🔬 Test"]
    )

# ========================
# 📥 Load & Prepare Data
# ========================
df = load_database(data.config.data_dir, data.config.file)
df.rename(columns={"ID": "BibTexID"}, inplace=True)

# Create display-ready dataframe
display_df = df.copy().drop(
    columns=['rayyan_ID', # 'exclusion_reasons',
             # 'user_notes',
             # 'other_labels'
             ]
    )

# Apply flattening to the entire DataFrame
display_df = display_df.applymap(flatten_cell)

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
    st.markdown("**Included in Paper Review**")
    # todo for final version: set default to False (show all)
    include_only = st.checkbox("Show only included papers", value=True)
    if include_only:
        filtered_df = display_df[display_df['included in paper review'] == True]
    else:
        filtered_df = display_df

    # --- Keyword Search ---
    keyword = st.text_input(
        "🔎 Keyword search", placeholder="Type to filter any column", help="Filters rows if any "
                                                                          "column contains this text"
        )

    # --- Year Filter ---
    st.markdown("**Publication Year**")
    min_year = int(display_df["year"].min())
    max_year = int(display_df["year"].max())
    selected_years = st.slider(
        "Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="year_range"
        )
    st.markdown("---")
    # --- Reset Button ---
    if st.button("🔄 Reset Category Filters"):
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
        intro = category_tooltips[category].rstrip()
        list_items = [f"- **{lbl}**: {label_dict.get(lbl, '')}" for lbl in all_labels if lbl in label_dict]
        tooltip_text = intro + "\n\n" + "\n".join(list_items)

        st.markdown(f"**{category.capitalize()}**️", help=tooltip_text)

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

# 2. Filter by inclusion in paper review
if include_only:
    filtered_df = filtered_df[filtered_df['included in paper review'] == True]

# 3. Keyword filter: keep only rows where any column contains the keyword
if keyword:
    # convert all columns to string, do a case‐insensitive contains, and any() across columns
    mask = filtered_df.astype(str).apply(
        lambda row: row.str.contains(keyword, case=False, na=False).any(), axis=1
        )
    filtered_df = filtered_df[mask]
# 4. Filter by article selection
if selected_articles:
    filtered_df = filtered_df[filtered_df.index.isin(selected_articles)]

# 5. Apply category filters
for category, selected_labels in selected_filters.items():
    if category not in filtered_df.columns or not selected_labels:
        continue

    # OR within category: match if any label in the cell matches any selected label
    filtered_df = filtered_df[filtered_df[category].apply(
        lambda x: any(tag in normalize_cell(x) for tag in selected_labels)
        )]

# 6. Update global display_df
display_df = filtered_df


# ========================
# 📄 Data Overview Tab
# ========================
def create_tab_header(df, display_df):
    st.markdown(f"Total studies in database: N = {len(df)}")
    st.markdown(f"Currently included studies: N = {len(display_df)}")
    avg_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').mean()
    sd_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').std()
    sem_sample_size = round(sd_sample_size / (len(display_df) ** 0.5) if len(display_df) > 0 else 0, 2)
    min_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').min()
    max_sample_size = pd.to_numeric(display_df['sample size'].str.extract(r'(\d+)')[0], errors='coerce').max()
    st.markdown(f"Descriptives of sample size: Mean = {avg_sample_size:.1f} ± {sd_sample_size:.1f} (SEM = "
        f"{sem_sample_size}), "
        f"min: {min_sample_size}, max: {max_sample_size}"
        )


with data_overview_tab:
    create_tab_header(df, display_df)
    st.markdown("This table provides an overview of the studies included in the analysis.")

    # Column view selector
    view_option = st.radio(
        "Select view mode:",
        options=["Dev Mode (temp)", "Default", "Participants", "Paradigm", "Measurement & Analysis", "All Columns"],
        horizontal=True
        )

    view_configs = {
        'Dev Mode (temp)': ['article', 'DOI Link', 'included in paper review', 'exclusion_reasons', 'user_notes',
                            'other_labels', ],
        'Default': ['article', 'DOI Link', 'included in paper review', 'measurement modality', 'sample size',
                    'pairing configuration', 'paradigm', 'cognitive function'],
        'Participants': ['article', 'DOI Link', 'sample size', 'sample', 'pairing configuration', 'pairing setup',
                         'relationship pair'],
        'Paradigm': ['article', 'DOI Link', 'interaction scenario', 'interaction manipulative',
                     'transfer of information', 'type of communication', 'paradigm', 'task symmetry'],
        'Measurement & Analysis': ['article', 'DOI Link', 'measurement modality', 'analysis method',
                                   'cognitive function'],
        'All Columns': list(display_df.columns)
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
with data_plots_tab:
    create_tab_header(df, display_df)
    # todo add figure descriptions and download tips for all
    with st.spinner("The generation of figures may take a few seconds, please be patient...", show_time=False):
        try:
            # ▶️ Publication Year figure
            st.subheader("Publications over Time")

            # Show streamlit-native line chart
            year_counts = display_df["year"].value_counts().sort_index()
            year_df = pd.DataFrame({"Publications": year_counts})
            year_df.index = year_df.index.astype(str)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.line_chart(
                    year_df, x_label="Year", y_label="Number of Publications", use_container_width=True
                    )
            with col2:
                st.markdown(
                    """
                    💡 **Tip:** To save the figure, 
                    simply **right-click** on the chart and choose **"Save image as..."**  
                    *(wording may vary slightly depending on your browser)*
                    """
                    )

            # ▶️ Category counts
            st.subheader("Category Counts")
            fig2 = generate_category_counts_streamlit_figure(display_df, data_plots_tab)
            counts_df = export_all_category_counts(display_df)
            st.download_button(
                "📥 Download Category Counts (CSV)",
                data=counts_df.to_csv(index=True),
                file_name="category_counts.csv",
                mime="text/csv"
                )

            # ▶️ Interaction figure
            col3, col4 = st.columns([1, 1])
            with col3:
                fig1, condition_count, number_studies = generate_interaction_figure(display_df, data_plots_tab)
                # fig1b = generate_interaction_figure_streamlit(display_df, data_plots_tab)
                buf = io.BytesIO()
                fig1.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=800)
                buf.seek(0)
                st.subheader("Interaction Conditions")
                st.image(buf, use_container_width=True)
                st.pyplot(fig1)
                st.markdown(
                    f"""
                *Note*. The cross-sectional distribution all {condition_count} hyperscanning conditions of {number_studies} 
                studies across interaction manipulative and interaction scenario axes. The numbers provide the counted 
                occurrences of the combination of an interaction manipulative and scenario. The colors represent the 
                measurement modalities reported for a cross-section of conditions. The lines indicate reported 
                cross-condition comparisons separated per axis, where all horizontal connections account for scenario 
                comparisons and all vertical connections represent a manipulative comparison). The studies involving 
                a digital component either through a digital manipulative or virtual interaction scenario are marked 
                through a gray shaded area. 
                """
                    )
            with col4:
                st.markdown(
                    """
                    💡 **Tip:** The interaction figure is not based on streamlit-compatible plotting libraries, 
                    therefore you may use this download button to save the figure as a high-resolution image.
                    """
                    )
                st.download_button(
                    "📥 Download Interaction Figure (JPG)",
                    data=buf.getvalue(),
                    file_name="interaction_figure.jpg",
                    mime="image/jpg"
                    )

        except Exception as e:
            st.error(f"❌ Could not generate figures: {e}")

# ========================
# 📈 Data Plots Tab (Slow)
# ========================
with data_plots_tab2:
    create_tab_header(df, display_df)
    with st.spinner("The generation of figures may take a few seconds, please be patient...", show_time=False):
        try:
            # ▶️ Cluster Plot figure
            st.subheader("Cluster Plot")
            available_cats = [col for col in display_df.columns if display_df[col].dtype == object]
            col1, col2 = st.columns([1, 1])
            with col1:
                # Select 2 axes
                selected_cats = st.multiselect(
                    "Select 2 categorical axes (x and y)",
                    options=available_cats,
                    max_selections=2,
                    default=['type of communication', 'transfer of information']
                    )

                # Select color category with default "paradigm" if available
                filtered_cats = [c for c in available_cats if c not in selected_cats]
                color_default_index = filtered_cats.index("paradigm") if "paradigm" in filtered_cats else 0
                color_cat = st.selectbox(
                    "Select category to color points",
                    options=[c for c in available_cats if c not in selected_cats],
                    index=color_default_index
                    )

                generate_plot = st.button("🎨 Generate Plot")

                if generate_plot and len(selected_cats) == 2 and color_cat:
                    fig = generate_2d_cluster_plot(display_df, selected_cats[0], selected_cats[1], color_cat)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if generate_plot and len(selected_cats) == 2 and color_cat:
                    csv_data = display_df[selected_cats + [color_cat]].dropna().copy()
                    st.download_button(
                        label="📥 Download Plot Data (CSV)",
                        data=csv_data.to_csv(index=True),
                        file_name="2d_category_plot_data.csv",
                        mime="text/csv"
                        )

        except Exception as e:
            st.error(f"❌ Could not generate figures: {e}")

with test_tab:
    create_tab_header(df, display_df)
    try:
        st.subheader(
            "Test Space (under development)"
            )  # 2. Sunburst or Treemap Plots  # Show hierarchical relationships or proportions between categories and subcategories for a more intuitive overview.

        # 3. Category Evolution Over Time  # Plot how the frequency of each category label changes by year to identify emerging or declining trends.

        # 4. Category Diversity Metrics  # Calculate and visualize diversity indices (e.g., Shannon entropy) for each category to quantify heterogeneity.


    except Exception as e:
        st.error(f"❌ Could not generate funnel plot: {e}")
