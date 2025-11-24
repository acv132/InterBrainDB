# ========================
# 📦 Imports & Setup
# ========================
import ast
import io

import pandas as pd
import plotly.io as pio
import streamlit as st
import yaml

try:
    import altair as alt

    alt.data_transformers.disable_max_rows()  # prevent silent failures on larger data
except Exception:
    pass
try:
    import plotly.graph_objects as go
except Exception as _:
    go = None

from src.utils.config import file, data_dir
from src.utils.data_loader import (load_database, create_article_handle, generate_bibtex_content,
                                   generate_apa7_latex_table, normalize_cell, generate_excel_table, flatten_cell,
                                   create_tab_header, generate_bibtexid, generate_csv_table, custom_column_picker)
from src.utils.app_utils import footer, set_mypage_config
from src.plotting.figures import (generate_interaction_figure, generate_category_counts_figure,
                                  plot_publications_over_time, generate_heatmap_figure, generate_alluvial_plot)
from src.plotting.plot_utils import export_all_category_counts, _slug

# ========================
# 💅 UI Configuration
# ========================
set_mypage_config()

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
st.markdown(
    """
    <style>
    /* Increase tab label text size */
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.2rem;      /* increase number to make bigger */
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)
data_overview_tab, data_plots_tab = st.tabs(
    ["📋 Data Overview", "📈 Plots", ]
    )

# ========================
# 📥 Load & Prepare Data
# ========================
df = load_database(data_dir, file)

df = generate_bibtexid(df)

# Create display-ready dataframe
if "exclusion_reasons" in df.columns:
    df = df.drop(
        columns=['exclusion_reasons']
        )
display_df = df.copy()

# Apply flattening to the entire DataFrame
display_df = display_df.map(flatten_cell)

display_df["article"] = df.apply(create_article_handle, axis=1)
# sort display df alphabetically by article, then by year descending
display_df = display_df.sort_values(by=["article", "year"], ascending=[True, False])
display_df["DOI Link"] = "https://doi.org/" + df["doi"]
display_df["sample size"] = df.apply(lambda row: f"N = {row['sample size']}", axis=1)
display_df.set_index("article", inplace=True)
# Move DOI Link column to first position in df
cols = ["DOI Link"] + [c for c in display_df.columns if c != "DOI Link"]
display_df = display_df[cols]

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

        st.markdown(f"**{category.capitalize()}**", help=tooltip_text)
        col1, col2 = st.columns([1, 4], vertical_alignment="top", gap="small")
        with col1:
            if st.button("", icon=":material/checklist_rtl:", key=f"{category}_selectall"):
                st.session_state[multi_key] = all_labels

        with col2:
            defaults = st.session_state.get(multi_key, [])
            defaults = [val for val in defaults if val in all_labels]

            selected = st.multiselect(
                label=f"{multi_key}",
                options=all_labels,
                default=defaults,
                key=multi_key,
                placeholder="Choose a label",
                label_visibility="collapsed"
                )
        selected_filters[category] = selected

    # --- Article Filter ---
    st.markdown("**Select Individual Articles**")
    article_options = display_df.index.tolist()
    article_multiselect_key = "article_select"
    col1, col2 = st.columns([1, 4], vertical_alignment="top", gap="small")
    with col1:
        if st.button("", icon=":material/checklist_rtl:", key=f"article_selectall"):
            st.session_state[article_multiselect_key] = article_options

    with col2:
        # Set default to empty unless session state holds selected articles
        article_defaults = st.session_state.get(article_multiselect_key, [])
        article_defaults = [a for a in article_defaults if a in article_options]

        selected_articles = st.multiselect(
            "Choose individual articles",
            options=article_options,
            default=article_defaults,
            key=article_multiselect_key,
            label_visibility="collapsed",
            placeholder="Choose an article", )

# ========================
# 🔍 Apply Filters
# ========================
# Filter by year range
filtered_df = display_df[(display_df["year"] >= selected_years[0]) & (display_df["year"] <= selected_years[1])].copy()

# Filter by inclusion in paper review
if include_only:
    filtered_df = filtered_df[filtered_df['included in paper review'] == True]

# Keyword filter: keep only rows where any column contains the keyword
if keyword:
    # convert all columns to string, do a case‐insensitive contains, and any() across columns
    mask = filtered_df.astype(str).apply(
        lambda row: row.str.contains(keyword, case=False, na=False).any(), axis=1
        )
    filtered_df = filtered_df[mask]
# Filter by article selection
if selected_articles:
    filtered_df = filtered_df[filtered_df.index.isin(selected_articles)]

# Apply category filters
for category, selected_labels in selected_filters.items():
    if category not in filtered_df.columns or not selected_labels:
        continue

    # OR within category: match if any label in the cell matches any selected label
    filtered_df = filtered_df[filtered_df[category].apply(
        lambda x: any(tag in normalize_cell(x) for tag in selected_labels)
        )]

if filtered_df.empty:
    st.warning("No studies match the selected filters. Please adjust your filter settings.")

# Update global display_df
display_df = filtered_df

# ========================
# 📄 Data Overview Tab
# ========================
with data_overview_tab:
    create_tab_header(df, display_df)
    st.markdown("This table provides an overview of the studies included in the analysis.")

    # # Column view selector
    view_option = st.radio(
        "Select view mode:",
        options=["Default", "Participants", "Paradigm", "Measurement & Analysis", "All Columns", "Custom"],
        horizontal=True, )

    # Available columns (note: 'article' is your index)
    available_cols = list(display_df.columns)

    view_configs = {
        "Default": ['article', 'DOI Link', 'included in paper review', 'measurement modality', 'sample size',
                    'pairing configuration', 'paradigm', 'cognitive function'],
        "Participants": ['article', 'DOI Link', 'sample size', 'sample', 'pairing configuration', 'pairing setup',
                         'relationship pair'],
        "Paradigm": ['article', 'DOI Link', 'condition design', 'interaction scenario', 'interaction medium',
                     'transfer of information', 'type of communication', 'paradigm', 'task symmetry'],
        "Measurement & Analysis": ['article', 'DOI Link', 'measurement modality', 'analysis method',
                                   'cognitive function'],
        "All Columns": available_cols,
        }
    # Custom column picker (preserve selection in session_state)
    if view_option == "Custom":
        column_order = custom_column_picker(available_cols, default_select=["DOI Link", 'other labels'])
    else:
        column_order = view_configs.get(view_option, view_configs["Default"])

    column_order = [c for c in column_order if c in display_df.columns]

    # Configure column help tooltips
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

    # Show filtered dataframe
    st.dataframe(display_df, column_config=column_config, column_order=column_order, hide_index=False)

    # Create export dataframe
    export_columns = column_order + ["BibTexID"] if "BibTexID" in display_df.columns else column_order
    export_df = display_df.reset_index()[export_columns].copy()

    # Export buttons
    bibtex_content = generate_bibtex_content(export_df)
    st.download_button(
        "📥 Download BibTeX", data=bibtex_content, file_name="database.bib", mime="application/x-bibtex"
        )
    latex_table = generate_apa7_latex_table(export_df)
    st.download_button(
        "📥 Download APA7-Style LaTeX Table", data=latex_table, file_name="database.tex", mime="text/plain"
        )
    csv_table = generate_csv_table(export_df)
    st.download_button(
        "📥 Download CSV", data=csv_table, file_name="database.csv", mime="text/csv", )
    excel_table = generate_excel_table(export_df)
    st.download_button(
        "📥 Download Excel Table",
        data=excel_table,
        file_name="database.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ========================
# 📈 Data Plots Tab
# ========================
st.markdown(
    """
    <style>
    /* Spinner container styling */
    .stSpinner {
        border: 2px solid #ffa040 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        background-color: rgba(255, 160, 64, 0.07);
    }

    /* Spinner icon color */
    .stSpinner > div {
        border-top-color: #ffa040 !important;
        border-left-color: #ffa040 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with data_plots_tab:
    create_tab_header(df, display_df)
    with st.spinner("The generation of figures may take a few seconds, please be patient...", show_time=False):
        try:
            # > Publication Year figure
            st.subheader("Publications over Time")
            col1, col2 = st.columns([1, 1], gap="medium")
            with col1:
                # Select category for line splitting
                selected_category = None
                category_options = list(label_tooltips.keys())
                selected_category = st.selectbox(
                    "Choose a category to display as stacked bars:", options=['None'] + category_options, )
                # Group by year and selected category, count publications
                comps = plot_publications_over_time(
                    display_df,
                    selected_category,
                    label_tooltips=label_tooltips,
                    container=st,
                    count_mode="auto",
                    return_components=True if selected_category not in [None, 'None'] else False
                    )
                if comps and comps["chart"] is not None:
                    st.altair_chart(comps["chart"], use_container_width=True)
            with col2:
                if selected_category not in (None, "None") and comps and comps["legend_chart"] is not None:
                    st.altair_chart(comps["legend_chart"], use_container_width=True)
                st.markdown(
                    """
                    💡 **Tip:** To save the figure,
                    simply **right-click** on the chart and choose **"Save image as..."**
                    *(wording may vary slightly depending on your browser)*
                    """
                    )
        except Exception as e:
            st.error(f"❌ Could not generate figures: {e}")

        try:
            # > Category counts
            st.subheader("Category Counts")
            fig2 = generate_category_counts_figure(display_df, data_plots_tab)
            counts_df = export_all_category_counts(display_df)
            st.download_button(
                "📥 Download Category Counts (CSV)",
                data=counts_df.to_csv(index=True),
                file_name="category_counts.csv",
                mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Could not generate figures: {e}")

        try:
            # > Interaction figure
            col3, col4 = st.columns([1, 1])
            with col3:
                st.subheader("Interaction Conditions")
                combine_modalities = st.checkbox(
                    "Combine across modalities",
                    value=False,
                    help=("When enabled, cells in the figure do not distinguish modalities.")
                    )

                result = generate_interaction_figure(display_df, data_plots_tab, combine_modalities=combine_modalities)
                if result is None:
                    st.warning("No figure was generated for interaction conditions.")
                else:
                    fig1, condition_count, number_studies, connection_df, summary_stats = result

                    # --- PNG render ---
                    png_buf = io.BytesIO()
                    fig1.savefig(png_buf, format="png", bbox_inches="tight", transparent=True, dpi=300)
                    png_buf.seek(0)
                    # --- SVG export ---
                    svg_buf = io.BytesIO()
                    fig1.savefig(svg_buf, format="svg", bbox_inches="tight", transparent=True)
                    svg_buf.seek(0)

                    st.image(png_buf, use_container_width=True)

                    st.markdown(
                        f"""
                    *Note*. The cross-sectional distribution of all hyperscanning conditions of 
                    {number_studies} studies across interaction medium and interaction scenario axes. The 
                    numbers provide the counted occurrences (n = {condition_count} conditions across modalities) of the 
                    combination of an interaction medium and scenario. The colors represent the measurement 
                    modalities reported for a cross-section of conditions. The connection lines indicate reported 
                    cross-condition occurrences separated per axis (n = {summary_stats['cross-condition occurrences']} 
                    simultaneous condition occurrences). Studies involving a digital component either through a digital 
                    medium or virtual interaction scenario are marked through a gray shaded area. 
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
                    "📥 Download Interaction Figure (PNG)",
                    data=png_buf.getvalue(),
                    file_name="interaction_figure.png",
                    mime="image/png"
                    )

                st.download_button(
                    "📥 Download Interaction Figure (SVG)",
                    data=svg_buf.getvalue(),
                    file_name="interaction_figure.svg",
                    mime="image/svg+xml"
                    )

                with st.expander("Show dataframe of connection lines"):
                    st.dataframe(connection_df[["modality", "condition1", "condition2", "count"]], hide_index=True)
                with st.expander("Summary of interaction-condition comparisons"):
                    summary_df = pd.DataFrame([summary_stats])
                    st.dataframe(summary_df, hide_index=True)

        except Exception as e:
            st.error(f"❌ Could not generate figures.")

        try:
            st.subheader("Alluvial Plot")

            if go is None:
                st.error("Plotly is required for this chart but could not be imported.")
            else:
                # ---------- selection UI (uses your custom picker) ----------
                available_parset_cols = [c for c in label_tooltips.keys() if c in display_df.columns]

                st.markdown("It is recommended to choose 2–5 categories to visualize the flow.")
                # Use your app’s custom picker so selection is preserved & ordered:
                chosen_parset_cols = custom_column_picker(
                    available_parset_cols,
                    custom_key="def_parset_cols",
                    default_select=['measurement modality', 'analysis method', 'cognitive function'], )

                if len(chosen_parset_cols) < 2:
                    st.info("Select at least two categories to draw the flow.")
                else:
                    # ---------- 1) value_mode selection ----------
                    value_mode_label = st.selectbox(
                        "Scale link thickness by",
                        options=["Raw counts (studies)", "Share of source node (%)", ],
                        index=0,
                        help=("**Raw counts (studies):** Link width reflects the absolute number of studies "
                              "showing a specific transition between categories. This is the most direct measure "
                              "of how often each combination appears in the dataset. "

                              "**Share of source node (%):** Each node’s outgoing links are normalized to 100%. "
                              "Link width reflects the *percentage* of studies that move from a given source "
                              "category to each next category. This highlights how each category distributes its "
                              "studies, independent of how common that category is overall."), )
                    value_mode_map = {
                        "Raw counts (studies)": "count", "Share of source node (%)": "source_frac",
                        }
                    value_mode = value_mode_map[value_mode_label]

                    # ---------- 2) which dimensions to combine multi-labels for ----------
                    combine_default = ['measurement modality'] if 'measurement modality' in chosen_parset_cols else []
                    combine_multilabel_cols = st.multiselect(
                        "Collapse multi-label combinations into single nodes for these dimensions "
                        "(e.g., 'EEG + eye-tracking').",
                        options=chosen_parset_cols,
                        default=combine_default,
                        help=("If a study has multiple labels in these columns, they will be combined into a single "
                              "category (e.g., 'EEG + eye-tracking'), making multimodal designs explicit."), )

                    # ---------- 3) color_by selection (before probe!) ----------
                    def_color_by = 'measurement modality'
                    color_by = st.selectbox(
                        "Color links by (optional)",
                        options=['(none)'] + available_parset_cols,
                        index=(['(none)'] + available_parset_cols).index(def_color_by),
                        help="Pick one category to color the link flows. Leave as '(none)' to color by source stage.", )
                    color_by = None if color_by == '(none)' else color_by

                    # ---------- 4) PROBE call to get max_count with SAME grouping logic ----------
                    _fig_probe, _link_count_probe, _node_count_probe, _links_df_probe, diag_md_probe = generate_alluvial_plot(
                        display_df=display_df, chosen_parset_cols=chosen_parset_cols, min_count=1,
                        # no filtering → true raw counts for this grouping
                        color_by=color_by,  # SAME color_by as real plot
                        value_mode="count",  # slider is based on raw counts, not fractions
                        combine_multilabel_cols=combine_multilabel_cols,  # SAME combination logic
                        )

                    if _links_df_probe.empty:
                        # No links at all for this configuration; fall back gracefully
                        max_count = 1
                    else:
                        max_count = int(_links_df_probe["count"].max())

                    # ---------- 5) Slider based on probe counts ----------
                    min_count = st.slider(
                        "Minimum link count (expanded label combinations)",
                        min_value=1,
                        max_value=max_count,
                        value=min(1, max_count),
                        help=("Hide links that are supported by only a few expanded label combinations. "
                              "Counts are based on the current color and node grouping: each study with "
                              "multiple labels can contribute to multiple combinations."),
                        key="parset_min_link_count", )

                    # ---------- 6) REAL call using user-selected min_count & value_mode ----------
                    fig, link_count, node_count, links_df, diag_md = generate_alluvial_plot(
                        display_df=display_df,
                        chosen_parset_cols=chosen_parset_cols,
                        min_count=min_count,
                        color_by=color_by,
                        value_mode=value_mode,
                        combine_multilabel_cols=combine_multilabel_cols, )

                    sankey_key = "sankey_" + _slug(
                        "_".join(chosen_parset_cols)
                        ) + f"_{min_count}_{link_count}_{color_by or 'stage'}_{value_mode}"

                    st.plotly_chart(fig, use_container_width=True, key=sankey_key, render_mode="svg")

                    # --------- Caption & description ----------
                    if value_mode == "count":
                        thickness_desc = "Link thickness = expanded-combination count (see tooltip for details)."
                    else:  # "source_frac"
                        thickness_desc = ("Link thickness = proportion of all outgoing links from the source node, "
                                          "within the currently displayed links.")

                    st.caption(thickness_desc)

                    color_by_desc = f" Colors indicate {color_by}." if color_by else " Colors indicate the source stage."
                    combine_desc = ""
                    if combine_multilabel_cols:
                        combine_desc = (
                            " Multi-label studies in the following dimensions are collapsed into explicit combination "
                            f"nodes (e.g., 'EEG + eye-tracking'): {', '.join(combine_multilabel_cols)}.")

                    st.markdown(
                        f"Alluvial (Sankey) graph of hyperscanning studies across the categorical dimensions "
                        f"{chosen_parset_cols}. Nodes represent the number of studies assigned to each category "
                        f"(including multimodal combinations where specified), and link thickness reflects transitions "
                        f"between adjacent categories according to the selected scaling scheme. "
                        f"{color_by_desc}{combine_desc}"
                        )

                    # --- Build an HTML page with a "Download SVG" button ---
                    try:
                        fig_html = pio.to_html(
                            fig, include_plotlyjs="cdn", full_html=False, div_id="alluvial-fig", )
                        html_page = f"""
                        <!DOCTYPE html>
                        <html lang="en">
                        <head>
                            <meta charset="utf-8">
                            <title>Alluvial Plot</title>
                        </head>
                        <body style="font-family: sans-serif;">

                            <h2>Alluvial Plot</h2>

                            {fig_html}

                            <div style="margin-top: 20px;">
                                <button id="download-svg-btn" style="padding: 8px 16px; margin-right: 10px;">
                                    Download SVG
                                </button>

                                <button id="download-png-btn" style="padding: 8px 16px;">
                                    Download PNG
                                </button>
                            </div>

                            <script>
                            // SVG Export
                            document.getElementById('download-svg-btn').addEventListener('click', function() {{
                                var gd = document.getElementById('alluvial-fig');
                                Plotly.downloadImage(gd, {{
                                    format: 'svg',
                                    filename: 'alluvial_plot',
                                    height: 563,
                                    width: 1000,
                                    scale: 1
                                }});
                            }});

                            // PNG Export
                            document.getElementById('download-png-btn').addEventListener('click', function() {{
                                var gd = document.getElementById('alluvial-fig');
                                Plotly.downloadImage(gd, {{
                                    format: 'png',
                                    filename: 'alluvial_plot',
                                    height: 563,
                                    width: 1000,
                                    scale: 1
                                }});
                            }});
                            </script>

                        </body>
                        </html>
                        """
                        st.download_button(
                            label="📥 Download Interactive HTML (with SVG export)",
                            data=html_page.encode("utf-8"),
                            file_name="alluvial_plot.html",
                            mime="text/html",
                            key="alluvial_html_download", )

                    except Exception as e:
                        st.warning(
                            "Could not create HTML download for the alluvial plot. "
                            f"Technical details: {e}"
                            )

        except Exception as e:
            st.error(f"❌ Could not generate figure: {e}")

        try:
            st.subheader("Heatmap of Categories")
            generate_heatmap_figure(display_df, data_plots_tab)
        except Exception as e:
            st.error(f"❌ Could not generate figure: {e}")

footer()
