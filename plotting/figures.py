import ast
from collections import defaultdict
from itertools import combinations
from threading import RLock
import yaml
import pandas as pd

try:
    import altair as alt

    alt.data_transformers.disable_max_rows()  # prevent silent failures on larger data
except Exception:
    pass

import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from plotly import express as px
from sklearn.preprocessing import LabelEncoder

from utils.config import ColorMap
from plotting.plot_utils import *

_lock = RLock()


def generate_category_counts_figure(df, tab):
    """
    Generate multiple horizontal bar charts using Streamlit's st.bar_chart.
    Each chart shows counts of one-hot encoded labels within a category.
    """

    if df is None or df.empty:
        tab.warning("⚠️ No data found; make sure that at least some data is passing the selected filters.")
        return

    # Theme-aware color logic (for future use or consistency)
    figure_background_color = st.get_option('theme.backgroundColor')
    font_color = "#ffffff" if is_dark_color(figure_background_color) else "#000000"

    # Decode one-hot columns if needed
    prefix = "_"
    df = decode_one_hot(df, prefix)

    # Load categories
    with open("./data/info_yamls/categories.yaml", 'r', encoding='utf-8') as f:
        category_definitions = yaml.safe_load(f)
    base_categories = list(category_definitions.keys())

    # Identify one-hot columns per category
    all_columns = df.columns
    category_columns = {cat: [col for col in all_columns if col.startswith(f"{cat}{prefix}")] for cat in
                        base_categories}
    category_columns = {k: v for k, v in category_columns.items() if v}

    n_categories = len(category_columns)
    # User layout choice
    layout_option = tab.radio(
        "Choose plot layout:", ["Auto", "Single Column", "Grid (3 columns)"], horizontal=True
        )

    if layout_option == "Single Column":
        n_cols = 1
    elif layout_option == "Grid (3 columns)":
        n_cols = 3
    else:  # Auto layout
        n_cols = min(4, n_categories)
    n_rows = int(np.ceil(-(-n_categories / n_cols)))

    # Render bar charts
    cat_list = list(category_columns.items())
    for row_idx in range(n_rows):
        cols = tab.columns(n_cols)
        for col_idx in range(n_cols):
            i = row_idx * n_cols + col_idx
            if i >= len(cat_list):
                break
            cat, columns = cat_list[i]

            # Count each label in the one-hot group
            counts = df[columns].sum().sort_values(ascending=False)
            counts.index = [col.replace(f"{cat}{prefix}", "") for col in counts.index]

            with cols[col_idx]:
                if (counts > 0).sum() > 1:
                    st.markdown(f"**{cat}**")
                    # st.bar_chart(counts, horizontal=True, color=st.get_option('theme.primaryColor'))
                    import altair as alt
                    chart_df = pd.DataFrame(
                        {
                            "label": counts.index, "count": counts.values
                            }
                        )
                    bar_chart = alt.Chart(chart_df).mark_bar(color=st.get_option('theme.primaryColor')).encode(
                        x=alt.X("count:Q", title="Count"), y=alt.Y("label:N", sort="-x", title=""),
                        # Sort by count descending
                        tooltip=["label", "count"]
                        ).properties(
                        height=300, padding={"left": 20, "right": 20, "top": 20, "bottom": 20}, ).configure_view(
                        stroke=None
                        ).configure_axis(
                        labelFontSize=12, labelLimit=150,  # distance in pixels between axis‐line and tick labels
                        labelPadding=10,  # distance in pixels between axis‐line and title
                        titlePadding=15, )

                    st.altair_chart(bar_chart, use_container_width=True)
                else:
                    # Print the label and its count if only one label is present and associated count
                    st.markdown(f"**{cat}**")
                    label = counts.index[0]
                    value = counts.iloc[0]
                    st.write(f"Only label is {label}: {value} counts")

    return


def generate_interaction_figure(df, tab):
    """Generate a confusion matrix-like figure showing the interaction conditions across studies."""

    # Check if DataFrame has any rows
    if len(df) == 0:
        tab.warning("⚠️ No data found; make sure that at least some data is passing the selected filters.")
        return None

    ###############
    # Configs
    ###############
    # get defined category labels for CM and color legend
    yaml_file = "./data/info_yamls/categories.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        categories = yaml.safe_load(f)

    default_scenario_order = list(categories["interaction scenario"].keys())
    default_manipulation_order = list(categories["interaction manipulation"].keys())
    default_modalities_order = list(categories["measurement modality"].keys())

    # define some figure specs
    row_spacing = 1.5  # Increase spacing between rows
    col_spacing = 1.5  # Increase spacing between columns
    # Base width for line thickness
    base_width = 1.5
    base_curvature = .4
    line_styles = ['-',  # solic
                   '--',  # dashed
                   ':',  # dotted
                   '-.',  # dash-dot
                   (0, (1, 1)),  # densely dotted
                   (0, (5, 1)),  # loosely dashed
                   (0, (3, 5, 1, 5)),  # dash-dot-dot
                   (0, (5, 5)),  # loosely dash-dot
                   (0, (3, 1, 1, 1)),  # dash-dotted pattern
                   (0, (5, 2, 1, 2)),  # dash-dot with more spacing
                   (0, (2, 2)),  # dashed, dense
                   (0, (1, 10)),  # very sparse dotted
                   (0, (1, 5)),  # medium sparse dotted
                   (0, (3, 1, 1, 1, 1, 1)),  # custom: dash-dot-dot-dot
                   ]
    # Adjust radius for connections to start/end at pie edges
    pie_radius = 0.5
    colormap = ColorMap
    modality_colors = dict(zip(default_modalities_order, colormap))
    figure_background_color = st.get_option('theme.backgroundColor')
    color_rect = "#43454a" if is_dark_color(figure_background_color) else "#e9e9e9"
    font_color = "#ffffff" if is_dark_color(figure_background_color) else "#000000"
    # Legend configs
    legend_title_props = FontProperties(size=16)
    facecolor_legend = None
    y_axis_legend_start = .7

    ###############
    # prepare data
    ###############
    prefix = "_"
    df_orig = df.copy()
    df = decode_one_hot(df, prefix)
    number_studies = len(df)

    scenario_columns = [col for col in df.columns if col.startswith(f"interaction scenario{prefix}")]
    manipulation_columns = [col for col in df.columns if col.startswith(f"interaction manipulation{prefix}")]
    modality_columns = [col for col in df.columns if col.startswith(f"measurement modality{prefix}")]
    df['interaction scenario'] = df[scenario_columns].apply(
        lambda row: [col.replace(f"interaction scenario{prefix}", "") for col, val in row.items() if val], axis=1
        )
    df['interaction manipulation'] = df[manipulation_columns].apply(
        lambda row: [col.replace(f"interaction manipulation{prefix}", "") for col, val in row.items() if val], axis=1
        )
    df['measurement modality'] = df[modality_columns].apply(
        lambda row: [col.replace(f"measurement modality{prefix}", "") for col, val in row.items() if val], axis=1
        )

    ####################
    # Count conditions #
    ####################
    condition_rows = []
    for doi, group in df_orig.groupby('doi'):
        # Use original, pre-exploded, pre-onehot DataFrame for correct pairing!
        orig_row = df_orig[df_orig['doi'] == doi].iloc[0]

        modalities = ensure_list(orig_row['measurement modality'])
        scenarios = ensure_list(orig_row['interaction scenario'])
        manipulations = ensure_list(orig_row['interaction manipulation'])

        for modality in modalities:
            if len(scenarios) == len(manipulations) and len(scenarios) > 1:
                for s, m in zip(scenarios, manipulations):
                    condition_rows.append([modality, s, m, doi])
            elif len(scenarios) == 1:
                for m in manipulations:
                    condition_rows.append([modality, scenarios[0], m, doi])
            elif len(manipulations) == 1:
                for s in scenarios:
                    condition_rows.append([modality, s, manipulations[0], doi])
            else:
                # cartesian
                for s in scenarios:
                    for m in manipulations:
                        condition_rows.append([modality, s, m, doi])

    condition_df = pd.DataFrame(
        condition_rows, columns=["measurement modality", "interaction scenario", "interaction manipulation", "doi"]
        )
    # Now count
    cross_section_counts = condition_df.groupby(
        ['measurement modality', 'interaction scenario', 'interaction manipulation']
        ).size().reset_index(name='count')

    # For later plotting
    scenario_contained = [col.replace("interaction scenario_", "") for col in scenario_columns]
    manipulation_contained = [col.replace("interaction manipulation_", "") for col in manipulation_columns]
    modalities_contained = [col.replace("measurement modality_", "") for col in modality_columns]

    # for plotting
    scenario_order = [scenario for scenario in default_scenario_order if scenario in scenario_contained]
    manipulation_order = [manipulation for manipulation in default_manipulation_order if
                          manipulation in manipulation_contained]

    # prepare confusion matrix rows and columns
    row_pos = np.arange(len(scenario_order)) * row_spacing
    col_pos = np.arange(len(manipulation_order)) * col_spacing

    ####################
    # Count connection #
    ####################
    connection_counts = defaultdict(int)
    connection_display_orders = defaultdict(list)
    pairing_warnings = []

    for doi, group in df_orig.groupby('doi'):
        modalities = [mod for mod in modalities_contained if group[f"measurement modality_{mod}"].iloc[0] == 1]

        # Fetch scenarios and manipulations directly from df_orig using DOI
        orig_row = df_orig[df_orig['doi'] == doi].iloc[0]

        scenarios = ensure_list(orig_row['interaction scenario'])
        manipulations = ensure_list(orig_row['interaction manipulation'])

        for modality in modalities:
            pairs = []
            if len(scenarios) == len(manipulations) and len(scenarios) > 1:
                for s, m in zip(scenarios, manipulations):
                    y = row_pos[scenario_order.index(s)]
                    x = col_pos[manipulation_order.index(m)]
                    pairs.append((x, y, s, m))
            elif len(scenarios) == 1:
                for m in manipulations:
                    y = row_pos[scenario_order.index(scenarios[0])]
                    x = col_pos[manipulation_order.index(m)]
                    pairs.append((x, y, scenarios[0], m))
            elif len(manipulations) == 1:
                for s in scenarios:
                    y = row_pos[scenario_order.index(s)]
                    x = col_pos[manipulation_order.index(manipulations[0])]
                    pairs.append((x, y, s, manipulations[0]))
            else:
                for s in scenarios:
                    for m in manipulations:
                        y = row_pos[scenario_order.index(s)]
                        x = col_pos[manipulation_order.index(m)]
                        pairs.append((x, y, s, m))
                pairing_warnings.append((doi, modality, scenarios, manipulations))

            if len(pairs) == 2:
                pt1, pt2 = pairs
                endpoints = ((pt1[0], pt1[1]), (pt2[0], pt2[1]))
                sorted_endpoints = tuple(sorted([endpoints[0], endpoints[1]]))
                connection_counts[(sorted_endpoints, modality)] += 1
                connection_display_orders[(sorted_endpoints, modality)].append((pt1, pt2))
            elif len(pairs) > 2:
                for a, b in combinations(pairs, 2):
                    endpoints = ((a[0], a[1]), (b[0], b[1]))
                    sorted_endpoints = tuple(sorted([endpoints[0], endpoints[1]]))
                    connection_counts[(sorted_endpoints, modality)] += 1
                    connection_display_orders[(sorted_endpoints, modality)].append(
                        (a, b)
                        )

    connection_data = []
    for (sorted_endpoints, modality), count in connection_counts.items():
        display_orders = connection_display_orders[(sorted_endpoints, modality)]
        pt1, pt2 = display_orders[0]
        connection_data.append(
            ((pt1[0], pt1[1]), (pt2[0], pt2[1]), f"{pt1[2]} - {pt1[3]}", f"{pt2[2]} - {pt2[3]}", modality, count)
            )

    connection_df = pd.DataFrame(
        connection_data, columns=["start", "end", "condition1", "condition2", "modality", "count"]
        )

    ###############
    # Prepare the plot
    ###############
    unique_counts = sorted(connection_df['count'].unique())
    count_to_style = {count: line_styles[i % len(line_styles)] for i, count in enumerate(unique_counts)}
    with _lock:
        fig, ax = plt.subplots(figsize=(16, 12))

        # draw connection lines
        for idx, row in connection_df.iterrows():
            start, end = row['start'], row['end']
            modality = row['modality']
            count = row['count']

            # Get color and line width
            color = modality_colors.get(modality, 'black')
            line_width = base_width
            line_style = count_to_style[count]
            curvature = (list(modality_colors.keys()).index(row['modality']) + 1) * base_curvature

            start = np.array(start)
            end = np.array(end)

            # Draw curved lines using Bezier path
            if start[0] == end[0]:  # vertical
                control_point = ((start[0] + end[0]) / 2 + curvature, (start[1] + end[1]) / 2)
            elif start[1] == end[1]:  # horizontal
                control_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + curvature)
            else:  # diagonal
                # Compute the perpendicular direction to the line
                delta = end - start
                # Perpendicular vector
                perp = np.array([-delta[1], delta[0]])
                if np.linalg.norm(perp) != 0:
                    perp = perp / np.linalg.norm(perp)
                # Midpoint
                mid = (start + end) / 2
                # Offset control point along perpendicular, scaled by curvature
                control_point = mid + perp * curvature
                control_point = tuple(control_point)

            vertices = np.array([start, control_point, end])
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            bezier_path = Path(vertices, codes)
            patch = PathPatch(bezier_path, color=color, lw=line_width, fill=False, linestyle=line_style, zorder=1)
            ax.add_patch(patch)

        # Draw pie charts for each scenario and manipulation combination
        for i, scenario in enumerate(scenario_order):
            for j, manipulation in enumerate(manipulation_order):
                # Filter cross_section_counts for this scenario-manipulation cell
                cell_counts = cross_section_counts[(cross_section_counts['interaction scenario'] == scenario) & (
                        cross_section_counts['interaction manipulation'] == manipulation)]
                modality_counts = dict(zip(cell_counts['measurement modality'], cell_counts['count']))

                if modality_counts:  # Only draw pie if something present
                    pie_sizes = list(modality_counts.values())
                    pie_colors = [modality_colors[mod] for mod in modality_counts.keys()]

                    # Scale pie size (optional, as before)
                    size_factor = sum(pie_sizes) / cross_section_counts['count'].sum()  # relative to all data
                    radius = pie_radius + size_factor

                    ax.pie(
                        pie_sizes,
                        colors=pie_colors,
                        center=(col_pos[j], row_pos[i]),
                        radius=radius,
                        wedgeprops=dict(width=0.3), )
                    ax.text(
                        col_pos[j],
                        row_pos[i],
                        str(sum(pie_sizes)),
                        color='k',
                        fontsize=12,
                        ha='center',
                        va='center',
                        bbox=dict(
                            boxstyle="circle", facecolor="white", edgecolor="none", pad=radius + 0.3, )
                        )
        # Draw gray area in plot to highlight virtual row and digital IM columns
        plt.xlim(left=-1, right=12)
        plt.ylim(bottom=-1, top=12)
        virtual_row_index = scenario_order.index("virtual")

        digital_im_column_indices = [manipulation_order.index(col) for col in manipulation_order if "digital" in col]

        virtual_y_start = row_pos[virtual_row_index] - row_spacing / 2
        virtual_y_end = row_pos[virtual_row_index] + row_spacing / 2
        virtual_x_start = col_pos[0] - col_spacing / 2
        virtual_x_end = col_pos[-1] + col_spacing / 2
        ax.add_patch(
            Rectangle(
                (virtual_x_start, virtual_y_start),
                virtual_x_end - virtual_x_start,
                virtual_y_end - virtual_y_start,
                color=color_rect,
                zorder=0,
                alpha=1, )
            )

        digital_x_start = col_pos[digital_im_column_indices[0]] - col_spacing / 2
        digital_x_end = col_pos[digital_im_column_indices[-1]] + col_spacing / 2
        digital_y_start = row_pos[0] - row_spacing / 2
        digital_y_end = row_pos[-1] + row_spacing / 2
        ax.add_patch(
            Rectangle(
                (digital_x_start, digital_y_start),
                digital_x_end - digital_x_start,
                digital_y_end - digital_y_start,
                color=color_rect,
                zorder=0,
                alpha=1, )
            )

        # Format the plot
        # Add gridlines for clarity
        ax.set_xticks(col_pos)
        ax.set_yticks(row_pos)

        # Increase padding for tick labels
        ax.set_xticklabels(
            [c.replace(" IM", '') for c in manipulation_order], rotation=45, ha='right', fontsize=14, color=font_color
            )
        ax.tick_params(axis='x', which='both', length=0, pad=10)
        ax.set_yticklabels(scenario_order, fontsize=14, color=font_color)
        ax.tick_params(axis='y', which='both', length=0, pad=10)

        # Add labels for axes with consistent padding
        ax.set_xlabel("Interaction manipulation", fontsize=16, labelpad=10, color=font_color)
        ax.set_ylabel("Interaction Scenario", fontsize=16, labelpad=10, color=font_color)

        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        condition_count = cross_section_counts["count"].sum()
        ax.set_title(
            f"Confusion matrix of experimental interaction conditions ({condition_count} "
            f"conditions in total {number_studies} studies)", fontsize=14, pad=10, color=font_color
            )

        # Create dictionary for legend handles
        modality_handles = {label: plt.Line2D([0], [0], color=color, lw=4, linestyle='-')
                            # Default solid line for modality
                            for label, color in modality_colors.items()}

        # Get only modalities actually used
        used_modalities = connection_df['modality'].unique()
        filtered_modality_handles = {label: handle for label, handle in modality_handles.items() if
                                     label in used_modalities}

        # Get only counts actually used
        used_counts = connection_df['count'].unique()
        filtered_style_handles = {
            f"count: {count}": plt.Line2D([0], [0], color=font_color, lw=4, linestyle=count_to_style[count]) for count
            in used_counts}

        # Only add digital component if it's actually highlighted
        filtered_area_handle = {}
        if digital_im_column_indices:  # non-empty list
            filtered_area_handle = {
                'digital component': mpatches.Patch(color=color_rect, label='digital component')
                }

        legend_args = dict(
            loc="center left",
            fontsize=14,
            alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props, )

        # 1. Modality Legend (Fixed at top)
        mod_legend = ax.legend(
            handles=list(filtered_modality_handles.values()),
            labels=list(filtered_modality_handles.keys()),
            title="measurement modality",
            bbox_to_anchor=(1.0, 0.7),
            **legend_args
            )
        plt.setp(mod_legend.get_title(), color=font_color)

        # 2. Comparison Legend (middle)
        comparison_legend = Legend(
            ax,
            handles=list(filtered_style_handles.values()),
            labels=list(filtered_style_handles.keys()),
            title="comparisons",
            bbox_to_anchor=(1.0, 0.45),
            **legend_args
            )
        plt.setp(comparison_legend.get_title(), color=font_color)
        ax.add_artist(comparison_legend)

        # 3. Study Design Legend (bottom, only if used)
        if filtered_area_handle:
            study_design_legend = Legend(
                ax,
                handles=list(filtered_area_handle.values()),
                labels=list(filtered_area_handle.keys()),
                title="study design",
                bbox_to_anchor=(1.0, 0.30),
                **legend_args
                )
            plt.setp(study_design_legend.get_title(), color=font_color)
            ax.add_artist(study_design_legend)

        plt.tight_layout()
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + 1)

        connection_df.drop(columns=["start", "end"])
        col = connection_df.pop("modality")
        connection_df.insert(0, col.name, col)

        return fig, condition_count, number_studies, connection_df


def generate_2d_cluster_plot(df, x_cat, y_cat, color_cat, jitter_scale=0.15):
    df_clean = df[[x_cat, y_cat, color_cat]].dropna().copy()
    df_clean["article"] = df_clean.index

    label_maps = {}
    for col in [x_cat, y_cat]:
        le = LabelEncoder()
        df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
        label_maps[col] = dict(zip(le.transform(le.classes_), le.classes_))

    df_clean["coord_key"] = df_clean[[x_cat + "_encoded", y_cat + "_encoded"]].astype(str).agg("-".join, axis=1)

    coords = []
    for _, group in df_clean.groupby("coord_key"):
        base = group.iloc[0][[x_cat + "_encoded", y_cat + "_encoded"]].values.astype(float)
        offsets = np.random.normal(scale=jitter_scale, size=(len(group), 2))
        coords.append(base + offsets)
    coords = np.vstack(coords)

    df_clean["x"] = coords[:, 0]
    df_clean["y"] = coords[:, 1]

    fig = px.scatter(
        df_clean,
        x="x",
        y="y",
        color=color_cat,
        hover_name="article",
        hover_data={col: True for col in [x_cat, y_cat, color_cat]},
        title=f"2D Plot: {x_cat} vs {y_cat} (color: {color_cat})"
        )

    fig.update_layout(
        width=800, height=700, xaxis=dict(
            title=x_cat,
            tickmode="array",
            tickvals=list(label_maps[x_cat].keys()),
            ticktext=list(label_maps[x_cat].values())
            ), yaxis=dict(
            title=y_cat,
            tickmode="array",
            tickvals=list(label_maps[y_cat].keys()),
            ticktext=list(label_maps[y_cat].values())
            )
        )

    return fig


def plot_publications_over_time(
    df,
    selected_category,
    label_tooltips=None,
    container=None,
    count_mode="auto",          # 'auto' | 'study_weighted' | 'occurrence'
    color_map=ColorMap,         # optional list of hex/rgb colors
    year_col="year",
):
    # -- Resolve container --
    if container is None:
        try:
            import streamlit as st
            container = st
        except Exception:
            class _Null:
                def __getattr__(self, _):
                    def _noop(*a, **k): pass
                    return _noop
            container = _Null()

    # Prevent silent failures on larger data (optional)
    try:
        alt.data_transformers.disable_max_rows()
    except Exception as e:
        pass

    # -- Determine effective count mode --
    effective_mode = "study_weighted" if count_mode == "auto" else count_mode
    if effective_mode not in ["study_weighted", "occurrence"]:
        container.error(f"Selected count_mode {count_mode} is not supported.")
        return

    # -- Parse labels robustly per row --
    year_label_rows = []
    for _, row in df.iterrows():
        year = row.get(year_col, np.nan)
        if pd.isna(year):
            continue
        year = str(year)

        col_val = row.get(selected_category, np.nan)

        if isinstance(col_val, list):
            labels = col_val
        elif pd.isna(col_val):
            labels = []
        elif isinstance(col_val, str) and col_val.strip().startswith("[") and col_val.strip().endswith("]"):
            try:
                parsed = ast.literal_eval(col_val)
                labels = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                labels = [col_val]
        elif isinstance(col_val, str) and col_val.strip():
            labels = [col_val]
        else:
            labels = []

        labels = [l if l is None else str(l) for l in labels]
        labels = [l for l in labels if (l is not None) and (l != "") and (not pd.isna(l))]
        labels = list(dict.fromkeys(labels))

        if not labels:
            continue

        k = len(labels)
        w = 1.0 / k
        for label in labels:
            year_label_rows.append({"year": year, "label": label, "weight": w, "raw": 1.0})

    if not year_label_rows:
        container.warning("No data to plot for the selected category.")
        return

    long_df = pd.DataFrame(year_label_rows)
    label_tooltips = label_tooltips or {}
    long_df["label_display"] = long_df["label"].map(lambda x: label_tooltips.get(x, x))

    # Use weighted or raw values
    value_field = "weight" if effective_mode == "study_weighted" else "raw"

    # ---- Pre-aggregate in pandas (stable on rerun) ----
    agg = (
        long_df.groupby(["year", "label_display"], as_index=False)
        .agg(value=(value_field, "sum"), raw=("raw", "sum"))
    )
    agg["year"] = agg["year"].astype(str)

    # Totals per year (overlay line)
    total_df = (
        agg.groupby("year", as_index=False)
           .agg(total_studies=("value", "sum"))
           .assign(year=lambda d: d["year"].astype(str))
           .sort_values("year")
           .dropna(subset=["total_studies"])
           .reset_index(drop=True)
    )
    if total_df.empty:
        container.warning("No totals to plot.")
        return

    # Legend order by contribution
    label_order = (
        agg.groupby("label_display")["value"].sum()
           .sort_values(ascending=False).index.tolist()
    )

    # Unified, explicit x-domain for BOTH layers
    year_order = sorted(set(agg["year"]).union(set(total_df["year"])))
    x_enc = alt.X(
        "year:N",
        title="Year",
        sort=year_order,
        scale=alt.Scale(domain=year_order),
    )

    # ---- Bars (can be commented out safely) ----
    bars = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=x_enc,
            y=alt.Y(
                "value:Q",
                title="Studies (study-weighted)" if effective_mode == "study_weighted" else "Label occurrences",
            ),
            color=alt.Color(
                "label_display:N",
                title="Category",
                sort=label_order,
                scale=alt.Scale(range=color_map) if color_map else alt.Undefined,
            ),
            tooltip=[
                alt.Tooltip("year:N", title="Year"),
                alt.Tooltip("label_display:N", title="Category"),
                alt.Tooltip("value:Q",
                            title="Stack value",
                            format=".2f" if effective_mode == "study_weighted" else ".0f"),
                alt.Tooltip("raw:Q", title="Raw count", format=".0f"),
            ],
            order=alt.Order("label_display", sort="ascending"),
        )
    )

    # ---- Line ----
    line = (
        alt.Chart(total_df)
        .mark_line(point=True, strokeWidth=2.5, clip=True)
        .encode(
            x=x_enc,
            y=alt.Y("total_studies:Q", title=None),
            order=alt.Order("year:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("year:N", title="Year"),
                alt.Tooltip("total_studies:Q",
                            title="Total studies (year)",
                            format=".2f" if effective_mode == "study_weighted" else ".0f"),
            ],
        )
        .transform_filter(alt.datum.total_studies != None)
    )

    # Layer plots
    chart = (
        alt.layer(bars, line)
           .resolve_scale(x="shared", y="shared")
           .properties(height=380)
    )
    # -- Render (force fresh rerender via key to beat Streamlit caching issues) --
    key = f"pubs_{selected_category}_{effective_mode}_{len(agg)}_{len(label_order)}_{year_order[0]}_{year_order[-1]}"
    try:
        container.altair_chart(chart, use_container_width=True, key=key)
    except Exception:
        pass
