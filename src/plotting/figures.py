import itertools
from collections import defaultdict, Counter
from itertools import combinations
from threading import RLock
from typing import List, Tuple

import pandas as pd

from src.plotting.plot_utils import _norm_list, _rgba, _norm_list_safe_str

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

from src.utils.config import ColorMap
from src.plotting.plot_utils import *

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


def generate_interaction_figure(df, tab, combine_modalities=False):
    """Generate a confusion matrix-like figure showing the interaction conditions across studies.
       If combine_modalities=True, pies show a single combined slice instead of modality breakdown.
    """

    # Check if DataFrame has any rows
    if len(df) == 0:
        tab.warning("⚠️ No data found; make sure that at least some data is passing the selected filters.")
        return None

    ###############
    # Configs
    ###############
    yaml_file = "./data/info_yamls/categories.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        categories = yaml.safe_load(f)

    default_scenario_order = list(categories["interaction scenario"].keys())
    default_manipulation_order = list(categories["interaction manipulation"].keys())
    default_modalities_order = list(categories["measurement modality"].keys())

    row_spacing = 1.5
    col_spacing = 1.5
    base_width = 1.5
    base_curvature = .4
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 1, 1, 1)),
                   (0, (5, 2, 1, 2)), (0, (2, 2)), (0, (1, 10)), (0, (1, 5)), (0, (3, 1, 1, 1, 1, 1))]
    pie_radius = 0.5

    colormap = ColorMap
    modality_colors = dict(zip(default_modalities_order, colormap))

    figure_background_color = st.get_option('theme.backgroundColor')
    color_rect = "#43454a" if is_dark_color(figure_background_color) else "#e9e9e9"
    font_color = "#ffffff" if is_dark_color(figure_background_color) else "#000000"

    legend_title_props = FontProperties(size=16)
    facecolor_legend = None

    ###############
    # prepare data
    ###############
    prefix = "_"
    df_raw = df.copy()

    # detect rows to drop on the raw df
    def valid_values(x, defaults):
        values = ensure_list(x)
        return len(values) > 0 and all(v in defaults for v in values)

    non_default_rows = df_raw[
        ~df_raw['interaction scenario'].apply(lambda x: valid_values(x, default_scenario_order)) | ~df_raw[
            'interaction manipulation'].apply(lambda x: valid_values(x, default_manipulation_order)) | ~df_raw[
            'measurement modality'].apply(lambda x: valid_values(x, default_modalities_order))]

    if not non_default_rows.empty:
        tab.warning(
            f"⚠️ Some studies contain non-default entries in interaction scenario, manipulation, or modality. These studies are excluded from the figure. (DOIs: {', '.join(non_default_rows['doi'].unique())})"
            )
        df_work = df_raw.drop(non_default_rows.index)
    else:
        df_work = df_raw

    # df_orig = filtered, *unmodified* copy for grouping/metadata
    df_orig = df_work.copy()

    # df = filtered, *decoded* working copy for plotting
    df = decode_one_hot(df_work, prefix)
    number_studies = len(df)

    scenario_columns = [c for c in df.columns if c.startswith(f"interaction scenario{prefix}")]
    manipulation_columns = [c for c in df.columns if c.startswith(f"interaction manipulation{prefix}")]
    modality_columns = [c for c in df.columns if c.startswith(f"measurement modality{prefix}")]

    ####################
    # Count conditions #
    ####################
    condition_rows = []
    for doi, group in df_orig.groupby('doi'):
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
                for s in scenarios:
                    for m in manipulations:
                        condition_rows.append([modality, s, m, doi])

    condition_df = pd.DataFrame(
        condition_rows, columns=["measurement modality", "interaction scenario", "interaction manipulation", "doi"]
        )

    cross_section_counts = condition_df.groupby(
        ['measurement modality', 'interaction scenario', 'interaction manipulation']
        ).size().reset_index(name='count')

    scenario_contained = [col.replace("interaction scenario_", "") for col in scenario_columns]
    manipulation_contained = [col.replace("interaction manipulation_", "") for col in manipulation_columns]
    modalities_contained = [col.replace("measurement modality_", "") for col in modality_columns]

    scenario_order = [s for s in default_scenario_order if s in scenario_contained]
    manipulation_order = [m for m in default_manipulation_order if m in manipulation_contained]

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
                    connection_display_orders[(sorted_endpoints, modality)].append((a, b))

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
    with _lock:
        # ---- Figure with reserved legend strip ----
        axes_w_in = 16
        legend_w_in = 1
        fig_h_in = 12
        fig, ax = plt.subplots(figsize=(axes_w_in + legend_w_in, fig_h_in))

        # Shrink the axes to leave a right strip for legends
        right_frac = axes_w_in / (axes_w_in + legend_w_in)  # e.g., 16 / 21 = ~0.762
        fig.subplots_adjust(right=right_frac)

        if combine_modalities:
            # Sum counts across modalities for the same endpoint pair (order-invariant)
            def _norm_pair(row):
                a = tuple(row['start'])
                b = tuple(row['end'])
                return tuple(sorted([a, b]))

            tmp = connection_df.copy()
            tmp['pair'] = tmp.apply(_norm_pair, axis=1)

            # Aggregate counts across modalities
            agg = tmp.groupby('pair', as_index=False)['count'].sum()

            # Pick a representative orientation (first occurrence) for plotting
            rep = tmp.drop_duplicates('pair')[['pair', 'start', 'end']]

            lines_df = rep.merge(agg, on='pair', how='left')[['start', 'end', 'count']]
            combined_color = list(modality_colors.values())[0]

            # Recompute styles based on combined counts
            unique_counts = sorted(lines_df['count'].unique()) if not lines_df.empty else [1]
            count_to_style = {c: line_styles[i % len(line_styles)] for i, c in enumerate(unique_counts)}

        else:
            # Use per-modality rows as-is
            lines_df = connection_df[['start', 'end', 'modality', 'count']].copy()
            unique_counts = sorted(connection_df['count'].unique()) if not connection_df.empty else [1]
            count_to_style = {c: line_styles[i % len(line_styles)] for i, c in enumerate(unique_counts)}

        # --- Draw connection lines ---
        for _, row in lines_df.iterrows():
            start = np.array(row['start'])
            end = np.array(row['end'])
            count = row['count']

            if combine_modalities:
                color = combined_color
                line_style = count_to_style.get(count, '-')
                curvature = base_curvature
            else:
                modality = row['modality']
                color = modality_colors.get(modality, 'black')
                line_style = count_to_style.get(count, '-')
                curvature = ((list(modality_colors.keys()).index(
                    modality
                    ) + 1) * base_curvature if modality in modality_colors else base_curvature)

            # Control point for quadratic Bezier
            if start[0] == end[0]:  # vertical
                control_point = ((start[0] + end[0]) / 2 + curvature, (start[1] + end[1]) / 2)
            elif start[1] == end[1]:  # horizontal
                control_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + curvature)
            else:  # diagonal -> bend perpendicular to the line
                delta = end - start
                perp = np.array([-delta[1], delta[0]])
                if np.linalg.norm(perp) != 0:
                    perp = perp / np.linalg.norm(perp)
                mid = (start + end) / 2
                control_point = tuple(mid + perp * curvature)

            vertices = np.array([start, control_point, end])
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            bezier_path = Path(vertices, codes)
            patch = PathPatch(
                bezier_path, color=color, lw=base_width, fill=False, linestyle=line_style, zorder=1
                )
            ax.add_patch(patch)

        # Draw pie charts for each scenario and manipulation combination
        for i, scenario in enumerate(scenario_order):
            for j, manipulation in enumerate(manipulation_order):
                cell_counts = cross_section_counts[(cross_section_counts['interaction scenario'] == scenario) & (
                        cross_section_counts['interaction manipulation'] == manipulation)]
                modality_counts = dict(zip(cell_counts['measurement modality'], cell_counts['count']))

                if modality_counts:
                    if combine_modalities:
                        # Single combined slice
                        total = sum(modality_counts.values())
                        combined_color = list(modality_colors.values())[0]
                        pie_sizes = [total]
                        pie_colors = [combined_color]
                        radius = pie_radius + (total / cross_section_counts['count'].sum())
                    else:
                        # Original multi-modality slices
                        pie_sizes = list(modality_counts.values())
                        pie_colors = [modality_colors[m] for m in modality_counts.keys()]
                        radius = pie_radius + (sum(pie_sizes) / cross_section_counts['count'].sum())

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
                        bbox=dict(boxstyle="circle", facecolor="white", edgecolor="none", pad=radius + 0.3)
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
                alpha=1
                )
            )

        if digital_im_column_indices:
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
                    alpha=1
                    )
                )

        # Axes and labels
        ax.set_xticks(col_pos)
        ax.set_yticks(row_pos)
        ax.set_xticklabels(
            [c.replace(" IM", '') for c in manipulation_order], rotation=45, ha='right', fontsize=14, color=font_color
            )
        ax.tick_params(axis='x', which='both', length=0, pad=10)

        scenario_labels = [categories["interaction scenario"].get(s, s).lower() for s in scenario_order]
        ax.set_yticklabels(scenario_labels, fontsize=14, color=font_color)
        ax.tick_params(axis='y', which='both', length=0, pad=10)

        ax.set_xlabel("Interaction manipulation", fontsize=16, labelpad=10, color=font_color)
        ax.set_ylabel("Interaction Scenario", fontsize=16, labelpad=10, color=font_color)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        condition_count = cross_section_counts["count"].sum()
        ax.set_title(
            f"Confusion matrix of experimental interaction conditions ({condition_count} "
            f"conditions in total {number_studies} studies)", fontsize=14, pad=10, color=font_color
            )

        # Legends
        legend_args = dict(
            loc="center left",
            fontsize=14,
            alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props, )
        legend_x = right_frac + 0.05
        y_top_bound = 0.7
        y_bottom_bound = 0.3

        filtered_area_handle = {}
        filtered_modality_handles = {}
        filtered_style_handles = {}
        # Get only counts actually used
        used_counts = count_to_style.keys()
        filtered_style_handles = {
            f"count: {count}": plt.Line2D([0], [0], color=font_color, lw=4, linestyle=count_to_style[count]) for count
            in used_counts}

        if not combine_modalities:
            # Create dictionary for legend handles
            modality_handles = {label: plt.Line2D([0], [0], color=color, lw=4, linestyle='-')
                                # Default solid line for modality
                                for label, color in modality_colors.items()}

            # Get only modalities actually used
            used_modalities = connection_df['modality'].unique()
            filtered_modality_handles = {label: handle for label, handle in modality_handles.items() if
                                         label in used_modalities}
        else:
            filtered_modality_handles = {
                'all modalities combined': plt.Line2D(
                    [0], [0], color=combined_color, lw=4, linestyle='-'
                    )
                }

        # Only add digital component if it's actually highlighted
        if digital_im_column_indices:  # non-empty list
            filtered_area_handle = {
                'digital component': mpatches.Patch(color=color_rect, label='digital component')
                }

        N = len(filtered_modality_handles) + len(filtered_style_handles) + len(filtered_area_handle)
        y_positions = np.linspace(y_top_bound, y_bottom_bound, N)

        y_top = y_positions[0]
        y_low = y_positions[-1]
        if len(filtered_modality_handles) == len(filtered_area_handle):
            y_mid = y_positions[len(y_positions) // 2]
        else:
            # take middle value between last modality and y_positions[-1]
            bottom_line_modality = y_positions[len(filtered_modality_handles) - 1]
            top_line_area = y_positions[-len(filtered_style_handles) - len(filtered_area_handle)]
            y_mid = (bottom_line_modality + top_line_area) / 2.0

        # Modality Legend (Fixed at top)
        mod_legend = ax.legend(
            handles=list(filtered_modality_handles.values()),
            labels=list(filtered_modality_handles.keys()),
            title="measurement modality",
            bbox_to_anchor=(legend_x, y_top),
            **legend_args
            )
        plt.setp(mod_legend.get_title(), color=font_color)

        # Comparison Legend (middle)
        comparison_legend = Legend(
            ax,
            handles=list(filtered_style_handles.values()),
            labels=list(filtered_style_handles.keys()),
            title="comparisons",
            bbox_to_anchor=(legend_x, y_mid),
            **legend_args
            )
        plt.setp(comparison_legend.get_title(), color=font_color)
        ax.add_artist(comparison_legend)

        # Study Design Legend (bottom, only if used)
        if filtered_area_handle:
            study_design_legend = Legend(
                ax,
                handles=list(filtered_area_handle.values()),
                labels=list(filtered_area_handle.keys()),
                title="study design",
                bbox_to_anchor=(legend_x, y_low),
                **legend_args
                )
            plt.setp(study_design_legend.get_title(), color=font_color)
            ax.add_artist(study_design_legend)

        plt.tight_layout()
        # adjust xlims to show connection lines fully that are in the outer parts of the figure
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
        df, selected_category, label_tooltips=None, container=None, count_mode="auto",
        # 'auto' | 'study_weighted' | 'occurrence'
        color_map=ColorMap,  # optional list of hex/rgb colors
        year_col="year", return_components=False, ):
    # -- Resolve container --
    if container is None:
        try:
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
        long_df.groupby(["year", "label_display"], as_index=False).agg(value=(value_field, "sum"), raw=("raw", "sum")))
    agg["year"] = agg["year"].astype(str)

    # Totals per year (overlay line)
    total_df = (agg.groupby("year", as_index=False).agg(total_studies=("value", "sum")).assign(
        year=lambda d: d["year"].astype(str)
        ).sort_values("year").dropna(subset=["total_studies"]).reset_index(drop=True))
    if total_df.empty:
        container.warning("No totals to plot.")
        return

    # Legend order by contribution
    label_order = (agg.groupby("label_display")["value"].sum().sort_values(ascending=False).index.tolist())

    # Unified, explicit x-domain for BOTH layers
    year_order = sorted(set(agg["year"]).union(set(total_df["year"])))
    x_enc = alt.X(
        "year:N", title="Year", sort=year_order, scale=alt.Scale(domain=year_order), )

    # ---- Bars (can be commented out safely) ----
    bars = (alt.Chart(agg).mark_bar().encode(
        x=x_enc,
        y=alt.Y(
            "value:Q",
            title="Studies (study-weighted)" if effective_mode == "study_weighted" else "Label occurrences", ),
        color=alt.Color(
            "label_display:N",
            title="Category",
            sort=label_order,
            scale=alt.Scale(range=color_map) if color_map else alt.Undefined,
            legend=None
            ),
        tooltip=[alt.Tooltip("year:N", title="Year"), alt.Tooltip("label_display:N", title="Category"), alt.Tooltip(
            "value:Q", title="Stack value", format=".2f" if effective_mode == "study_weighted" else ".0f"
            ), alt.Tooltip("raw:Q", title="Raw count", format=".0f"), ],
        order=alt.Order("label_display", sort="ascending"), ))

    # ---- Line ----
    line = (alt.Chart(total_df).mark_line(point=True, strokeWidth=2.5, clip=True).encode(
        x=x_enc,
        y=alt.Y("total_studies:Q", title=None),
        order=alt.Order("year:N", sort="ascending"),
        tooltip=[alt.Tooltip("year:N", title="Year"), alt.Tooltip(
            "total_studies:Q",
            title="Total studies (year)",
            format=".2f" if effective_mode == "study_weighted" else ".0f"
            ), ], ).transform_filter(alt.datum.total_studies != None))

    # Layer plots
    chart = (alt.layer(bars, line).resolve_scale(x="shared", y="shared").properties(height=380))
    legend_df = pd.DataFrame({"label_display": label_order})

    legend_chart = (alt.Chart(legend_df).mark_point(size=0)  # invisible points; just to trigger the legend
                    .encode(
        color=alt.Color(
            "label_display:N",
            title="Category",
            sort=label_order,
            scale=alt.Scale(range=color_map) if color_map else alt.Undefined,
            legend=alt.Legend(
                orient="none",  # free legend (not docked to a side)
                legendX=0, legendY=0, direction="vertical", labelLimit=0,  # no truncation
                ), )
        ).properties(width=300, height=380))

    if return_components:
        return {
            "chart": chart, "legend_chart": legend_chart, "label_order": label_order,
            }
    # -- Render (force fresh rerender via key to beat Streamlit caching issues) --
    key = f"pubs_{selected_category}_{effective_mode}_{len(agg)}_{len(label_order)}_{year_order[0]}_{year_order[-1]}"
    try:
        container.altair_chart(chart, use_container_width=True, key=key)
    except Exception:
        pass


def generate_parallel_sets_figure_colored(
        display_df: pd.DataFrame,
        chosen_parset_cols: List[str],
        *,
        min_count: int = 1,
        ColorMap: List[str] = None,
        color_by: str | None = None,  # can be any column in display_df (not just a stage)
        color_palette: List[str] | None = None,
        figure_background_color: str = None,
        primary_color: str = None,
        font_color: str = None, ) -> Tuple["go.Figure", int, int, pd.DataFrame]:
    """
    Build a Plotly Sankey figure (Parallel Sets) where link colors come from one chosen category.
    - If `color_by` is provided, it can be ANY column in `display_df` (not limited to `chosen_parset_cols`).
    - Counts reflect the expanded cartesian product across multi-label cells (same as original behavior).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    link_count : int
    node_count : int
    links_df : pd.DataFrame
        Columns:
        ['source_stage','source_value','target_stage','target_value','count']
        plus ['color_by_stage','color_by_value'] when `color_by` is not None.
    """
    # Local imports
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("Plotly is required to build the parallel sets figure.") from e

    # --- enforce correct type ---
    if isinstance(color_by, pd.Series):
        # likely user passed display_df[column] instead of column name
        color_by = color_by.name
    elif isinstance(color_by, (list, tuple)) and len(color_by) == 1:
        color_by = color_by[0]

    # ---- defaults / theming ----
    if ColorMap is None:
        ColorMap = ["#ADE1DC", "#C5C2E4", "#F28C8C", "#9FCBE1", "#F8BC63", "#CBEA7B", "#F5C6DA", "#BFD8B8", "#F9D9B6"]

    if color_palette is None:
        color_palette = ["#ADE1DC", "#C5C2E4", "#F28C8C", "#9FCBE1", "#F8BC63", "#CBEA7B", "#F5C6DA", "#BFD8B8",
                         "#F9D9B6"]

    # If running inside Streamlit, pick up theme values when not provided
    try:
        if figure_background_color is None:
            figure_background_color = st.get_option("theme.backgroundColor") or "#FFFFFF"
        if primary_color is None:
            primary_color = st.get_option("theme.primaryColor") or "#2C7BE5"
        if font_color is None:
            font_color = "#FFFFFF" if is_dark_color(figure_background_color) else "#000000"
    except Exception:
        figure_background_color = figure_background_color or "#FFFFFF"
        primary_color = primary_color or "#2C7BE5"
        font_color = font_color or ("#FFFFFF" if is_dark_color(figure_background_color) else "#000000")

    # ---- validate inputs ----
    if len(chosen_parset_cols) < 2:
        raise ValueError("chosen_parset_cols must contain at least two category columns.")

    if color_by is not None and color_by not in display_df.columns:
        raise ValueError("`color_by` must be a column in the currently selected data (it does not need to be in the "
                         "chosen columns).")

    # ---- build expanded tuples ----
    # We keep original behavior: expand FULL cartesian tuples across chosen stages for each row.
    # When `color_by` is set, we duplicate each tuple for every color value in that row (supports multi-label).
    tuples = []  # if color_by None: stores (v0, v1, ..., vk); else: stores (v0, ..., vk, cval)

    for _, row in display_df[chosen_parset_cols + ([color_by] if color_by else [])].iterrows():
        # stage_lists = [_norm_list(row[c]) for c in chosen_parset_cols]
        stage_lists = [_norm_list_safe_str(row[c]) for c in chosen_parset_cols]
        if any(len(lst) == 0 for lst in stage_lists):
            continue

        full_stage_combos = list(itertools.product(*stage_lists))

        if color_by is None:
            tuples.extend(full_stage_combos)
        else:
            # raw_val = row[color_by]
            # cvals = _norm_list_safe_str(raw_val)
            # if len(cvals) == 0:
            #     cvals = ["(missing)"]
            # for combo in full_stage_combos:
            #     for c in cvals:
            #         tuples.append((*combo, c))

            raw_val = row[color_by]
            if isinstance(raw_val, (list, tuple, set, np.ndarray)):
                cvals = list(raw_val)
            elif pd.isna(raw_val):

                cvals = ["(missing)"]
            else:
                cvals = _norm_list(raw_val)
                if len(cvals) == 0:
                    cvals = ["(missing)"]
            for combo in full_stage_combos:
                for c in cvals:
                    tuples.append((*combo, c))

    # Quick exit if no data
    if not tuples:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor=figure_background_color, plot_bgcolor=figure_background_color)
        return empty_fig, 0, 0, pd.DataFrame(
            columns=["source_stage", "source_value", "target_stage", "target_value", "count"]
            )

    # ---- nodes (unique per stage), and value-only labels for display ----
    stage_labels_full, stage_labels_display = [], []
    for i, col in enumerate(chosen_parset_cols):
        vals = sorted({t[i] for t in tuples})  # positions 0..k are always stages
        stage_labels_full.append([f"{col}: {v}" for v in vals])
        stage_labels_display.append([str(v) for v in vals])

    node_labels_full = list(itertools.chain.from_iterable(stage_labels_full))
    node_labels_display = list(itertools.chain.from_iterable(stage_labels_display))
    node_index = {lab: idx for idx, lab in enumerate(node_labels_full)}

    # ---- links (adjacent stages), counts ----
    link_rows = []
    if color_by is None:
        for i in range(len(chosen_parset_cols) - 1):
            pair_counter = Counter((t[i], t[i + 1]) for t in tuples)
            for (left, right), count in pair_counter.items():
                link_rows.append(
                    {
                        "source_stage": chosen_parset_cols[i],
                        "source_value": left,
                        "target_stage": chosen_parset_cols[i + 1],
                        "target_value": right,
                        "count": count
                        }
                    )
    else:
        # color value is stored at t[-1]
        for i in range(len(chosen_parset_cols) - 1):
            trip_counter = Counter((t[i], t[i + 1], t[-1]) for t in tuples)
            for (left, right, cval), count in trip_counter.items():
                link_rows.append(
                    {
                        "source_stage": chosen_parset_cols[i],
                        "source_value": left,
                        "target_stage": chosen_parset_cols[i + 1],
                        "target_value": right,
                        "count": count,
                        "color_by_stage": color_by,
                        "color_by_value": cval
                        }
                    )

    links_df = pd.DataFrame(link_rows)

    # ---- filter links by min_count ----
    mask = links_df["count"] >= max(1, int(min_count))
    if not mask.any():
        stage_colors = [ColorMap[i % len(ColorMap)] for i in range(len(chosen_parset_cols))]
        node_colors = []
        for stage_idx, labels_this_stage in enumerate(stage_labels_full):
            node_colors.extend([stage_colors[stage_idx]] * len(labels_this_stage))

        fig = go.Figure(
            data=[go.Sankey(
                arrangement="snap", node=dict(
                    pad=12,
                    thickness=16,
                    line=dict(width=1, color=primary_color),
                    label=node_labels_display,
                    color=node_colors
                    ), link=dict(source=[], target=[], value=[], color=[]), )]
            )
        annotations = []
        n_stages = len(chosen_parset_cols)
        for i, col in enumerate(chosen_parset_cols):
            x = (i / (n_stages - 1)) if n_stages > 1 else 0.5
            annotations.append(
                dict(
                    x=x,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{col}</b>",
                    showarrow=False,
                    font=dict(size=14, color=font_color)
                    )
                )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=figure_background_color,
            plot_bgcolor=figure_background_color,
            font=dict(color=font_color, size=12, family="Arial"),
            annotations=annotations
            )
        fig.update_traces(textfont=dict(color=font_color, size=12, family="Arial"))
        return fig, 0, len(node_labels_full), links_df.iloc[0:0]

    links_df = links_df.loc[mask].copy()

    # ---- node colors (by stage), link colors (by chosen `color_by` or source stage fallback) ----
    stage_colors = [ColorMap[i % len(ColorMap)] for i in range(len(chosen_parset_cols))]

    node_colors, stage_index_per_node = [], []
    for stage_idx, labels_this_stage in enumerate(stage_labels_full):
        node_colors.extend([stage_colors[stage_idx]] * len(labels_this_stage))
        stage_index_per_node.extend([stage_idx] * len(labels_this_stage))

    value_to_color, legend_items = {}, []
    if color_by is not None:
        ordered_vals = sorted(links_df["color_by_value"].unique(), key=lambda x: str(x))
        for i, v in enumerate(ordered_vals):
            value_to_color[v] = color_palette[i % len(color_palette)]
        legend_items = [(str(v), value_to_color[v]) for v in ordered_vals]

    # ---- derive filtered arrays in node-index space + colors ----
    fs, ft, fv, fcolors, hovertext = [], [], [], [], []
    for _, r in links_df.iterrows():
        src_full = f"{r['source_stage']}: {r['source_value']}"
        tgt_full = f"{r['target_stage']}: {r['target_value']}"
        s_idx = node_index[src_full]
        t_idx = node_index[tgt_full]

        fs.append(s_idx)
        ft.append(t_idx)
        fv.append(int(r["count"]))

        if color_by is None:
            s_stage_idx = stage_index_per_node[s_idx]
            fcolors.append(_rgba(stage_colors[s_stage_idx], 0.35))
            hovertext.append(
                f"<b>{r['source_stage']}</b>: {r['source_value']} → "
                f"<b>{r['target_stage']}</b>: {r['target_value']}<br>"
                f"Studies: {r['count']}"
                )
        else:
            cval = r["color_by_value"]
            base = value_to_color.get(cval, "#888888")
            fcolors.append(_rgba(base, 0.7))
            hovertext.append(
                f"<b>{r['source_stage']}</b>: {r['source_value']} → "
                f"<b>{r['target_stage']}</b>: {r['target_value']}<br>"
                f"<b>{color_by}</b>: {cval}<br>"
                f"Studies: {r['count']}"
                )

    # ---- build figure ----
    fig = go.Figure()
    fig.add_trace(
        go.Sankey(
            arrangement="snap", node=dict(
                pad=12, thickness=16, line=dict(width=1, color=font_color), label=node_labels_display, color=node_colors
                ), link=dict(
                source=list(fs), target=list(ft), value=list(fv),  # thickness = number of studies (expanded)
                color=fcolors, customdata=[[t] for t in hovertext], hovertemplate="%{customdata[0]}<extra></extra>", )
            )
        )

    # Stage headers as annotations
    annotations = []
    n_stages = len(chosen_parset_cols)
    for i, col in enumerate(chosen_parset_cols):
        x = (i / (n_stages - 1)) if n_stages > 1 else 0.5
        annotations.append(
            dict(
                x=x,
                y=1.08,
                xref="paper",
                yref="paper",
                text=f"<b>{col}</b>",
                showarrow=False,
                font=dict(size=14, color=font_color)
                )
            )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=figure_background_color,
        plot_bgcolor=figure_background_color,
        font=dict(color=font_color, size=12, family="Arial"),
        annotations=annotations
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_traces(textfont=dict(color=font_color, size=12, family="Arial"))

    # ---- (optional) legend for link colors ----
    if color_by is not None and legend_items:
        for name, colhex in legend_items:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=colhex),
                    name=f"{name}",
                    showlegend=True,
                    hoverinfo="skip",
                    )
                )
        fig.update_layout(
            legend=dict(
                title=dict(text=f"{color_by}"), orientation="h", yanchor="bottom", y=-0.15,
                # moves it below the plot
                xanchor="center", x=0.5, font=dict(size=12), )
            )
    link_count = len(links_df)
    node_count = len(node_labels_full)
    return fig, link_count, node_count, links_df
