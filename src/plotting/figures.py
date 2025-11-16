from src.plotting.plot_utils import _norm_list_safe_str, _rgba

try:
    import altair as alt

    alt.data_transformers.disable_max_rows()  # prevent silent failures on larger data
except Exception:
    pass

import itertools
from collections import defaultdict, Counter
from itertools import combinations
from threading import RLock
from typing import Tuple, List

import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from src.plotting.plot_utils import *
from src.utils.config import ColorMap

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

                    chart_df = pd.DataFrame(
                        {
                            "label": counts.index, "count": counts.values
                            }
                        )

                    max_count = int(chart_df["count"].max()) if not chart_df.empty else 0
                    # integer tick positions from 0 up to max_count
                    x_values = list(range(0, max_count + 1))

                    bar_chart = (alt.Chart(chart_df).mark_bar(color=st.get_option("theme.primaryColor")).encode(
                        x=alt.X(
                            "count:Q", title="count", scale=alt.Scale(domain=[0, max_count], nice=False), axis=alt.Axis(
                                values=x_values,  # only integer ticks
                                tickMinStep=1,  # ensure integer step
                                grid=True, labelColor=font_color, titleColor=font_color, ), ), y=alt.Y(
                            "label:N", sort=category_definitions[cat].keys(), title="", axis=alt.Axis(
                                labelColor=font_color, titleColor=font_color, labelLimit=1000, labelOverlap=False, ), ),

                        tooltip=["label", "count"], ).properties(
                        height=300, padding={"left": 20, "right": 20, "top": 20, "bottom": 20}, ).configure_view(
                        stroke=None
                        ).configure_axis(
                        labelFontSize=14, labelLimit=1000, labelPadding=10, titlePadding=15, ))

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
    default_medium_order = list(categories["interaction medium"].keys())
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
            'interaction medium'].apply(lambda x: valid_values(x, default_medium_order)) | ~df_raw[
            'measurement modality'].apply(lambda x: valid_values(x, default_modalities_order))]

    if not non_default_rows.empty:
        tab.warning(
            f"⚠️ Some studies contain non-default entries in interaction scenario, medium, or modality. These studies are excluded from the figure. (DOIs: {', '.join(non_default_rows['doi'].unique())})"
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
    medium_columns = [c for c in df.columns if c.startswith(f"interaction medium{prefix}")]
    modality_columns = [c for c in df.columns if c.startswith(f"measurement modality{prefix}")]

    ####################
    # Count conditions #
    ####################
    # condition_rows = []
    # for doi, group in df_orig.groupby('doi'):
    #     orig_row = df_orig[df_orig['doi'] == doi].iloc[0]
    #     modalities = ensure_list(orig_row['measurement modality'])
    #     scenarios = ensure_list(orig_row['interaction scenario'])
    #     mediums = ensure_list(orig_row['interaction medium'])
    #
    #     for modality in modalities:
    #         if len(scenarios) == len(mediums) and len(scenarios) > 1:
    #             for s, m in zip(scenarios, mediums):
    #                 condition_rows.append([modality, s, m, doi])
    #         elif len(scenarios) == 1:
    #             for m in mediums:
    #                 condition_rows.append([modality, scenarios[0], m, doi])
    #         elif len(mediums) == 1:
    #             for s in scenarios:
    #                 condition_rows.append([modality, s, mediums[0], doi])
    #         else:
    #             for s in scenarios:
    #                 for m in mediums:
    #                     condition_rows.append([modality, s, m, doi])
    condition_rows = []
    for doi, group in df_orig.groupby("doi"):
        orig_row = group.iloc[0]
        modalities = ensure_list(orig_row["measurement modality"])

        # study-level conditions (scenario, medium), ignoring modality
        conds = conditions_from_row(orig_row)

        for modality in modalities:
            for s, m in conds:
                condition_rows.append([modality, s, m, doi])

    condition_df = pd.DataFrame(
        condition_rows, columns=["measurement modality", "interaction scenario", "interaction medium", "doi"], )

    cross_section_counts = condition_df.groupby(
        ['measurement modality', 'interaction scenario', 'interaction medium']
        ).size().reset_index(name='count')

    scenario_contained = [col.replace("interaction scenario_", "") for col in scenario_columns]
    medium_contained = [col.replace("interaction medium_", "") for col in medium_columns]
    modalities_contained = [col.replace("measurement modality_", "") for col in modality_columns]

    scenario_order = [s for s in default_scenario_order if s in scenario_contained]
    medium_order = [m for m in default_medium_order if m in medium_contained]

    row_pos = np.arange(len(scenario_order)) * row_spacing
    col_pos = np.arange(len(medium_order)) * col_spacing

    ####################
    # Count connection #
    ####################
    connection_counts = defaultdict(int)
    connection_display_orders = defaultdict(list)
    pairing_warnings = []

    for doi, group in df_orig.groupby("doi"):
        orig_row = group.iloc[0]
        modalities = [mod for mod in modalities_contained if group[f"measurement modality_{mod}"].iloc[0] == 1]

        # Same study-level conditions (scenario, medium)
        conds = conditions_from_row(orig_row)

        # Map conditions to plot coordinates once
        pairs_base = []
        for s, m in conds:
            if s not in scenario_order or m not in medium_order:
                continue
            y = row_pos[scenario_order.index(s)]
            x = col_pos[medium_order.index(m)]
            pairs_base.append((x, y, s, m))

        for modality in modalities:
            pairs = list(pairs_base)
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
                        )  # len(pairs) < 2 -> no connections for this modality/study

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

        # Draw pie charts for each scenario and medium combination
        for i, scenario in enumerate(scenario_order):
            for j, medium in enumerate(medium_order):
                cell_counts = cross_section_counts[(cross_section_counts['interaction scenario'] == scenario) & (
                        cross_section_counts['interaction medium'] == medium)]
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
        digital_im_column_indices = [medium_order.index(col) for col in medium_order if "digital" in col]

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
            [c.replace(" IM", '') for c in medium_order], rotation=45, ha='right', fontsize=14, color=font_color
            )
        ax.tick_params(axis='x', which='both', length=0, pad=10)

        scenario_labels = [categories["interaction scenario"].get(s, s).lower() for s in scenario_order]
        ax.set_yticklabels(scenario_labels, fontsize=14, color=font_color)
        ax.tick_params(axis='y', which='both', length=0, pad=10)

        ax.set_xlabel("Interaction medium", fontsize=16, labelpad=10, color=font_color)
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
            title="cross-condition occurence",
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

        # === Digital vs non-digital definition (IM/IS labels) ===
        digital_is_labels = {"virtual"}  # IS: virtual: Interaction via a virtual divide or environment

        digital_im_labels = {"separate digital IM and verbal IM",  # Separate digital interface with verbal interaction
                             "separate digital IM w/out verbal IM",  # Separate digital interface, no verbal interaction
                             "shared digital IM and verbal IM",  # Shared digital interface with verbal interaction
                             "shared digital IM w/out verbal IM",  # Shared digital interface, no verbal interaction
                             }

        def _is_digital_im_is(scenario_label: str, medium_label: str) -> bool:
            is_digital_is = scenario_label in digital_is_labels
            is_digital_im = medium_label in digital_im_labels
            return bool(is_digital_is or is_digital_im)

        # === Verbal vs non-verbal IM labels ===
        verbal_im_labels = {"separate digital IM and verbal IM", "shared digital IM and verbal IM",
                            "shared physical IM and verbal IM", "verbal IM", }

        nonverbal_im_labels = {"non-verbal IM", "physical IM w/out verbal IM", "separate digital IM w/out verbal IM",
                               "shared digital IM w/out verbal IM", }

        # === Digital vs non-digital definition (IM/IS labels) ===
        digital_is_labels = {"virtual"}  # IS: virtual divide / environment

        digital_im_labels = {"separate digital IM and verbal IM", "separate digital IM w/out verbal IM",
                             "shared digital IM and verbal IM", "shared digital IM w/out verbal IM", }

        def _is_digital_im_is(scenario_label: str, medium_label: str) -> bool:
            is_digital_is = scenario_label in digital_is_labels
            is_digital_im = medium_label in digital_im_labels
            return bool(is_digital_is or is_digital_im)

        # Verbal vs non-verbal IM labels
        verbal_im_labels = {"separate digital IM and verbal IM", "shared digital IM and verbal IM",
                            "shared physical IM and verbal IM", "verbal IM", }

        nonverbal_im_labels = {"non-verbal IM", "physical IM w/out verbal IM", "separate digital IM w/out verbal IM",
                               "shared digital IM w/out verbal IM", }

        # === Study-based summary stats, pair-level ===
        unique_im_is_global: set[tuple[str, str]] = set()

        cross_condition_occurrences = 0
        constant_im_varying_is = 0
        constant_is_varying_im = 0
        both_varying = 0
        digital_vs_nondigital_comparison = 0

        studies_with_verbal_im = 0
        studies_without_verbal_im = 0

        # IM labels per study for verbal/non-verbal classification
        im_by_study = (df_orig.groupby("doi")["interaction medium"].apply(lambda col: set(ensure_list(col.iloc[0]))))

        for doi, g in df_orig.groupby("doi"):
            orig_row = g.iloc[0]

            # Study-level conditions (scenario, medium), ordered pairing
            cond_pairs = conditions_from_row(orig_row)
            if not cond_pairs:
                continue

            # Deduplicate within study (same condition appearing multiple times)
            cond_pairs = list(set(cond_pairs))

            # Track global unique IM–IS
            for s, m in cond_pairs:
                unique_im_is_global.add((s, m))

            # Verbal vs non-verbal IM at study level
            im_vals = im_by_study.get(doi, set())
            has_verbal = bool(im_vals & verbal_im_labels)
            has_any_known_im = bool(im_vals & (verbal_im_labels | nonverbal_im_labels))

            if has_verbal:
                studies_with_verbal_im += 1
            elif has_any_known_im:
                studies_without_verbal_im += 1

            # Need at least 2 conditions to compare
            if len(cond_pairs) < 2:
                continue

            # Prepare condition tuples with digital flag
            conds = [(s, m, _is_digital_im_is(s, m)) for s, m in cond_pairs]

            # Pairwise comparisons within this study
            for i in range(len(conds)):
                s1, m1, d1 = conds[i]
                for j in range(i + 1, len(conds)):
                    s2, m2, d2 = conds[j]

                    cross_condition_occurrences += 1

                    same_im = (m1 == m2)
                    same_is = (s1 == s2)

                    if same_im and not same_is:
                        constant_im_varying_is += 1
                    elif not same_im and same_is:
                        constant_is_varying_im += 1
                    elif not same_im and not same_is:
                        both_varying += 1
                    # same_im & same_is shouldn't happen after dedup

                    if (d1 and not d2) or (d2 and not d1):
                        digital_vs_nondigital_comparison += 1

        unique_im_is_combinations = len(unique_im_is_global)

        summary_stats = {
            "total conditions": condition_count,
            "unique IM-IS combinations": unique_im_is_combinations,
            "cross-condition occurrences": cross_condition_occurrences,
            "constant IM, varying IS": constant_im_varying_is,
            "constant IS, varying IM": constant_is_varying_im,
            "both varying": both_varying,
            "digital vs non-digital comparison": digital_vs_nondigital_comparison,
            "studies with verbal IM": studies_with_verbal_im,
            "studies without verbal IM": studies_without_verbal_im,
            }

        return fig, condition_count, number_studies, connection_df, summary_stats


def plot_publications_over_time(
        df,
        selected_category,
        label_tooltips=None,
        container=None,
        count_mode="auto",
        color_map=ColorMap,
        year_col="year",
        return_components=False, ):
    """
    Plot number of publications over time for a given categorical label.
    Supports study-weighted and raw occurrence counts.
    """

    # Determine default font_color
    bg = st.get_option("theme.backgroundColor")
    font_color = "#FFFFFF" if is_dark_color(bg) else "#000000"

    if selected_category is None or selected_category == 'None':
        year_counts = df["year"].value_counts().sort_index()
        year_df = pd.DataFrame({"Publications": year_counts})
        year_df.index = year_df.index.astype(str)
        font_color = "#FFFFFF" if is_dark_color(st.get_option("theme.backgroundColor")) else "#000000"
        chart = (alt.Chart(year_df.reset_index()).mark_line(
            stroke=font_color, color=font_color
            ).encode(
            x=alt.X(
                "year:N", title="year", axis=alt.Axis(
                    labelColor=font_color, titleColor=font_color, labelAngle=0, ), ), y=alt.Y(
                "Publications:Q", title="number of publications", axis=alt.Axis(
                    labelColor=font_color, titleColor=font_color
                    ), ), tooltip=["year:N", "Publications:Q"], ).properties(height=380))

        st.altair_chart(chart, use_container_width=True)
    else:
        # Disable Altair row limit if possible
        try:
            alt.data_transformers.disable_max_rows()
        except Exception:
            pass

        # Determine count mode
        effective_mode = "study_weighted" if count_mode == "auto" else count_mode
        if effective_mode not in ["study_weighted", "occurrence"]:
            container.error(f"Selected count_mode {count_mode} is not supported.")
            return

        # Parse labels per row
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

            labels = [str(l) for l in labels if l not in (None, "", np.nan)]
            labels = list(dict.fromkeys(labels))
            if not labels:
                continue

            weight = 1.0 / len(labels)
            for lab in labels:
                year_label_rows.append(
                    {"year": year, "label": lab, "weight": weight, "raw": 1.0}
                    )

        if not year_label_rows:
            container.warning("No data to plot for the selected category.")
            return

        # Build long dataframe
        long_df = pd.DataFrame(year_label_rows)
        label_tooltips = label_tooltips or {}
        long_df["label_display"] = long_df["label"].map(lambda x: label_tooltips.get(x, x))

        # Choose weight field
        value_field = "weight" if effective_mode == "study_weighted" else "raw"

        # Aggregate
        agg = (long_df.groupby(["year", "label_display"], as_index=False).agg(
            value=(value_field, "sum"), raw=("raw", "sum")
            ))
        agg["year"] = agg["year"].astype(str)

        # Totals per year
        total_df = (
            agg.groupby("year", as_index=False).agg(total_studies=("value", "sum")).sort_values("year").reset_index(
                drop=True
                ))
        if total_df.empty:
            container.warning("No totals to plot.")
            return

        # Order labels by total contribution
        label_order = (agg.groupby("label_display")["value"].sum().sort_values(ascending=False).index.tolist())

        # Shared x domain
        year_order = sorted(set(agg["year"]).union(total_df["year"]))
        x_enc = alt.X(
            "year:N",
            title="Year",
            sort=year_order,
            scale=alt.Scale(domain=year_order),
            axis=alt.Axis(labelColor=font_color, titleColor=font_color), )

        # Bars
        bars = (alt.Chart(agg).mark_bar().encode(
            x=x_enc,
            y=alt.Y(
                "value:Q",
                title="Studies (study-weighted)" if effective_mode == "study_weighted" else "Label occurrences",
                axis=alt.Axis(labelColor=font_color, titleColor=font_color), ),
            color=alt.Color(
                "label_display:N",
                title="Category",
                sort=label_order,
                scale=alt.Scale(range=color_map) if color_map else alt.Undefined,
                legend=None, ),
            tooltip=[alt.Tooltip("year:N", title="Year"), alt.Tooltip("label_display:N", title="Category"), alt.Tooltip(
                "value:Q", title="Stack value", format=".2f" if effective_mode == "study_weighted" else ".0f", ),
                     alt.Tooltip("raw:Q", title="Raw count", format=".0f"), ],
            order=alt.Order("label_display", sort="ascending"), ))

        # Line for totals
        line = (alt.Chart(total_df).mark_line(
            strokeWidth=2.5, clip=True, color=font_color, point={"color": font_color}, ).encode(
            x=x_enc,
            y=alt.Y("total_studies:Q", title=None, axis=alt.Axis(labelColor=font_color), ),
            order=alt.Order("year:N", sort="ascending"),
            tooltip=[alt.Tooltip("year:N", title="Year"), alt.Tooltip(
                "total_studies:Q",
                title="Total studies (year)",
                format=".2f" if effective_mode == "study_weighted" else ".0f", ), ], ).transform_filter(
            alt.datum.total_studies != None
            ))

        # Layer main chart
        chart = (alt.layer(bars, line).resolve_scale(x="shared", y="shared").properties(height=380))

        # Legend chart
        legend_df = pd.DataFrame({"label_display": label_order})
        legend_chart = (alt.Chart(legend_df).mark_point(size=0).encode(
            color=alt.Color(
                "label_display:N",
                title="Category",
                sort=label_order,
                scale=alt.Scale(range=color_map) if color_map else alt.Undefined,
                legend=alt.Legend(
                    orient="none",
                    legendX=0,
                    legendY=0,
                    direction="vertical",
                    labelLimit=0,
                    labelColor=font_color,
                    titleColor=font_color, )
                )
            ).properties(width=300, height=380))

        if return_components:
            return {
                "chart": chart, "legend_chart": legend_chart, "label_order": label_order,
                }

        # Render
        key = f"pubs_{selected_category}_{effective_mode}_{len(agg)}_{len(label_order)}_{year_order[0]}_{year_order[-1]}"
        try:
            container.altair_chart(chart, use_container_width=True, key=key)
        except Exception:
            pass


def generate_alluvial_plot(
        display_df: pd.DataFrame,
        chosen_parset_cols: List[str],
        *,
        min_count: int = 1,
        color_by: str | None = None, ) -> Tuple["go.Figure", int, int, pd.DataFrame, str]:
    """
    Build a Plotly Sankey (parallel sets) figure.

    Parameters
    ----------
    display_df : pd.DataFrame
        Source data.
    chosen_parset_cols : list of str
        Ordered stages (left to right).
    min_count : int, optional
        Minimum link count to retain.
    color_by : str or None, optional
        Column name used to color links.

    Returns
    -------
    fig : go.Figure
    link_count : int
    node_count : int
    links_df : pd.DataFrame
    diag_markdown : str
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("Plotly is required to build the parallel sets figure.") from e

    # Category definitions (for stable ordering)
    try:
        with open("./data/info_yamls/categories.yaml", "r", encoding="utf-8") as f:
            category_definitions = yaml.safe_load(f) or {}
    except Exception:
        category_definitions = {}

    # Normalize color_by input type
    if isinstance(color_by, pd.Series):
        color_by = color_by.name
    elif isinstance(color_by, (list, tuple)) and len(color_by) == 1:
        color_by = color_by[0]

    # Theme defaults
    ColorMap = ["#ADE1DC", "#C5C2E4", "#F28C8C", "#9FCBE1", "#F8BC63", "#CBEA7B", "#F5C6DA", "#BFD8B8", "#F9D9B6"]

    color_palette = ["#ADE1DC", "#C5C2E4", "#F28C8C", "#9FCBE1", "#F8BC63", "#CBEA7B", "#F5C6DA", "#BFD8B8", "#F9D9B6"]

    try:
        figure_background_color = st.get_option("theme.backgroundColor") or "#FFFFFF"
        font_color = "#FFFFFF" if is_dark_color(figure_background_color) else "#000000"
        primary_color = st.get_option("theme.primaryColor") or "#2C7BE5"
    except Exception:
        figure_background_color = figure_background_color or "#FFFFFF"
        primary_color = primary_color or "#2C7BE5"
        font_color = "#000000"

    # Basic validation
    if len(chosen_parset_cols) < 2:
        raise ValueError("chosen_parset_cols must contain at least two category columns.")

    if color_by is not None and color_by not in display_df.columns:
        raise ValueError(
            "`color_by` must be a column in the currently selected data "
            "(it does not need to be in chosen_parset_cols)."
            )

    # Normalize selected columns to list[str] per cell
    norm_cols = list(chosen_parset_cols)
    if color_by is not None and color_by not in norm_cols:
        norm_cols.append(color_by)

    norm_df = display_df.copy()
    for col in norm_cols:
        norm_df[col] = norm_df[col].apply(_norm_list_safe_str)

    # Per-stage category counts (for node diagnostics)
    cat_counts: dict[str, Counter] = {col: Counter() for col in chosen_parset_cols}
    for _, row in norm_df[chosen_parset_cols].iterrows():
        for col in chosen_parset_cols:
            for lab in row[col]:
                cat_counts[col][lab] += 1

    # Node labels per stage, ordered by categories.yaml
    stage_labels_full: list[list[str]] = []
    stage_labels_display: list[list[str]] = []

    for col in chosen_parset_cols:
        vals_present = set(cat_counts[col].keys())
        yaml_order = list(category_definitions.get(col, {}).keys())

        ordered_vals = [v for v in yaml_order if v in vals_present]
        extra_vals = sorted(v for v in vals_present if v not in yaml_order)
        ordered_vals.extend(extra_vals)

        stage_labels_full.append([f"{col}: {v}" for v in ordered_vals])
        stage_labels_display.append([str(v) for v in ordered_vals])

    node_labels_full = list(itertools.chain.from_iterable(stage_labels_full))
    node_labels_display = list(itertools.chain.from_iterable(stage_labels_display))
    node_index = {lab: idx for idx, lab in enumerate(node_labels_full)}

    # Build link table (adjacent stages)
    link_rows: list[dict] = []
    color_in_stages = color_by in chosen_parset_cols if color_by else False
    n_stages = len(chosen_parset_cols)

    for i in range(n_stages - 1):
        src_col = chosen_parset_cols[i]
        tgt_col = chosen_parset_cols[i + 1]

        if color_by is None:
            pair_counter: Counter = Counter()

            for _, row in norm_df[[src_col, tgt_col]].iterrows():
                src_labels = row[src_col]
                tgt_labels = row[tgt_col]
                if not src_labels or not tgt_labels:
                    continue

                seen_pairs = set()
                for s in src_labels:
                    for t in tgt_labels:
                        key = (s, t)
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            pair_counter[key] += 1

            for (left, right), count in pair_counter.items():
                link_rows.append(
                    dict(
                        source_stage=src_col,
                        source_value=left,
                        target_stage=tgt_col,
                        target_value=right,
                        count=count, )
                    )

        else:
            trip_counter: Counter = Counter()

            for _, row in norm_df[norm_cols].iterrows():
                src_labels = row[src_col]
                tgt_labels = row[tgt_col]
                if not src_labels or not tgt_labels:
                    continue

                if color_in_stages:
                    if color_by == src_col:
                        color_labels = src_labels
                    elif color_by == tgt_col:
                        color_labels = tgt_labels
                    else:
                        color_labels = row[color_by]
                else:
                    color_labels = row[color_by]

                if not color_labels:
                    color_labels = ["(missing)"]

                seen_triples = set()
                for s in src_labels:
                    for t in tgt_labels:
                        for c in color_labels:
                            key = (s, t, str(c))
                            if key not in seen_triples:
                                seen_triples.add(key)
                                trip_counter[key] += 1

            for (left, right, cval), count in trip_counter.items():
                link_rows.append(
                    dict(
                        source_stage=src_col,
                        source_value=left,
                        target_stage=tgt_col,
                        target_value=right,
                        count=count,
                        color_by_stage=color_by,
                        color_by_value=str(cval), )
                    )

    links_df = pd.DataFrame(link_rows)

    # Filter by minimum link count
    min_count = max(1, int(min_count))
    mask = links_df["count"] >= min_count

    if not mask.any():
        # Empty plot with nodes only
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
                    color=node_colors, ), link=dict(source=[], target=[], value=[], color=[]), )]
            )

        annotations = []
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
                    font=dict(size=14, color=font_color), )
                )

        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor=figure_background_color,
            plot_bgcolor=figure_background_color,
            font=dict(color=font_color, size=12, family="Arial"),
            annotations=annotations, )
        fig.update_traces(textfont=dict(color=font_color, size=12, family="Arial"))
        return fig, 0, len(node_labels_full), links_df.iloc[0:0], ""

    links_df = links_df.loc[mask].copy()

    # Incoming/outgoing flow per node (for diagnostics)
    incoming_flow = {col: Counter() for col in chosen_parset_cols}
    outgoing_flow = {col: Counter() for col in chosen_parset_cols}

    for _, r in links_df.iterrows():
        s_col = r["source_stage"]
        t_col = r["target_stage"]
        s_val = r["source_value"]
        t_val = r["target_value"]
        cnt = r["count"]

        outgoing_flow[s_col][s_val] += cnt
        incoming_flow[t_col][t_val] += cnt

    # Node hover text
    node_hover_texts: list[str] = []
    for col, labels_display in zip(chosen_parset_cols, stage_labels_display):
        for lab in labels_display:
            count = cat_counts[col].get(lab, 0)
            inc = incoming_flow[col].get(lab, 0)
            out = outgoing_flow[col].get(lab, 0)
            node_hover_texts.append(
                f"<b>{col}</b>: {lab}<br>"
                f"outgoing flow: {out:.2f}<br>"
                f"incoming flow: {inc:.2f}<br>"
                f"category count: {count}"
                )

    # Node & link colors
    stage_colors = [ColorMap[i % len(ColorMap)] for i in range(len(chosen_parset_cols))]

    node_colors: list[str] = []
    stage_index_per_node: list[int] = []
    for stage_idx, labels_this_stage in enumerate(stage_labels_full):
        stage_index_per_node.extend([stage_idx] * len(labels_this_stage))

    if color_by is None:
        # Nodes colored by stage, neutral links
        for stage_idx, labels_this_stage in enumerate(stage_labels_full):
            node_colors.extend([stage_colors[stage_idx]] * len(labels_this_stage))
    else:
        # Links colored by color_by; nodes in neutral gray
        neutral_node_color = "#43454a" if is_dark_color(figure_background_color) else "#e9e9e9"
        node_colors = [neutral_node_color] * len(node_labels_full)

    value_to_color, legend_items = {}, []
    if color_by is not None:
        unique_vals = list({str(v) for v in links_df["color_by_value"].unique()})
        yaml_order = list(category_definitions.get(color_by, {}).keys())

        ordered_vals = [v for v in yaml_order if v in unique_vals]
        ordered_vals.extend(sorted(v for v in unique_vals if v not in yaml_order))

        for i, v in enumerate(ordered_vals):
            value_to_color[v] = color_palette[i % len(color_palette)]
        legend_items = [(v, value_to_color[v]) for v in ordered_vals]

    # Links in node-index space
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
        else:
            cval = r["color_by_value"]
            base = value_to_color.get(cval, "#888888")
            fcolors.append(_rgba(base, 0.7))

        hovertext.append(
            f"<b>{r['source_stage']}</b>: {r['source_value']} → "
            f"<b>{r['target_stage']}</b>: {r['target_value']}<br>"
            f"count: {r['count']}"
            )

    # Build figure
    fig = go.Figure()
    fig.add_trace(
        go.Sankey(
            arrangement="snap", node=dict(
                pad=12,
                thickness=16,
                line=dict(width=1, color=font_color),
                label=node_labels_display,
                color=node_colors,
                customdata=[[t] for t in node_hover_texts],
                hovertemplate="%{customdata[0]}<extra></extra>", ), link=dict(
                source=list(fs),
                target=list(ft),
                value=list(fv),
                color=fcolors,
                customdata=[[t] for t in hovertext],
                hovertemplate="%{customdata[0]}<extra></extra>", ), )
        )

    # Stage headers
    annotations = []
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
                font=dict(size=14, color=font_color), )
            )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=figure_background_color,
        plot_bgcolor=figure_background_color,
        font=dict(color=font_color, size=12, family="Arial"),
        annotations=annotations, )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_traces(textfont=dict(color=font_color, size=12, family="Arial"))

    # Legend for link colors when color_by is set
    if color_by is not None and legend_items:
        for name, colhex in legend_items:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=colhex),
                    name=str(name),
                    showlegend=True,
                    hoverinfo="skip", )
                )
        fig.update_layout(
            legend=dict(
                title=dict(text=str(color_by)),
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=12), )
            )

    link_count = len(links_df)
    node_count = len(node_labels_full)

    # Diagnostic: compare first stage category counts vs outgoing flow
    first_stage = chosen_parset_cols[0]
    cat_counter = cat_counts[first_stage]

    sankey_counter = Counter()
    for _, r in links_df.iterrows():
        if r["source_stage"] == first_stage:
            sankey_counter[r["source_value"]] += int(r["count"])

    lines = [f"**Diagnostic: counts for first stage (`{first_stage}`)**", "",
             "| Label | Category-count style (per row) | Sankey outgoing sum |", "| --- | ---: | ---: |", ]
    for lab in sorted(cat_counter.keys(), key=str):
        lines.append(
            f"| {lab} | {cat_counter[lab]} | {sankey_counter.get(lab, 0)} |"
            )

    diag_markdown = "\n".join(lines)

    return fig, link_count, node_count, links_df, diag_markdown


def generate_heatmap_figure(df, tab):
    """
    Generate a heatmap (confusion matrix-style) comparing label co-occurrence
    between two categorical columns. Handles list-like cells robustly and
    preserves label order from categories.yaml, while dropping rows/columns
    that are all zeros.
    """
    if df is None or df.empty:
        tab.warning("⚠️ No data found; make sure that at least some data is passing the selected filters.")
        return

    # --- Theme / categories setup ---
    figure_background_color = st.get_option("theme.backgroundColor")
    font_color = "#ffffff" if is_dark_color(figure_background_color) else "#000000"

    with open("./data/info_yamls/categories.yaml", "r", encoding="utf-8") as f:
        categories = yaml.safe_load(f) or {}

    available_cats = [c for c in categories.keys() if c in df.columns]
    if len(available_cats) < 2:
        tab.info("At least two categorical columns are required for a heatmap.")
        return

    # --- Category pickers ---
    c1, c2 = tab.columns([1, 1])
    with c1:
        default_a_idx = available_cats.index("analysis method") if "analysis method" in available_cats else \
            available_cats[0]
        cat_a = st.selectbox("Rows (Category A)", available_cats, index=default_a_idx, key="heatmap_cat_a")
    with c2:
        default_b_idx = available_cats.index("measurement modality") if "measurement modality" in available_cats else \
            available_cats[-1]
        cat_b = st.selectbox("Columns (Category B)", available_cats, index=default_b_idx, key="heatmap_cat_b")

    if cat_a == cat_b:
        tab.info("Please select two different categories.")
        return

    # --- Display options ---
    norm_mode = st.radio(
        "Normalization",
        ["None", "Proportion (row)", "Proportion (column)", "Proportion (overall)"],
        horizontal=True,
        key="heatmap_norm"
        )

    # default decimals: 0 when raw counts, 2 when any normalization is applied
    default_decimals = 0 if norm_mode == "None" else 2
    decimals = st.number_input(
        "Decimals (for displayed values)",
        min_value=0,
        max_value=4,
        value=default_decimals,
        step=1,
        key="heatmap_decimals", )
    show_values = st.checkbox("Overlay numeric values", value=True, key="heatmap_show_values")

    # --- Parse both columns into lists ---
    A_lists = df[cat_a].apply(to_list_cell)
    B_lists = df[cat_b].apply(to_list_cell)

    # --- Co-occurrence at study level when DOI available ---
    pairs = []
    unique_id_col = "doi" if "doi" in df.columns else None
    a_counter = Counter()
    b_counter = Counter()

    if unique_id_col:
        for _, sub in df.groupby(unique_id_col):
            A_sub = sub[cat_a].apply(to_list_cell)
            B_sub = sub[cat_b].apply(to_list_cell)

            a_labels = {a for lst in A_sub for a in lst}
            b_labels = {b for lst in B_sub for b in lst}

            a_counter.update(a_labels)
            b_counter.update(b_labels)

            if a_labels and b_labels:
                pairs.extend((a, b) for a in a_labels for b in b_labels)
    else:
        for a_vals, b_vals in zip(A_lists, B_lists):
            a_set = set(a_vals)
            b_set = set(b_vals)

            a_counter.update(a_set)
            b_counter.update(b_set)

            if a_set and b_set:
                pairs.extend((a, b) for a in a_set for b in b_set)

    if not pairs:
        tab.warning("No co-occurrence found between the selected categories.")
        return

    # --- Co-occurrence counts & label ordering (YAML first, then drop all-zero rows/cols) ---
    pairs_df = pd.DataFrame(pairs, columns=["A", "B"])
    counts = pairs_df.value_counts().rename("count").reset_index()

    # base order from YAML
    a_order_yaml = list(categories.get(cat_a, {}).keys())
    b_order_yaml = list(categories.get(cat_b, {}).keys())

    # fall back to data-driven order if YAML is empty for a category
    if not a_order_yaml:
        a_order_yaml = sorted(counts["A"].unique())
    if not b_order_yaml:
        b_order_yaml = sorted(counts["B"].unique())

    # restrict to labels defined in YAML-based order
    counts = counts[counts["A"].isin(a_order_yaml) & counts["B"].isin(b_order_yaml)]

    # build full matrix in YAML order
    idx = pd.MultiIndex.from_product([a_order_yaml, b_order_yaml], names=["A", "B"])
    mat = counts.set_index(["A", "B"]).reindex(idx, fill_value=0)["count"].unstack("B")

    # drop rows/columns that are all zeros, but preserve YAML order explicitly
    row_sums = mat.sum(axis=1)
    col_sums = mat.sum(axis=0)

    nonzero_a_labels = [lab for lab in a_order_yaml if row_sums.get(lab, 0) > 0]
    nonzero_b_labels = [lab for lab in b_order_yaml if col_sums.get(lab, 0) > 0]

    if not nonzero_a_labels or not nonzero_b_labels:
        tab.warning("All categories are zero after filtering; nothing to display.")
        return

    # reindex in YAML order, restricted to non-zero labels
    mat = mat.loc[nonzero_a_labels, nonzero_b_labels]
    a_order = nonzero_a_labels
    b_order = nonzero_b_labels

    # --- Normalization ---
    mat_display = mat.copy()
    if norm_mode == "Proportion (row)":
        mat_display = mat_display.div(mat_display.sum(axis=1), axis=0).fillna(0)
    elif norm_mode == "Proportion (column)":
        mat_display = mat_display.div(mat_display.sum(axis=0), axis=1).fillna(0)
    elif norm_mode == "Proportion (overall)":
        total = mat_display.values.sum()
        mat_display = mat_display / total if total > 0 else mat_display

    # --- Long format for Altair ---
    heat_df = (mat_display.reset_index().melt(id_vars="A", var_name="B", value_name="value").merge(
        mat.reset_index().melt(id_vars="A", var_name="B", value_name="count"), on=["A", "B"], how="left", ))

    heat_df["display_text"] = heat_df.apply(
        lambda row: "-" if row["count"] == 0 else f"{row['value']:.{decimals}f}", axis=1, )

    # --- Base heatmap ---
    start_color = "#f7eedf"
    end_color = "#F28C8C" if is_dark_color(figure_background_color) else "#9FCBE1"
    vmin = 0
    vmax = float(heat_df["value"].max()) if not heat_df.empty else 1.0
    color_title = "count" if norm_mode == "None" else {norm_mode.lower()}
    base_rect = (alt.Chart(heat_df).mark_rect().encode(
        x=alt.X(
            "B:N", title=cat_b, sort=None, scale=alt.Scale(domain=b_order), axis=alt.Axis(
                labelLimit=0, labelAngle=315,  # <— rotate labels by 45°
                labelAlign="right",  # <— makes angled labels look nicer
                labelBaseline="top", labelOverlap=False, labelColor=font_color, ), ), y=alt.Y(
            "A:N",
            title=cat_a,
            sort=None,
            scale=alt.Scale(domain=a_order),
            axis=alt.Axis(labelLimit=0, labelAngle=0, labelOverlap=False, labelColor=font_color, ), ), color=alt.Color(
            "value:Q",
            title=color_title,
            scale=alt.Scale(domain=[vmin, vmax], range=[start_color, end_color]),
            legend=alt.Legend(
                titleAlign="center", titleAnchor="middle", values=[vmin, (vmin + vmax) / 2, vmax],  # <--- custom ticks
                format=".0f" if norm_mode == "None" else f".{decimals}f", ), ),

        tooltip=[alt.Tooltip("A:N", title=cat_a), alt.Tooltip("B:N", title=cat_b),
                 alt.Tooltip("count:Q", title="Raw count", format=",.0f"),
                 alt.Tooltip("value:Q", title="Value", format=f".{decimals}f"), ], ).properties(height=400, width=800))

    chart = base_rect

    # --- Optional numeric values overlay ---
    if show_values:
        text_layer = (alt.Chart(heat_df).mark_text(baseline="middle", align="center", color='black').encode(
            x=alt.X("B:N", sort=None, scale=alt.Scale(domain=b_order)),
            y=alt.Y("A:N", sort=None, scale=alt.Scale(domain=a_order)),
            text=alt.Text("display_text:N"), ))
        chart = chart + text_layer
    # Make exported chart visually stable & theme-aware
    chart = (chart.configure_view(
        stroke=None,  # no outer border
        fill=figure_background_color  # match Streamlit theme background
        ).configure_axis(
        labelColor=font_color, titleColor=font_color
        ).configure_legend(
        labelColor=font_color, titleColor=font_color
        ).properties(
        height=400, width=800
        ))
    # --- Render in Streamlit ---
    tab.altair_chart(chart, use_container_width=True)

    # --- Figure caption & description ---
    tab.caption(
        "Cell color intensity increases with the value in each cell; darker cells indicate stronger co-occurrence."
        )

    if norm_mode == "None":
        norm_desc = ("Cell values show raw co-occurrence counts between the row and column categories.")
    elif norm_mode == "Proportion (row)":
        norm_desc = ("Cell values are normalized within each row between zero and one; each cell shows the "
                     "proportion of observations in the row that also fall into the column category.")
    elif norm_mode == "Proportion (column)":
        norm_desc = ("Cell values are normalized within each column between zero and one; each cell shows the "
                     "proportion of observations in the column that also fall into the row category.")
    else:  # "Proportion (overall)"
        norm_desc = ("Cell values are normalized by the grand total so that all cells together sum to one; "
                     "each cell shows the overall proportion of observations assigned to that combination.")

    tab.markdown(
        f"Heatmap of studies across the categorical dimensions **{cat_a}** (rows) and "
        f"**{cat_b}** (columns). {norm_desc} Rows and columns that contain only zeros are omitted."
        )

    # --- Download matrix as displayed ---
    csv_data = mat_display.to_csv(index=True)
    st.download_button(
        "📥 Download Heatmap Matrix (CSV)",
        data=csv_data,
        file_name=f"heatmap_{cat_a}_x_{cat_b}_{norm_mode.lower().replace(' ', '_')}.csv",
        mime="text/csv", )
