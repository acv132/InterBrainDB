from collections import defaultdict
from itertools import combinations, product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

from data.config import ColorMap
from plotting.plot_utils import *


def generate_category_counts_figure(df, tab):
    """
    Generate a horizontal bar plot showing the counts for each category in the DataFrame.
    Each one-hot encoded column is grouped by its category and summed.
    """
    # Check if DataFrame has any rows
    if df is None or df.empty:
        tab.warning("No data found; make sure that at least some data is passing the selected filters.")
        return None

    # Theme-aware font color
    figure_background_color = st.get_option('theme.backgroundColor')
    font_color = "#ffffff" if is_dark_color(figure_background_color) else "#000000"
    bar_color = st.get_option('theme.primaryColor')

    # Set the font properties for the figure
    font_properties = FontProperties(size=14)
    plt.rcParams['font.size'] = font_properties.get_size()

    # Decode one-hot columns
    prefix = "_"
    df = decode_one_hot(df, prefix)

    # Load full category list
    with open("./data/info_yamls/categories.yaml", 'r', encoding='utf-8') as f:
        category_definitions = yaml.safe_load(f)

    # Get all one-hot column groups (e.g. "measurement modality", "paradigm", ...)
    base_categories = list(category_definitions.keys())

    # Get all one-hot encoded column names
    all_columns = df.columns

    # Identify one-hot columns for each category
    category_columns = {
        cat: [col for col in all_columns if col.startswith(f"{cat}{prefix}")]
        for cat in base_categories
    }

    # Clean up: only keep categories with at least 1 matching one-hot column
    category_columns = {k: v for k, v in category_columns.items() if v}

    # Determine layout height
    # Example input
    n_categories = len(category_columns)
    max_labels = max(len(cols) for cols in category_columns.values())

    # User chooses layout style
    layout_option = st.radio(
        "Choose plot layout:", ["Auto", "Single Column", "Grid (3 columns)"], horizontal=True
        )

    # Determine layout
    if layout_option == "Single Column":
        n_cols = 1
    elif layout_option == "Grid (3 columns)":
        n_cols = 3
    else:  # Auto layout
        n_cols = min(4, n_categories)
    n_rows = int(-(-n_categories / n_cols))

    # Set dynamic height
    subplot_height = max(4, max_labels * 0.4)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, subplot_height * n_rows))
    axes = axes.flatten()

    # If there's only one category, ensure axes is a list
    if len(category_columns) == 1:
        axes = [axes]

    # Create bar plots per category
    for ax, (cat, columns) in zip(axes, category_columns.items()):
        counts = df[columns].sum().sort_values(ascending=True)
        counts.index = [col.replace(f"{cat}{prefix}", "") for col in counts.index]

        counts.plot(kind='barh', ax=ax, color=bar_color)
        # tab.bar_chart(counts, horizontal=True)
        ax.set_title(f"{cat} counts", color=font_color)
        ax.set_xlabel("Number of Studies", color=font_color)
        ax.set_ylabel(cat, color=font_color)
        ax.tick_params(axis='x', colors=font_color)
        ax.tick_params(axis='y', colors=font_color)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")

    fig.tight_layout()
    return fig


def generate_interaction_figure(df, tab):
    """Generate a confusion matrix-like figure showing the interaction conditions across studies."""

    # Check if DataFrame has any rows
    if len(df) == 0:
        tab.warning("No data found; make sure that at least some data is passing the selected filters.")
        return None

    ###############
    # Configs
    ###############

    # get defined category labels for CM and color legend
    yaml_file = "./data/info_yamls/categories.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        categories = yaml.safe_load(f)

    default_scenario_order = categories["interaction scenario"].keys()
    default_manipulative_order = categories["interaction manipulative"].keys()
    default_modalities_order = categories["measurement modality"].keys()

    # define some figure specs
    row_spacing = 1.5  # Increase spacing between rows
    col_spacing = 1.5  # Increase spacing between columns
    # Base width for line thickness
    base_width = 1.5
    base_curvature = .4
    line_styles = ['-', '--', ':', '-.']  # Solid, dashed, dash-dot, dotted

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
    df = decode_one_hot(df, prefix)
    number_studies = len(df)

    scenario_columns = [col for col in df.columns if col.startswith(f"interaction scenario{prefix}")]
    manipulative_columns = [col for col in df.columns if col.startswith(f"interaction manipulative{prefix}")]
    modality_columns = [col for col in df.columns if col.startswith(f"measurement modality{prefix}")]
    df['interaction scenario'] = df[scenario_columns].apply(
        lambda row: [col.replace(f"interaction scenario{prefix}", "") for col, val in row.items() if val], axis=1
        )
    df['interaction manipulative'] = df[manipulative_columns].apply(
        lambda row: [col.replace(f"interaction manipulative{prefix}", "") for col, val in row.items() if val], axis=1
        )
    df['measurement modality'] = df[modality_columns].apply(
        lambda row: [col.replace(f"measurement modality{prefix}", "") for col, val in row.items() if val], axis=1
        )
    df_exploded = df.explode('measurement modality')
    df_exploded = df_exploded.explode('interaction manipulative')
    df_exploded = df_exploded.explode('interaction scenario')
    pivot_table = df_exploded.groupby(
        ['measurement modality', 'interaction scenario', 'interaction manipulative']
        ).size().unstack(fill_value=0)
    cross_section_counts = df_exploded.groupby(
        ['measurement modality', 'interaction scenario', 'interaction manipulative']
        ).size().reset_index(name='count')
    scenario_contained = [col.replace("interaction scenario_", "") for col in scenario_columns]
    manipulative_contained = [col.replace("interaction manipulative_", "") for col in manipulative_columns]
    modalities_contained = [col.replace("measurement modality_", "") for col in modality_columns]

    scenario_order = [scenario for scenario in default_scenario_order if scenario in scenario_contained]
    manipulative_order = [manipulative for manipulative in default_manipulative_order if manipulative in
                          manipulative_contained]
    modalities_order = [modality for modality in default_modalities_order if modality in modalities_contained]

    # prepare confusion matrix rows and columns
    row_pos = np.arange(len(scenario_order)) * row_spacing
    col_pos = np.arange(len(manipulative_order)) * col_spacing
    # try:
    #     modality_colors = dict(zip(modalities_order, plt.get_cmap(colormap)(np.linspace(0, 1, len(modalities_order)))))
    # except ValueError:
    #     modality_colors = dict(zip(modalities_order, colormap))

    cooccurrance_per_study = []
    # get any accounts of co-existence of conditions in one study (ID)
    for study_id, group in df.groupby('article'):
        # Get all scenarios, manipulatives, and modalities used in this study
        scenarios = [col.replace("interaction scenario_", "") for col in scenario_columns if group[col].iloc[0] == 1]
        manipulatives = [col.replace("interaction manipulative_", "") for col in manipulative_columns if
                         group[col].iloc[0] == 1]
        modalities = [col.replace("measurement modality_", "") for col in modality_columns if group[col].iloc[0] == 1]

        # get coordinates of each scenario and manipulative for each modality
        for modality in modalities:
            y_coor = row_pos[[scenario_order.index(s) for s in scenarios]]
            x_coor = col_pos[[manipulative_order.index(m) for m in manipulatives]]
            cooccurrance_per_study.append((x_coor, y_coor, modality))

    # count unique connection lines
    connection_counts = defaultdict(int)
    for x_coor, y_coor, modality in cooccurrance_per_study:
        # connection lines are valid to count if they are in fact a line (not a dot)
        if len(x_coor) > 1 or len(y_coor) > 1:
            # get color according to modality
            color = modality_colors.get(modality, 'black')

            coordinates = list(product(x_coor, y_coor))
            connections = list(combinations(coordinates, 2))
            normalized_connections = [tuple(sorted(el)) for el in connections]
            for element in normalized_connections:
                # if connection is not diagnoal, simply count it
                if is_horizontal_or_vertical(element[0][0], element[0][1], element[1][0], element[1][1]):
                    connection_counts[(element, modality)] += 1
                # if it is diagnoal, form corresponding triangle connections and count them if not already counted
                else:
                    new_connections = convert_to_horizontal_and_vertical_connections(element)
                    for new_element in new_connections:
                        normalized_conn = tuple(sorted(new_element))
                        if normalized_conn not in normalized_connections:
                            connection_counts[(normalized_conn, modality)] += 1

    connection_data = []
    for ((start, end), modality), count in connection_counts.items():
        start_scenario = get_scenario_or_manipulative(
            start[1],
            'y',
            scenario_order,
            manipulative_order,
            row_pos,
            col_pos
            )
        start_manipulative = get_scenario_or_manipulative(
            start[0], 'x', scenario_order, manipulative_order, row_pos, col_pos
            )
        end_scenario = get_scenario_or_manipulative(end[1], 'y', scenario_order, manipulative_order, row_pos, col_pos)
        end_manipulative = get_scenario_or_manipulative(
            end[0],
            'x',
            scenario_order,
            manipulative_order,
            row_pos,
            col_pos
            )

        connection_data.append(
            (start, end, f"{start_scenario} - {start_manipulative}", f"{end_scenario} - {end_manipulative}", modality,
             count,)
            )

    # Create updated DataFrame
    connection_df = pd.DataFrame(
        connection_data, columns=["start", "end", "start_pairing", "end_pairing", "modality", "count"]
        )
    unique_counts = sorted(connection_df['count'].unique())
    count_to_style = {count: line_styles[i % len(line_styles)] for i, count in enumerate(unique_counts)}

    ###############
    # Prepare the plot
    ###############
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

        # Draw curved lines using Bezier path
        control_point = (0., 0.)
        if start[0] == end[0]:
            control_point = ((start[0] + end[0]) / 2 + curvature, (start[1] + end[1]) / 2)
        elif start[1] == end[1]:
            control_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + curvature)
        vertices = np.array([start, control_point, end])
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        bezier_path = Path(vertices, codes)

        # Add Bezier path to the plot
        patch = PathPatch(bezier_path, color=color, lw=line_width, fill=False, linestyle=line_style, zorder=1)
        ax.add_patch(patch)

    # Draw pie charts for each scenario and manipulative combination
    for i, scenario in enumerate(scenario_order):
        for j, manipulative in enumerate(manipulative_order):
            # Get counts for this cell
            cell_data = df_exploded[(df_exploded['interaction scenario'] == scenario) & (
                    df_exploded['interaction manipulative'] == manipulative)]
            modality_counts = cell_data['measurement modality'].value_counts()

            if not modality_counts.empty:
                # Prepare pie sizes and colors
                pie_sizes = modality_counts.values
                pie_colors = [modality_colors[mod] for mod in modality_counts.index]

                # Scale pie size
                size_factor = sum(pie_sizes) / df_exploded.shape[0]  # Scale relative to total data
                radius = pie_radius + size_factor  # Adjust radius dynamically
                ax.pie(
                    pie_sizes,
                    colors=pie_colors,
                    center=(col_pos[j], row_pos[i]),
                    radius=radius,
                    wedgeprops=dict(width=0.3), )
                ax.text(
                    col_pos[j],
                    row_pos[i],
                    str(pie_sizes.sum()),
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

    digital_im_column_indices = [manipulative_order.index(col) for col in manipulative_order if "digital" in col]

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
    ax.set_xticklabels([c.replace(" IM", '') for c in manipulative_order], rotation=45, ha='right', fontsize=14, color=font_color)
    ax.tick_params(axis='x', which='both', length=0, pad=10)
    ax.set_yticklabels(scenario_order, fontsize=14, color=font_color)
    ax.tick_params(axis='y', which='both', length=0, pad=10)

    # Add labels for axes with consistent padding
    ax.set_xlabel("Interaction Manipulative", fontsize=16, labelpad=10, color=font_color)
    ax.set_ylabel("Interaction Scenario", fontsize=16, labelpad=10, color=font_color)

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title(
        f"Confusion matrix of experimental interaction conditions ({cross_section_counts["count"].sum()} "
        f"conditions in total {number_studies} studies)", fontsize=14, pad=10, color=font_color
        )

    # Create dictionary for legend handles
    modality_handles = {label: plt.Line2D([0], [0], color=color, lw=4, linestyle='-')  # Default solid line for modality
                        for label, color in modality_colors.items()}
    style_handles = {
        f"count: {count}": plt.Line2D([0], [0], color=font_color, lw=4, linestyle=line_styles[i % len(line_styles)]) for
        i, count in enumerate(unique_counts)}
    area_handle = {
        'digital component': mpatches.Patch(color=color_rect, label='digital component')
        }
    # Get only modalities actually used
    used_modalities = connection_df['modality'].unique()
    filtered_modality_handles = {label: handle for label, handle in modality_handles.items() if
        label in used_modalities}

    # Get only counts actually used
    used_counts = connection_df['count'].unique()
    filtered_style_handles = {
        f"count: {count}": plt.Line2D([0], [0], color=font_color, lw=4, linestyle=count_to_style[count]) for count in
        used_counts}

    # Only add digital component if it's actually highlighted
    filtered_area_handle = {}
    if digital_im_column_indices:  # non-empty list
        filtered_area_handle = {
            'digital component': mpatches.Patch(color=color_rect, label='digital component')
            }

    # 1. Modality Legend (Fixed at top)
    mod_legend = ax.legend(
        handles=list(filtered_modality_handles.values()),
        labels=list(filtered_modality_handles.keys()),
        title="measurement modality",
        loc="center left",
        bbox_to_anchor=(1.0, 0.7),
        fontsize=14,
        alignment="left",
        framealpha=0,
        facecolor=facecolor_legend,
        labelcolor=font_color,
        title_fontproperties=legend_title_props, )
    plt.setp(mod_legend.get_title(), color=font_color)

    # 2. Comparison Legend (middle)
    comparison_legend = Legend(
        ax,
        handles=list(filtered_style_handles.values()),
        labels=list(filtered_style_handles.keys()),
        title="comparisons",
        loc="center left",
        bbox_to_anchor=(1.0, 0.45),
        fontsize=14,
        alignment="left",
        framealpha=0,
        facecolor=facecolor_legend,
        labelcolor=font_color,
        title_fontproperties=legend_title_props, )
    plt.setp(comparison_legend.get_title(), color=font_color)
    ax.add_artist(comparison_legend)

    # 3. Study Design Legend (bottom, only if used)
    if filtered_area_handle:
        study_design_legend = Legend(
            ax,
            handles=list(filtered_area_handle.values()),
            labels=list(filtered_area_handle.keys()),
            title="study design",
            loc="center left",
            bbox_to_anchor=(1.0, 0.30),
            fontsize=14,
            alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props, )
        plt.setp(study_design_legend.get_title(), color=font_color)
        ax.add_artist(study_design_legend)

    plt.tight_layout()

    return fig


def generate_interaction_figure_backup(df):

    if len(df) == 0:
        st.warning("No data found; make sure that at least some data is passing the selected filters.")
        return None
    else:
        ###############
        # Configs
        ###############

        # get defined category labels for CM and color legend
        yaml_file = "./data/info_yamls/categories.yaml"
        with open(yaml_file, 'r', encoding='utf-8') as f:
            categories = yaml.safe_load(f)

        default_scenario_order = categories["interaction scenario"].keys()
        default_manipulative_order = categories["interaction manipulative"].keys()
        default_modalities_order = categories["measurement modality"].keys()

        # define some figure specs
        row_spacing = 1.5  # Increase spacing between rows
        col_spacing = 1.5  # Increase spacing between columns
        # Base width for line thickness
        base_width = 1.5
        base_curvature = .4
        line_styles = ['-', '--', ':', '-.']  # Solid, dashed, dash-dot, dotted

        # Adjust radius for connections to start/end at pie edges
        pie_radius = 0.5
        colormap = ['#00928f', '#00567c', '#6d90a0', '#38b6c0', '#bad23c', '#000000', '#81c5cb', '#19bdff','#bcbec0',]
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
        df = decode_one_hot(df, prefix)
        number_studies = len(df)

        scenario_columns = [col for col in df.columns if col.startswith(f"interaction scenario{prefix}")]
        manipulative_columns = [col for col in df.columns if col.startswith(f"interaction manipulative{prefix}")]
        modality_columns = [col for col in df.columns if col.startswith(f"measurement modality{prefix}")]
        df['interaction scenario'] = df[scenario_columns].apply(
            lambda row: [col.replace(f"interaction scenario{prefix}", "") for col, val in row.items() if val], axis=1
            )
        df['interaction manipulative'] = df[manipulative_columns].apply(
            lambda row: [col.replace(f"interaction manipulative{prefix}", "") for col, val in row.items() if val], axis=1
            )
        df['measurement modality'] = df[modality_columns].apply(
            lambda row: [col.replace(f"measurement modality{prefix}", "") for col, val in row.items() if val], axis=1
            )
        df_exploded = df.explode('measurement modality')
        df_exploded = df_exploded.explode('interaction manipulative')
        df_exploded = df_exploded.explode('interaction scenario')
        pivot_table = df_exploded.groupby(
            ['measurement modality', 'interaction scenario', 'interaction manipulative']
            ).size().unstack(fill_value=0)
        cross_section_counts = df_exploded.groupby(
            ['measurement modality', 'interaction scenario', 'interaction manipulative']
            ).size().reset_index(name='count')
        scenario_contained = [col.replace("interaction scenario_", "") for col in scenario_columns]
        manipulative_contained = [col.replace("interaction manipulative_", "") for col in manipulative_columns]
        modalities_contained = [col.replace("measurement modality_", "") for col in modality_columns]

        scenario_order = [scenario for scenario in default_scenario_order if scenario in scenario_contained]
        manipulative_order = [manipulative for manipulative in default_manipulative_order if manipulative in
                              manipulative_contained]
        modalities_order = [modality for modality in default_modalities_order if modality in modalities_contained]

        # prepare confusion matrix rows and columns
        row_pos = np.arange(len(scenario_order)) * row_spacing
        col_pos = np.arange(len(manipulative_order)) * col_spacing
        # try:
        #     modality_colors = dict(zip(modalities_order, plt.get_cmap(colormap)(np.linspace(0, 1, len(modalities_order)))))
        # except ValueError:
        #     modality_colors = dict(zip(modalities_order, colormap))

        cooccurrance_per_study = []
        # get any accounts of co-existence of conditions in one study (ID)
        for study_id, group in df.groupby('article'):
            # Get all scenarios, manipulatives, and modalities used in this study
            scenarios = [col.replace("interaction scenario_", "") for col in scenario_columns if group[col].iloc[0] == 1]
            manipulatives = [col.replace("interaction manipulative_", "") for col in manipulative_columns if
                             group[col].iloc[0] == 1]
            modalities = [col.replace("measurement modality_", "") for col in modality_columns if group[col].iloc[0] == 1]

            # get coordinates of each scenario and manipulative for each modality
            for modality in modalities:
                y_coor = row_pos[[scenario_order.index(s) for s in scenarios]]
                x_coor = col_pos[[manipulative_order.index(m) for m in manipulatives]]
                cooccurrance_per_study.append((x_coor, y_coor, modality))

        # count unique connection lines
        connection_counts = defaultdict(int)
        for x_coor, y_coor, modality in cooccurrance_per_study:
            # connection lines are valid to count if they are in fact a line (not a dot)
            if len(x_coor) > 1 or len(y_coor) > 1:
                # get color according to modality
                color = modality_colors.get(modality, 'black')

                coordinates = list(product(x_coor, y_coor))
                connections = list(combinations(coordinates, 2))
                normalized_connections = [tuple(sorted(el)) for el in connections]
                for element in normalized_connections:
                    # if connection is not diagnoal, simply count it
                    if is_horizontal_or_vertical(element[0][0], element[0][1], element[1][0], element[1][1]):
                        connection_counts[(element, modality)] += 1
                    # if it is diagnoal, form corresponding triangle connections and count them if not already counted
                    else:
                        new_connections = convert_to_horizontal_and_vertical_connections(element)
                        for new_element in new_connections:
                            normalized_conn = tuple(sorted(new_element))
                            if normalized_conn not in normalized_connections:
                                connection_counts[(normalized_conn, modality)] += 1

        connection_data = []
        for ((start, end), modality), count in connection_counts.items():
            start_scenario = get_scenario_or_manipulative(
                start[1],
                'y',
                scenario_order,
                manipulative_order,
                row_pos,
                col_pos
                )
            start_manipulative = get_scenario_or_manipulative(
                start[0], 'x', scenario_order, manipulative_order, row_pos, col_pos
                )
            end_scenario = get_scenario_or_manipulative(end[1], 'y', scenario_order, manipulative_order, row_pos, col_pos)
            end_manipulative = get_scenario_or_manipulative(
                end[0],
                'x',
                scenario_order,
                manipulative_order,
                row_pos,
                col_pos
                )

            connection_data.append(
                (start, end, f"{start_scenario} - {start_manipulative}", f"{end_scenario} - {end_manipulative}", modality,
                 count,)
                )

        # Create updated DataFrame
        connection_df = pd.DataFrame(
            connection_data, columns=["start", "end", "start_pairing", "end_pairing", "modality", "count"]
            )
        unique_counts = sorted(connection_df['count'].unique())
        count_to_style = {count: line_styles[i % len(line_styles)] for i, count in enumerate(unique_counts)}

        ###############
        # Prepare the plot
        ###############
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

            # Draw curved lines using Bezier path
            control_point = (0., 0.)
            if start[0] == end[0]:
                control_point = ((start[0] + end[0]) / 2 + curvature, (start[1] + end[1]) / 2)
            elif start[1] == end[1]:
                control_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + curvature)
            vertices = np.array([start, control_point, end])
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            bezier_path = Path(vertices, codes)

            # Add Bezier path to the plot
            patch = PathPatch(bezier_path, color=color, lw=line_width, fill=False, linestyle=line_style, zorder=1)
            ax.add_patch(patch)

        # Draw pie charts for each scenario and manipulative combination
        for i, scenario in enumerate(scenario_order):
            for j, manipulative in enumerate(manipulative_order):
                # Get counts for this cell
                cell_data = df_exploded[(df_exploded['interaction scenario'] == scenario) & (
                        df_exploded['interaction manipulative'] == manipulative)]
                modality_counts = cell_data['measurement modality'].value_counts()

                if not modality_counts.empty:
                    # Prepare pie sizes and colors
                    pie_sizes = modality_counts.values
                    pie_colors = [modality_colors[mod] for mod in modality_counts.index]

                    # Scale pie size
                    size_factor = sum(pie_sizes) / df_exploded.shape[0]  # Scale relative to total data
                    radius = pie_radius + size_factor  # Adjust radius dynamically
                    ax.pie(
                        pie_sizes,
                        colors=pie_colors,
                        center=(col_pos[j], row_pos[i]),
                        radius=radius,
                        wedgeprops=dict(width=0.3), )
                    ax.text(
                        col_pos[j],
                        row_pos[i],
                        str(pie_sizes.sum()),
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

        digital_im_column_indices = [manipulative_order.index(col) for col in manipulative_order if "digital" in col]

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
        ax.set_xticklabels([c.replace(" IM", '') for c in manipulative_order], rotation=45, ha='right', fontsize=14, color=font_color)
        ax.tick_params(axis='x', which='both', length=0, pad=10)
        ax.set_yticklabels(scenario_order, fontsize=14, color=font_color)
        ax.tick_params(axis='y', which='both', length=0, pad=10)

        # Add labels for axes with consistent padding
        ax.set_xlabel("Interaction Manipulative", fontsize=16, labelpad=10, color=font_color)
        ax.set_ylabel("Interaction Scenario", fontsize=16, labelpad=10, color=font_color)

        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        ax.set_title(
            f"Confusion matrix of experimental interaction conditions ({cross_section_counts["count"].sum()} "
            f"conditions in total {number_studies} studies)", fontsize=14, pad=10, color=font_color
            )

        # Create dictionary for legend handles
        modality_handles = {label: plt.Line2D([0], [0], color=color, lw=4, linestyle='-')  # Default solid line for modality
                            for label, color in modality_colors.items()}
        style_handles = {
            f"count: {count}": plt.Line2D([0], [0], color='black', lw=4, linestyle=line_styles[i % len(line_styles)]) for
            i, count in enumerate(unique_counts)}
        area_handle = {
            'digital component': mpatches.Patch(color=color_rect, label='digital component')
            }

        # Add the first legend for measurement modality
        mod_legend = ax.legend(
            handles=list(modality_handles.values()),
            labels=list(modality_handles.keys()),
            title="measurement modality",
            loc="center left",
            bbox_to_anchor=(1., y_axis_legend_start),
            fontsize=14,
            alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props,
            )
        plt.setp(mod_legend.get_title(), color=font_color)
        # Create and add the second legend for comparisons
        comparison_legend = Legend(
            ax,
            handles=list(style_handles.values()),
            labels=list(style_handles.keys()),
            title="comparisons",
            loc="center left",
            bbox_to_anchor=(1., y_axis_legend_start - .026*len(modality_handles)),
            fontsize=14,
            alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props,
            )
        plt.setp(comparison_legend.get_title(), color=font_color)
        ax.add_artist(comparison_legend)

        # Create and add the third legend for study designs
        study_design_legend = Legend(
            ax,
            handles=list(area_handle.values()),
            labels=list(area_handle.keys()),
            title="study design",
            loc="center left",
            bbox_to_anchor=(1., y_axis_legend_start - .026*sum([len(modality_handles), len(style_handles), 1])),
            fontsize=14, alignment="left",
            framealpha=0,
            facecolor=facecolor_legend,
            labelcolor=font_color,
            title_fontproperties=legend_title_props,
            )
        plt.setp(study_design_legend.get_title(), color=font_color)
        ax.add_artist(study_design_legend)

        plt.tight_layout()

        return fig


def generate_interaction_figure_streamlit(df, tab):
    """
    Streamlit-compatible alternative: stacked bar matrix for scenario × manipulative, colored by modality.
    Each cell is a stacked bar showing the count per modality. All elements are interactive and aligned.
    """
    import plotly.graph_objects as go
    import yaml
    import numpy as np

    if len(df) == 0:
        tab.warning("No data found; make sure that at least some data is passing the selected filters.")
        return None

    # Load categories
    yaml_file = "./data/info_yamls/categories.yaml"
    with open(yaml_file, 'r', encoding='utf-8') as f:
        categories = yaml.safe_load(f)

    default_scenario_order = list(categories["interaction scenario"].keys())
    default_manipulative_order = list(categories["interaction manipulative"].keys())
    default_modalities_order = list(categories["measurement modality"].keys())
    colormap = ["#00928f", "#00567c", "#6d90a0", "#38b6c0", "#bad23c", "#000000", "#81c5cb", "#19bdff", "#bcbec0"]
    modality_colors = dict(zip(default_modalities_order, colormap))

    # Prepare data (decode one-hot, explode, etc.)
    prefix = "_"
    df = decode_one_hot(df, prefix)
    scenario_columns = [col for col in df.columns if col.startswith(f"interaction scenario{prefix}")]
    manipulative_columns = [col for col in df.columns if col.startswith(f"interaction manipulative{prefix}")]
    modality_columns = [col for col in df.columns if col.startswith(f"measurement modality{prefix}")]
    df['interaction scenario'] = df[scenario_columns].apply(
        lambda row: [col.replace(f"interaction scenario{prefix}", "") for col, val in row.items() if val], axis=1
    )
    df['interaction manipulative'] = df[manipulative_columns].apply(
        lambda row: [col.replace(f"interaction manipulative{prefix}", "") for col, val in row.items() if val], axis=1
    )
    df['measurement modality'] = df[modality_columns].apply(
        lambda row: [col.replace(f"measurement modality{prefix}", "") for col, val in row.items() if val], axis=1
    )
    df_exploded = df.explode('measurement modality')
    df_exploded = df_exploded.explode('interaction manipulative')
    df_exploded = df_exploded.explode('interaction scenario')

    # Pivot table for stacked bar matrix
    pivot = df_exploded.groupby([
        'interaction scenario', 'interaction manipulative', 'measurement modality']
    ).size().reset_index(name='count')

    # Create a matrix for each modality
    data_matrix = {}
    for modality in default_modalities_order:
        mat = np.zeros((len(default_scenario_order), len(default_manipulative_order)))
        for _, row in pivot[pivot['measurement modality'] == modality].iterrows():
            i = default_scenario_order.index(row['interaction scenario'])
            j = default_manipulative_order.index(row['interaction manipulative'])
            mat[i, j] = row['count']
        data_matrix[modality] = mat

    # Build stacked bar traces
    traces = []
    for m_idx, modality in enumerate(default_modalities_order):
        traces.append(go.Bar(
            x=[f"{manip}" for manip in default_manipulative_order]*len(default_scenario_order),
            y=data_matrix[modality].flatten(),
            name=modality,
            marker_color=modality_colors[modality],
            offsetgroup=0,
            customdata=[(default_scenario_order[i//len(default_manipulative_order)], default_manipulative_order[i%len(default_manipulative_order)]) for i in range(len(default_scenario_order)*len(default_manipulative_order))],
            hovertemplate="Scenario: %{customdata[0]}<br>Manipulative: %{customdata[1]}<br>Modality: " + modality + "<br>Count: %{y}<extra></extra>",
        ))

    # Layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            tickvals=[f"{manip}" for manip in default_manipulative_order]*len(default_scenario_order),
            ticktext=[f"{manip}" for manip in default_manipulative_order]*len(default_scenario_order),
            title="Interaction Manipulative",
        ),
        yaxis=dict(title="Count"),
        title="Stacked Bar Matrix: Scenario × Manipulative × Modality",
        legend_title="Measurement Modality",
        height=600,
        width=1200,
    )
    # Add scenario labels as annotations
    for i, scenario in enumerate(default_scenario_order):
        fig.add_annotation(
            x=-0.5 + i*len(default_manipulative_order),
            y=0,
            text=f"<b>{scenario}</b>",
            showarrow=False,
            yshift=-40,
            xanchor="left",
            font=dict(size=14, color="black")
        )
    tab.plotly_chart(fig, use_container_width=True)
    return fig
