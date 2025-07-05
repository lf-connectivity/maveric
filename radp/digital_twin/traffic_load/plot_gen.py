# maveric/apps/traffic_load/plot_gen.py

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # For custom legend entries

try:
    from scipy.spatial import Voronoi, voronoi_plot_2d
except ImportError:
    print("Warning: scipy.spatial.Voronoi not found. Voronoi plotting will be disabled.")
    Voronoi, voronoi_plot_2d = None, None

try:
    from shapely.geometry import Polygon
except ImportError:
    print("Error: Shapely library not found. Please install it for plotting spatial layouts: pip install Shapely")
    sys.exit(1)

# Attempt to import from RADP library for constants.
try:
    from radp.digital_twin.utils import constants as c
except ImportError:
    print("Warning: Failed to import RADP constants for plot_gen.py.")
    print("Ensure RADP_ROOT is in PYTHONPATH. Using fallback definitions.")
    class c: # Define fallback constants ONLY IF import fails
        CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"
        LAT = "lat"; LON = "lon"

logger = logging.getLogger(__name__)

class TrafficDataVisualizer:
    """
    Visualizes traffic demand data, including UE distributions, cell towers,
    and spatial area types.
    """

    def __init__(self):
        """Initializes the TrafficDataVisualizer."""
        pass

    def generate_tick_visualization(self,
                                    tick_to_plot: int,
                                    ue_data_for_tick: pd.DataFrame,
                                    topology_df: pd.DataFrame,
                                    spatial_layout: List[Dict[str, Any]],
                                    spatial_params_for_colors: Dict[str, Any], # For consistent coloring of types
                                    plot_output_path_template: str = "./plots/ue_distribution_tick_{tick}.png",
                                    serving_cell_data_for_tick: Optional[pd.DataFrame] = None):
        """
        Generates and saves a plot visualizing UE distribution, cell towers,
        and spatial types for a specific simulation tick.

        Args:
            tick_to_plot: The simulation tick to visualize.
            ue_data_for_tick: DataFrame with UE data for the specified tick.
                              Must contain c.LAT, c.LON. 'ue_id' is used if serving_cell_data is present.
            topology_df: DataFrame with network topology (c.CELL_LAT, c.CELL_LON, c.CELL_ID).
            spatial_layout: List of dicts defining spatial cells (from TrafficDemandModel),
                            each with 'bounds' (polygon coordinates) and 'type'.
            spatial_params_for_colors: Dict containing "types" list from spatial_params.json,
                                       used to assign consistent colors to area types.
            plot_output_path_template: Template for the output plot file path.
                                       The '{tick}' placeholder will be replaced.
            serving_cell_data_for_tick: Optional DataFrame with serving cell info for UEs.
                                        If provided, UEs are colored by serving cell.
                                        Must contain 'ue_id' and 'serving_cell_id'.
        """
        logger.info(f"Generating plot for tick {tick_to_plot}...")

        # --- Constants ---
        COL_LAT = getattr(c, 'LAT', 'lat')
        COL_LON = getattr(c, 'LON', 'lon')
        COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
        COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')

        # --- Basic Checks ---
        if ue_data_for_tick is None or ue_data_for_tick.empty:
            logger.warning(f"No UE data provided for tick {tick_to_plot}. Skipping plot.")
            return
        if not {COL_LAT, COL_LON}.issubset(ue_data_for_tick.columns):
             raise ValueError(f"UE data must include '{COL_LAT}' and '{COL_LON}'.")
        if not {COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID}.issubset(topology_df.columns):
             raise ValueError(f"Topology data must include '{COL_CELL_LAT}', '{COL_CELL_LON}', and '{COL_CELL_ID}'.")


        fig, ax = plt.subplots(figsize=(14, 11))

        # --- 1. Plot Shaded Spatial Area Polygons ---
        plotted_space_labels = set()
        spatial_legend_handles = []
        
        # Define colors for spatial types based on the order in spatial_params_for_colors
        space_types_defined = spatial_params_for_colors.get("types", [])
        if not space_types_defined:
            logger.warning("No 'types' defined in spatial_params_for_colors. Spatial areas may not be colored correctly.")
            # Fallback: try to get types from the layout itself, order might be inconsistent
            if spatial_layout:
                space_types_defined = sorted(list(set(cell['type'] for cell in spatial_layout)))

        cmap_spaces = plt.get_cmap('tab10', max(1, len(space_types_defined))) # Ensure at least 1 color
        space_colors = {stype: cmap_spaces(i) for i, stype in enumerate(space_types_defined)}

        if spatial_layout:
            logger.debug(f"Plotting {len(spatial_layout)} spatial cell areas...")
            for cell_info in spatial_layout:
                try:
                    polygon = Polygon(cell_info["bounds"])
                    space_type = cell_info["type"]
                    color = space_colors.get(space_type, 'lightgrey') # Default color if type not in map
                    ax.fill(*polygon.exterior.xy, color=color, alpha=0.15, zorder=1)
                    if space_type not in plotted_space_labels:
                        label_text = space_type.replace("_", " ").title()
                        spatial_legend_handles.append(Patch(facecolor=color, alpha=0.3, label=label_text))
                        plotted_space_labels.add(space_type)
                except Exception as e:
                    logger.warning(f"Could not plot spatial cell {cell_info.get('cell_id', 'N/A')}: {e}")
        else:
            logger.warning("No spatial_layout provided. Skipping background shading.")

        # --- 2. Plot Voronoi Lines (Optional) ---
        if Voronoi and voronoi_plot_2d and len(topology_df) >= 4:
            points = topology_df[[COL_CELL_LON, COL_CELL_LAT]].values
            try:
                vor = Voronoi(points)
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="grey", lw=0.5, line_style=':', alpha=0.5, point_size=0, zorder=2)
            except Exception as e:
                logger.warning(f"Voronoi plot generation failed: {e}")
        elif len(topology_df) < 4 and Voronoi:
             logger.info("Skipping Voronoi plot: Less than 4 sites in topology.")


        # --- 3. Plot Cell Towers ---
        tower_handle = ax.scatter(
            topology_df[COL_CELL_LON], topology_df[COL_CELL_LAT],
            marker="^", c="black", edgecolors='white', label="Cell Towers", s=100, zorder=10
        )

        # --- 4. Plot UEs ---
        ue_plot_data = ue_data_for_tick.copy()
        ue_legend_handles = []
        plotted_ue_labels = set()
        no_serve_handle = None

        if serving_cell_data_for_tick is not None and not serving_cell_data_for_tick.empty:
            if not {'ue_id', 'serving_cell_id'}.issubset(serving_cell_data_for_tick.columns):
                logger.warning("serving_cell_data_for_tick missing 'ue_id' or 'serving_cell_id'. Plotting all UEs with default color.")
                ax.scatter(ue_plot_data[COL_LON], ue_plot_data[COL_LAT], color='blue', alpha=0.6, s=15, zorder=5, label="UEs")
                ue_legend_handles.append(plt.Line2D([0], [0], marker='o', color='blue', linestyle='', ms=5, label="UEs"))
            else:
                # Ensure 'ue_id' is in ue_plot_data if we are merging
                if 'ue_id' not in ue_plot_data.columns:
                    logger.warning("'ue_id' column missing in ue_data_for_tick. Cannot merge serving cell data. Plotting with default color.")
                    ax.scatter(ue_plot_data[COL_LON], ue_plot_data[COL_LAT], color='blue', alpha=0.6, s=15, zorder=5, label="UEs")
                    ue_legend_handles.append(plt.Line2D([0], [0], marker='o', color='blue', linestyle='', ms=5, label="UEs"))
                else:
                    ue_plot_data = pd.merge(ue_plot_data, serving_cell_data_for_tick[['ue_id', 'serving_cell_id']], on="ue_id", how="left")
                    unique_serving_cells = sorted(ue_plot_data["serving_cell_id"].dropna().unique())
                    cmap_ues = plt.get_cmap("viridis", max(1, len(unique_serving_cells)))

                    for i, cell_id_val in enumerate(unique_serving_cells):
                        cell_ues = ue_plot_data[ue_plot_data["serving_cell_id"] == cell_id_val]
                        if not cell_ues.empty:
                            color = cmap_ues(i / max(1, len(unique_serving_cells) -1e-9)) # Normalize index
                            ax.scatter(cell_ues[COL_LON], cell_ues[COL_LAT], color=color, alpha=0.7, s=10, zorder=5)
                            label = f"UEs ({cell_id_val})"
                            if label not in plotted_ue_labels:
                                ue_legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, linestyle='', ms=5, label=label))
                                plotted_ue_labels.add(label)
                    
                    ues_no_serve = ue_plot_data[ue_plot_data["serving_cell_id"].isna()]
                    if not ues_no_serve.empty:
                        ax.scatter(ues_no_serve[COL_LON], ues_no_serve[COL_LAT], c='red', marker='x', s=15, label="UEs (No Serving)", zorder=6)
                        no_serve_handle = plt.Line2D([0],[0], marker='x', color='red', linestyle='', ms=6, label="UEs (No Serving)")
        else: # No serving cell data, plot all UEs with one color
            ax.scatter(ue_plot_data[COL_LON], ue_plot_data[COL_LAT], color='blue', alpha=0.6, s=15, zorder=5, label="UEs")
            ue_legend_handles.append(plt.Line2D([0], [0], marker='o', color='blue', linestyle='', ms=5, label="UEs"))


        # --- 5. Setup Plot Appearance & Combined Legend ---
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"UE Distribution, Serving Cells, and Spatial Types (Tick {tick_to_plot})")

        all_handles = [tower_handle] + spatial_legend_handles + ue_legend_handles
        if no_serve_handle:
            all_handles.append(no_serve_handle)
        
        if all_handles:
            ax.legend(handles=all_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="Legend", frameon=True)
        
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.subplots_adjust(right=0.75) # Adjust right margin to fit legend

        # --- 6. Save Plot ---
        output_filename = plot_output_path_template.format(tick=tick_to_plot)
        output_dir = os.path.dirname(output_filename)
        if output_dir: # Ensure directory exists if it's not the current one
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            plt.savefig(output_filename, bbox_inches='tight')
            logger.info(f"Plot saved: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot {output_filename}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running basic TrafficDataVisualizer example...")

    # Create dummy data for example
    example_ue_data = pd.DataFrame({
        getattr(c, 'LON', 'lon'): np.random.uniform(-73.99, -73.96, 50),
        getattr(c, 'LAT', 'lat'): np.random.uniform(40.71, 40.74, 50),
        'ue_id': np.arange(50) # For serving cell example
    })
    example_topology = pd.DataFrame({
        getattr(c, 'CELL_LON', 'cell_lon'): [-73.985, -73.975, -73.965, -73.970],
        getattr(c, 'CELL_LAT', 'cell_lat'): [40.73, 40.72, 40.735, 40.715],
        getattr(c, 'CELL_ID', 'cell_id'): ['cell_A', 'cell_B', 'cell_C', 'cell_D']
    })
    # Dummy spatial layout (simplified)
    example_spatial_layout = [
        {"bounds": [(-74.0, 40.7), (-73.95, 40.7), (-73.95, 40.75), (-74.0, 40.75), (-74.0, 40.7)], "type": "residential", "cell_id":0},
        {"bounds": [(-73.95, 40.7), (-73.90, 40.7), (-73.90, 40.75), (-73.95, 40.75), (-73.95, 40.7)], "type": "commercial", "cell_id":1}
    ]
    example_spatial_params_colors = {"types": ["residential", "commercial"]}

    # Dummy serving cell data
    example_serving_data = pd.DataFrame({
        'ue_id': np.arange(50),
        'serving_cell_id': np.random.choice(['cell_A', 'cell_B', 'cell_C', 'cell_D', None], 50)
    })


    visualizer = TrafficDataVisualizer()
    visualizer.generate_tick_visualization(
        tick_to_plot=0,
        ue_data_for_tick=example_ue_data,
        topology_df=example_topology,
        spatial_layout=example_spatial_layout,
        spatial_params_for_colors=example_spatial_params_colors,
        plot_output_path_template="./example_plot_tick_{tick}.png",
        serving_cell_data_for_tick=example_serving_data
    )
    logger.info("Example plot generation complete (if successful). Check for example_plot_tick_0.png")

