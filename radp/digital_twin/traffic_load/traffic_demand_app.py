# maveric/apps/traffic_load/traffic_demand_app.py

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any

import pandas as pd
import numpy as np

try:
    from traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c
except ImportError:
    print("Error: Could not import TrafficDemandModel or RADP constants.")
    sys.exit(1)

try:
    from config_gen import ScenarioConfigurationGenerator
    from plot_gen import TrafficDataVisualizer
except ImportError:
    print("Error: Could not import ScenarioConfigurationGenerator or TrafficDataVisualizer.")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    The orchestration of traffic demand generation, configuration, and visualization over multiple days is performed by this function.
    """

    logger.info("--- Starting Traffic Demand Application ---")

    # --- 0. Initialization of Generators and Models ---
    config_generator = ScenarioConfigurationGenerator()
    traffic_model = TrafficDemandModel()
    visualizer = TrafficDataVisualizer()

    # --- 1. Loading or Generation of Topology and Initial Configuration ---
    # The topology and initial configuration are loaded or generated once for the entire multi-day simulation.
    topology_df = None
    if args.generate_config_flag:
        # If the flag is set, new topology and initial configuration files are generated.
        logger.info("Generating new topology and initial configuration files...")
        default_cfg_params = {getattr(c, 'CELL_EL_DEG', 'cell_el_deg'): args.default_cell_tilt}
        topology_df, _ = config_generator.generate_topology_and_config_files(
            num_sites=args.num_sites, cells_per_site=args.cells_per_site,
            lat_range=tuple(args.lat_range), lon_range=tuple(args.lon_range),
            default_config_params=default_cfg_params, output_topology_path=args.topology_csv,
            output_config_path=args.config_csv
        )
    else:
        # If the flag is not set, the topology is loaded from the specified CSV file.
        logger.info(f"Loading topology from {args.topology_csv}")
        try:
            topology_df = pd.read_csv(args.topology_csv)
        except Exception as e:
            logger.error(f"Error loading topology: {e}. Exiting.")
            sys.exit(1)
            
    if topology_df is None or topology_df.empty:
        logger.error("Topology data is not available. Exiting.")
        sys.exit(1)

    # --- 2. Loading of Spatial and Temporal Parameters ---
    # The spatial and temporal parameters are loaded from the provided JSON files.
    try:
        with open(args.spatial_params_json, "r") as f:
            spatial_params = json.load(f)
        with open(args.time_params_json, "r") as f:
            time_params = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON parameter files: {e}. Exiting.")
        sys.exit(1)
    
    # --- 3. Generation of a Fixed Spatial Layout (Executed Once) ---
    # The spatial layout is generated once and is utilized for all simulation days.
    logger.info("Generating a fixed spatial layout to be used for all simulation days...")
    spatial_layout = traffic_model.generate_spatial_layout(
        topology_df=topology_df,
        spatial_params=spatial_params
    )
    if not spatial_layout:
        logger.error("Failed to generate the spatial layout. Cannot proceed with UE distribution.")
        sys.exit(1)

    all_ue_dfs_for_training = []

    # --- 4. Execution of the Main Simulation Loop for N Days ---
    for day in range(args.days):
        logger.info(f"--- Starting simulation for Day {day} ---")
        
        day_output_dir = os.path.join(args.output_dir, f"Day_{day}")
        day_ue_data_dir = os.path.join(day_output_dir, os.path.basename(args.ue_data_dir))
        day_plot_dir = os.path.join(day_output_dir, os.path.basename(args.plot_dir))
        os.makedirs(day_ue_data_dir, exist_ok=True)
        os.makedirs(day_plot_dir, exist_ok=True)
        
        # The distribution of UEs for the current day is performed using the fixed layout.
        logger.info(f"Distributing UEs for Day {day} using the fixed layout...")
        ue_data_per_tick_dict = traffic_model.distribute_ues_over_time(
            spatial_layout=spatial_layout,
            time_params=time_params,
            num_ues_per_tick=args.num_ues
        )

        if not ue_data_per_tick_dict:
            logger.warning(f"UE distribution failed for Day {day}. Skipping to next day.")
            continue
            
        logger.info(f"Saving per-tick UE data for Day {day} to: {day_ue_data_dir}")
        for tick, ue_df in ue_data_per_tick_dict.items():
            if ue_df.empty:
                continue
            
            ue_df_with_day = ue_df.copy()
            ue_df_with_day['day'] = day
            all_ue_dfs_for_training.append(ue_df_with_day)
            
            cols_to_save = {
                'mock_ue_id': ue_df.get('ue_id'),
                'lon': ue_df.get(getattr(c, 'LON', 'lon')),
                'lat': ue_df.get(getattr(c, 'LAT', 'lat')),
                'tick': ue_df.get('tick')
            }
            ue_df_to_save = pd.DataFrame({k: v for k, v in cols_to_save.items() if v is not None})
            output_filename = os.path.join(day_ue_data_dir, f"generated_ue_data_for_cco_{tick}.csv")
            ue_df_to_save.to_csv(output_filename, index=False)
        
        if args.generate_plots_flag:
            # If plot generation is enabled, visualizations for the current day are generated and saved.
            logger.info(f"Generating visualizations for Day {day} and saving to: {day_plot_dir}")
            ticks_to_plot = sorted(ue_data_per_tick_dict.keys())
            if args.plot_max_ticks > 0 and len(ticks_to_plot) > args.plot_max_ticks:
                sample_indices = np.linspace(0, len(ticks_to_plot) - 1, args.plot_max_ticks, dtype=int)
                ticks_to_plot = [ticks_to_plot[i] for i in sample_indices]

            for tick in ticks_to_plot:
                ue_df_for_plot = ue_data_per_tick_dict.get(tick)
                if ue_df_for_plot is not None and not ue_df_for_plot.empty:
                    plot_path_template = os.path.join(day_plot_dir, "ue_distribution_tick_{tick}.png")
                    visualizer.generate_tick_visualization(
                        tick_to_plot=tick, ue_data_for_tick=ue_df_for_plot,
                        topology_df=topology_df, spatial_layout=spatial_layout,
                        spatial_params_for_colors=spatial_params,
                        plot_output_path_template=plot_path_template
                    )
        logger.info(f"--- Finished simulation for Day {day} ---")

    # --- 5. Generation of Dummy Training Data ---
    if args.generate_dummy_training_flag:
        # If the flag is set and UE data is available, dummy training data is generated from all simulated days.
        if all_ue_dfs_for_training:
            combined_ue_data = pd.concat(all_ue_dfs_for_training, ignore_index=True)
            if not combined_ue_data.empty:
                logger.info("Generating dummy training data from all simulated days...")
                config_generator.generate_dummy_training_data(
                    topology_df=topology_df,
                    ue_data_all_ticks=combined_ue_data,
                    num_training_samples=args.num_training_samples,
                    output_training_data_path=args.dummy_training_csv
                )
        else:
            logger.warning("No UE data was generated across any day. Skipping dummy training data generation.")

    logger.info("--- Traffic Demand Application Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Day Traffic Demand Generation Application")
    parser.add_argument("--days", type=int, default=5, help="Number of days to simulate.")
    parser.add_argument("--generate_config_flag", action="store_true", help="Generate new topology and config files.")
    parser.add_argument("--num_sites", type=int, default=5, help="Number of sites to generate.")
    parser.add_argument("--cells_per_site", type=int, default=3, help="Cells per site.")
    parser.add_argument("--lat_range", type=float, nargs=2, default=[40.7, 40.8], help="Latitude range.")
    parser.add_argument("--lon_range", type=float, nargs=2, default=[-74.05, -73.95], help="Longitude range.")
    parser.add_argument("--default_cell_tilt", type=float, default=12.0, help="Default cell electrical tilt.")
    parser.add_argument("--output_dir", type=str, default="./generated_data", help="Base directory for all outputs.")
    parser.add_argument("--topology_csv", type=str, default="topology.csv", help="Filename for topology CSV within output_dir.")
    parser.add_argument("--config_csv", type=str, default="config.csv", help="Filename for initial config CSV within output_dir.")
    parser.add_argument("--dummy_training_csv", type=str, default="dummy_ue_training_data.csv", help="Filename for dummy training data CSV within output_dir.")
    parser.add_argument("--spatial_params_json", type=str, default="spatial_params.json", help="Path to spatial parameters JSON.")
    parser.add_argument("--time_params_json", type=str, default="time_params.json", help="Path to time parameters JSON.")
    parser.add_argument("--ue_data_dir", type=str, default="ue_data_per_tick", help="Subdirectory name for per-tick UE data within each Day directory.")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Subdirectory name for plots within each Day directory.")
    parser.add_argument("--num_ues", type=int, default=500, help="Number of UEs to generate per tick.")
    parser.add_argument("--generate_dummy_training_flag", action="store_true", help="Generate dummy training data file.")
    parser.add_argument("--num_training_samples", type=int, default=12000, help="Number of samples for dummy training data.")
    parser.add_argument("--generate_plots_flag", action="store_true", help="Generate visualization plots.")
    parser.add_argument("--plot_max_ticks", type=int, default=10, help="Max number of ticks to plot per day (0 for all).")
    args = parser.parse_args()

    args.topology_csv = os.path.join(args.output_dir, args.topology_csv)
    args.config_csv = os.path.join(args.output_dir, args.config_csv)
    args.dummy_training_csv = os.path.join(args.output_dir, args.dummy_training_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
