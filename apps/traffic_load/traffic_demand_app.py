# maveric/apps/traffic_load/traffic_demand_app.py

import os
import sys
import json
import logging
import argparse

import pandas as pd
import numpy as np

# Assuming these modules are in the same directory or Python path is set up correctly
# For a real application, these imports would depend on your project structure
# and how you install/manage packages.
try:
    # If radp is installed or in PYTHONPATH
    from radp.digital_twin.utils.traffic_demand_simulation import TrafficDemandModel
    from radp.digital_twin.utils import constants as c # For column names
except ImportError:
    print("Error: Could not import TrafficDemandModel or RADP constants.")
    print("Ensure 'radp' is installed or MAVERIC_ROOT/radp is in your PYTHONPATH.")
    sys.exit(1)

try:
    from config_gen import ScenarioConfigurationGenerator
    from plot_gen import TrafficDataVisualizer
except ImportError:
    print("Error: Could not import ScenarioConfigurationGenerator or TrafficDataVisualizer.")
    print("Ensure config_gen.py and plot_gen.py are in the same directory or accessible.")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function to orchestrate traffic demand generation, configuration,
    and visualization.
    """
    logger.info("--- Starting Traffic Demand Application ---")

    # --- 0. Initialize Generators/Models ---
    config_generator = ScenarioConfigurationGenerator()
    traffic_model = TrafficDemandModel()
    visualizer = TrafficDataVisualizer()

    # --- 1. Load/Generate Topology & Initial Config ---
    topology_df = None
    initial_config_df = None

    if args.generate_config_flag:
        logger.info("Generating new topology and initial configuration files...")
        default_cfg_params = {getattr(c, 'CELL_EL_DEG', 'cell_el_deg'): args.default_cell_tilt}
        topology_df, initial_config_df = config_generator.generate_topology_and_config_files(
            num_sites=args.num_sites,
            cells_per_site=args.cells_per_site,
            lat_range=tuple(args.lat_range),
            lon_range=tuple(args.lon_range),
            default_power_dbm=args.default_cell_power,
            default_config_params=default_cfg_params,
            output_topology_path=args.topology_csv,
            output_config_path=args.config_csv
        )
    else:
        logger.info(f"Loading topology from {args.topology_csv} and config from {args.config_csv}")
        try:
            topology_df = pd.read_csv(args.topology_csv)
            # Basic validation
            required_topo_cols = [getattr(c, 'CELL_ID', 'cell_id'), getattr(c, 'CELL_LAT', 'cell_lat'), getattr(c, 'CELL_LON', 'cell_lon')]
            if not all(col in topology_df.columns for col in required_topo_cols):
                 raise ValueError(f"Topology CSV missing one of required columns: {required_topo_cols}")
            logger.info(f"Loaded topology with {len(topology_df)} cells.")
        except FileNotFoundError:
            logger.error(f"Topology file not found: {args.topology_csv}. Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading topology: {e}. Exiting.")
            sys.exit(1)

        try:
            initial_config_df = pd.read_csv(args.config_csv)
            # Basic validation
            required_conf_cols = [getattr(c, 'CELL_ID', 'cell_id'), getattr(c, 'CELL_EL_DEG', 'cell_el_deg')]
            if not all(col in initial_config_df.columns for col in required_conf_cols):
                raise ValueError(f"Config CSV missing one of required columns: {required_conf_cols}")
            logger.info(f"Loaded initial config for {len(initial_config_df)} cells.")
        except FileNotFoundError:
            logger.error(f"Config file not found: {args.config_csv}. Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading config: {e}. Exiting.")
            sys.exit(1)
            
    if topology_df is None or topology_df.empty:
        logger.error("Topology data is not available. Exiting.")
        sys.exit(1)


    # --- 2. Load Spatial and Time Parameters ---
    try:
        with open(args.spatial_params_json, "r") as f:
            spatial_params = json.load(f)
        with open(args.time_params_json, "r") as f:
            time_params = json.load(f)
        logger.info(f"Loaded spatial parameters from {args.spatial_params_json} and time parameters from {args.time_params_json}")
    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {e}. Creating dummy files if specified, else exiting.")
        if args.create_dummy_params_if_missing:
            logger.info("Attempting to create dummy spatial and time parameter files.")
            num_types_generated = 3
            default_types = [f"type_{i+1}" for i in range(num_types_generated)]
            spatial_params = {"types": default_types, "proportions": list(np.random.dirichlet(np.ones(num_types_generated)))}
            with open(args.spatial_params_json, "w") as f: json.dump(spatial_params, f, indent=4)
            logger.info(f"Created dummy {args.spatial_params_json}")

            num_ticks_dummy = time_params.get("total_ticks", 24) # Use from args if possible, else default
            time_params = {
                "total_ticks": num_ticks_dummy, "tick_duration_min": 60,
                "time_weights": {stype: list(np.random.rand(num_ticks_dummy)) for stype in default_types},
            }
            with open(args.time_params_json, "w") as f: json.dump(time_params, f, indent=4)
            logger.info(f"Created dummy {args.time_params_json}")
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading JSON parameter files: {e}. Exiting.")
        sys.exit(1)

    # --- 3. Generate Traffic Demand ---
    logger.info(f"Generating traffic demand for {args.num_ues} UEs per tick...")
    ue_data_per_tick_dict, spatial_layout = traffic_model.generate_traffic_demand(
        topology_df=topology_df,
        spatial_params=spatial_params,
        time_params=time_params,
        num_ues_per_tick=args.num_ues
    )

    if not ue_data_per_tick_dict:
        logger.error("Traffic demand generation resulted in no UE data. Exiting.")
        sys.exit(1)
    if not spatial_layout:
        logger.warning("Spatial layout generation was empty. Plots might be affected.")
        
    # Save the generated spatial layout (optional)
    try:
        layout_output_path = os.path.join(args.output_dir, "generated_spatial_layout.json")
        os.makedirs(os.path.dirname(layout_output_path), exist_ok=True)
        serializable_layout = []
        for cell in spatial_layout:
            serializable_layout.append({
                "bounds": [[coord[0], coord[1]] for coord in cell["bounds"]],
                "type": cell["type"],
                "cell_id": cell.get("cell_id", -1), # Use .get for safety
                "area_sq_km": cell.get("area_sq_km", 0.0)
            })
        with open(layout_output_path, "w") as f:
            json.dump(serializable_layout, f, indent=2)
        logger.info(f"Saved generated spatial layout to {layout_output_path}")
    except Exception as e:
        logger.error(f"Could not save spatial layout: {e}")


    # --- 4. Save Per-Tick UE Data ---
    logger.info(f"Saving per-tick UE data to directory: {args.ue_data_dir}")
    os.makedirs(args.ue_data_dir, exist_ok=True)
    all_ue_dfs = [] # For dummy training data generation
    for tick, ue_df in ue_data_per_tick_dict.items():
        if ue_df.empty:
            logger.info(f"No UEs generated for tick {tick}, skipping save for this tick.")
            continue
        all_ue_dfs.append(ue_df)
        # Standardize column names for CCO app (mock_ue_id, lon, lat, tick)
        # Assuming c.LON and c.LAT are already the lon/lat column names from TrafficDemandModel
        COL_LAT = getattr(c, 'LAT', 'lat'); COL_LON = getattr(c, 'LON', 'lon')
        ue_df_to_save = ue_df[['ue_id', COL_LON, COL_LAT, 'tick']].rename(
            columns={'ue_id': 'mock_ue_id', COL_LON: 'lon', COL_LAT: 'lat'}
        )
        # Ensure correct order
        ue_df_to_save = ue_df_to_save[['mock_ue_id', 'lon', 'lat', 'tick']]

        output_filename = f"generated_ue_data_for_cco_{tick}.csv"
        output_csv_path = os.path.join(args.ue_data_dir, output_filename)
        try:
            ue_df_to_save.to_csv(output_csv_path, index=False)
            logger.debug(f"Saved UE data for tick {tick} to: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save UE data for tick {tick} to {output_csv_path}: {e}")
    logger.info("Finished saving per-tick UE data.")

    # --- 5. Generate Dummy Training Data ---
    if args.generate_dummy_training_flag:
        if all_ue_dfs:
            combined_ue_data_for_training = pd.concat(all_ue_dfs, ignore_index=True)
            logger.info("Generating dummy training data...")
            _ = config_generator.generate_dummy_training_data(
                topology_df=topology_df,
                ue_data_all_ticks=combined_ue_data_for_training,
                num_training_samples=args.num_training_samples,
                possible_tilts=list(np.arange(0.0, 21.0, 1.0)), # Example tilts
                output_training_data_path=args.dummy_training_csv
            )
        else:
            logger.warning("No UE data was generated across ticks. Skipping dummy training data generation.")
    else:
        logger.info("Skipping dummy training data generation as per flag.")


    # --- 6. Generate Visualizations ---
    if args.generate_plots_flag:
        logger.info(f"Generating visualizations and saving to directory: {args.plot_dir}")
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # For serving cell visualization (optional, requires a simple model here or pre-generated data)
        # This app currently does not implement its own serving cell calculation.
        # If you need it, you'd add logic here to compute or load it.
        # For now, plots will show UEs without serving cell specific coloring unless such data is loaded.
        
        ticks_to_plot = sorted(ue_data_per_tick_dict.keys())
        if args.plot_max_ticks > 0 and len(ticks_to_plot) > args.plot_max_ticks:
            # Plot first, middle, and last few if too many ticks
            sample_indices = np.linspace(0, len(ticks_to_plot) - 1, args.plot_max_ticks, dtype=int)
            ticks_to_plot = [ticks_to_plot[i] for i in sample_indices]
            logger.info(f"Too many ticks to plot all. Plotting a sample of {args.plot_max_ticks} ticks: {ticks_to_plot}")


        for tick in ticks_to_plot:
            ue_df_for_plot = ue_data_per_tick_dict.get(tick)
            if ue_df_for_plot is not None and not ue_df_for_plot.empty:
                plot_path_template = os.path.join(args.plot_dir, "ue_distribution_tick_{tick}.png")
                visualizer.generate_tick_visualization(
                    tick_to_plot=tick,
                    ue_data_for_tick=ue_df_for_plot,
                    topology_df=topology_df,
                    spatial_layout=spatial_layout,
                    spatial_params_for_colors=spatial_params, # Pass spatial_params for color consistency
                    plot_output_path_template=plot_path_template,
                    serving_cell_data_for_tick=None # Pass actual serving cell data here if available
                )
            else:
                logger.info(f"Skipping plot for tick {tick} as no UE data was found.")
        logger.info("Finished generating visualizations.")
    else:
        logger.info("Skipping plot generation as per flag.")

    logger.info("--- Traffic Demand Application Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Demand Generation Application")

    # Configuration generation
    parser.add_argument("--generate_config_flag", action="store_true", help="Generate new topology and config files.")
    parser.add_argument("--num_sites", type=int, default=5, help="Number of sites to generate if --generate_config_flag is set.")
    parser.add_argument("--cells_per_site", type=int, default=3, help="Cells per site if --generate_config_flag is set.")
    parser.add_argument("--lat_range", type=float, nargs=2, default=[40.7, 40.8], help="Latitude range for generation.")
    parser.add_argument("--lon_range", type=float, nargs=2, default=[-74.05, -73.95], help="Longitude range for generation.")
    parser.add_argument("--default_cell_power", type=float, default=25.0, help="Default cell Tx power (dBm).")
    parser.add_argument("--default_cell_tilt", type=float, default=12.0, help="Default cell electrical tilt.")

    # File paths
    parser.add_argument("--output_dir", type=str, default="./generated_data", help="Base directory for all generated outputs.")
    parser.add_argument("--topology_csv", type=str, default="topology.csv", help="Path to topology CSV file (input/output).")
    parser.add_argument("--config_csv", type=str, default="config.csv", help="Path to initial config CSV file (input/output).")
    parser.add_argument("--spatial_params_json", type=str, default="spatial_params.json", help="Path to spatial parameters JSON file.")
    parser.add_argument("--time_params_json", type=str, default="time_params.json", help="Path to time parameters JSON file.")
    parser.add_argument("--ue_data_dir", type=str, default="ue_data_per_tick", help="Directory to save per-tick UE data CSVs.")
    parser.add_argument("--dummy_training_csv", type=str, default="dummy_ue_training_data.csv", help="Path to save dummy training data CSV.")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save generated plots.")

    # Simulation parameters
    parser.add_argument("--num_ues", type=int, default=500, help="Number of UEs to generate per tick.")
    parser.add_argument("--create_dummy_params_if_missing", action="store_true", help="Create dummy spatial/time JSONs if not found.")
    
    # Dummy training data generation
    parser.add_argument("--generate_dummy_training_flag", action="store_true", help="Generate dummy training data file.")
    parser.add_argument("--num_training_samples", type=int, default=12000, help="Approximate number of samples for dummy training data.")

    # Plotting
    parser.add_argument("--generate_plots_flag", action="store_true", help="Generate visualization plots.")
    parser.add_argument("--plot_max_ticks", type=int, default=10, help="Maximum number of ticks to plot if many are generated (0 for all).")


    args = parser.parse_args()

    # Adjust relative paths to be under output_dir
    args.topology_csv = os.path.join(args.output_dir, args.topology_csv)
    args.config_csv = os.path.join(args.output_dir, args.config_csv)
    args.ue_data_dir = os.path.join(args.output_dir, args.ue_data_dir)
    args.dummy_training_csv = os.path.join(args.output_dir, args.dummy_training_csv)
    args.plot_dir = os.path.join(args.output_dir, args.plot_dir)
    
    # For spatial/time params, we expect them to be inputs, but can create dummies in place if specified
    # So, their paths are not prefixed by output_dir unless user explicitly sets them there.

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
