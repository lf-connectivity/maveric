# main_app.py
# The main orchestrator for the Energy Saving RL application pipeline is provided by this file.

import argparse
import logging
import os
import sys

# --- Python Path Setup ---
# The application directory is determined based on the location of this file.
# It is assumed that this application resides in a directory such as 'apps/energy_saving_app/'.
# The project root is determined by moving two levels up from the application directory.
# The project root is inserted at the beginning of the Python path to facilitate local imports.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
sys.path.insert(0, PROJECT_ROOT)

# --- Local Module Imports ---
# Local modules required for the pipeline are imported.
# If any import fails, an error message is displayed and the program is terminated.
try:
    from bdt_manager import BDTManager
    from data_preprocessor import UEDataPreprocessor
    from energy_saving_visualizer import EnergySavingVisualizer
    from rl_predictor import run_rl_prediction
    from rl_trainer import run_rl_training
except ImportError as e:
    print(f"FATAL: Could not import a required local module: {e}")
    print("Please ensure main_app.py and all its helper modules are in the same directory.")
    sys.exit(1)

# --- Logging Setup ---
# Logging is configured to display messages with timestamps, logger names, and severity levels.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    The main function is used to execute the energy saving application pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Energy Saving Application using BDT and RL.", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--preprocess-data", action="store_true", help="Step 1: Preprocess raw UE data CSVs for Gym compatibility."
    )
    parser.add_argument(
        "--train-bdt", action="store_true", help="Step 2: Train the Bayesian Digital Twin model map via the backend."
    )
    parser.add_argument("--train-rl", action="store_true", help="Step 3: Train the reinforcement learning model.")
    parser.add_argument(
        "--infer", action="store_true", help="Step 4: Run inference with the trained RL model for a specific tick."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Step 5: Generate comparison plots for a given tick on a test day."
    )

    # Arguments for specific steps
    # The list of day numbers to be used for RL training is specified.
    parser.add_argument(
        "--train-days",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of day numbers (e.g., 0 1 2 3) to use for RL training.",
    )
    # The day number to be used for inference and visualization is specified.
    parser.add_argument(
        "--test-day", type=int, default=4, help="The day number (e.g., 4) to use for inference and visualization."
    )
    # The specific tick or hour required for inference and visualization is specified.
    parser.add_argument(
        "--tick", type=int, default=None, help="The specific tick/hour (0-23) required for inference and visualization."
    )
    # The name of the Docker container running the 'training' service for BDT training is specified.
    parser.add_argument(
        "--container",
        type=str,
        default="radp_dev-training-1",
        help="Name of the Docker container running the 'training' service (for --train-bdt).",
    )
    # The total number of timesteps for RL training is specified.
    parser.add_argument(
        "--total-timesteps", type=int, default=48000, help="Total timesteps for RL training (for --train-rl)."
    )
    # The unique ID for the BDT model being trained on the backend is specified.
    parser.add_argument(
        "--bdt-model-id",
        type=str,
        default="bdt_energy_saving_v1",
        help="Unique ID for the BDT model being trained on the backend.",
    )

    args = parser.parse_args()

    # --- Configuration: Define shared file and directory paths ---
    # Shared file and directory paths used throughout the pipeline are defined.
    BASE_DATA_DIR = os.path.join(APP_DIR, "generated_data")
    STATIC_DATA_DIR = os.path.join(APP_DIR, "data")
    TOPOLOGY_PATH = os.path.join(STATIC_DATA_DIR, "topology.csv")
    CONFIG_PATH = os.path.join(STATIC_DATA_DIR, "config.csv")
    BDT_TRAINING_DATA_PATH = os.path.join(STATIC_DATA_DIR, "dummy_ue_training_data.csv")

    BDT_MODEL_PATH = os.path.join(APP_DIR, "bdt_model_map.pickle")
    RL_MODEL_PATH = os.path.join(APP_DIR, "energy_saver_agent.zip")
    RL_LOG_DIR = os.path.join(APP_DIR, "rl_training_logs")
    PLOT_OUTPUT_DIR = os.path.join(APP_DIR, "plots")

    # --- Pipeline Execution ---
    # The pipeline steps are executed based on the provided command-line arguments.
    if args.preprocess_data:
        days_to_process = list(set(args.train_days + [args.test_day]))
        logger.info(f"--- Running Step 1: UE Data Preprocessing for days: {sorted(days_to_process)} ---")
        preprocessor = UEDataPreprocessor(base_data_dir=BASE_DATA_DIR)
        preprocessor.run(days=days_to_process)
        logger.info("--- Preprocessing Step Finished ---")

    if args.train_bdt:
        logger.info("--- Running Step 2: BDT Training ---")
        bdt_manager = BDTManager(
            topology_path=TOPOLOGY_PATH, training_data_path=BDT_TRAINING_DATA_PATH, model_path=BDT_MODEL_PATH
        )
        bdt_manager.train(model_id=args.bdt_model_id, container_name=args.container)
        logger.info("--- BDT Training Step Finished ---")

    if args.train_rl:
        logger.info(f"--- Running Step 3: RL Training on Days: {args.train_days} ---")
        run_rl_training(
            bdt_model_path=BDT_MODEL_PATH,
            base_ue_data_dir=BASE_DATA_DIR,
            training_days=args.train_days,
            topology_path=TOPOLOGY_PATH,
            config_path=CONFIG_PATH,
            rl_model_path=RL_MODEL_PATH,
            log_dir=RL_LOG_DIR,
            total_timesteps=args.total_timesteps,
        )
        logger.info("--- RL Training Step Finished ---")

    if args.infer:
        if args.tick is None:
            parser.error("--tick is required for inference.")
        logger.info(f"--- Running Step 4: Inference for Tick {args.tick} ---")
        run_rl_prediction(model_load_path=RL_MODEL_PATH, topology_path=TOPOLOGY_PATH, target_tick=args.tick)
        logger.info("--- Inference Step Finished ---")

    if args.visualize:
        if args.tick is None:
            parser.error("--tick is required for visualization.")
        logger.info(f"--- Running Step 5: Visualization on Test Day {args.test_day} for Tick {args.tick} ---")
        try:
            visualizer = EnergySavingVisualizer(
                bdt_model_path=BDT_MODEL_PATH,
                rl_model_path=RL_MODEL_PATH,
                topology_path=TOPOLOGY_PATH,
                config_path=CONFIG_PATH,
                base_ue_data_dir=BASE_DATA_DIR,
            )
            visualizer.generate_comparison_plots(day=args.test_day, tick=args.tick, output_dir=PLOT_OUTPUT_DIR)
            logger.info("--- Visualization Step Finished ---")
        except Exception as e:
            logger.exception(f"Visualization failed with an error: {e}")

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
