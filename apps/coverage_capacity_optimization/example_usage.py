# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage script for the refactored CCO application.

This script demonstrates how to use the modular CCO components
both programmatically and through the CLI interface.
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.coverage_capacity_optimization.main_app import CCOMainApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_programmatic_usage():
    """Example of using the CCO application programmatically."""
    logger.info("=== Programmatic Usage Example ===")
    
    # Define file paths
    topology_path = "data/topology.csv"
    training_data_path = "data/ue_training_data.csv"
    prediction_data_path = "data/ue_data.csv"
    config_path = "data/config.csv"
    model_id = "example_model"
    
    # Check if data files exist
    required_files = [topology_path, training_data_path, prediction_data_path, config_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.warning(f"Missing data files: {missing_files}")
        logger.info("Please ensure data files are present before running the example")
        return
    
    try:
        # Initialize application
        app = CCOMainApp(
            topology_path=topology_path,
            training_data_path=training_data_path,
            prediction_data_path=prediction_data_path,
            config_path=config_path,
            model_id=model_id
        )
        
        # Run individual steps
        logger.info("Step 1: Training BDT model...")
        bdt_results = app.train_bdt(container_name="radp_training")
        logger.info(f"BDT training completed: {bdt_results}")
        
        logger.info("Step 2: Training CCO model...")
        cco_results = app.train_cco()
        logger.info(f"CCO training completed. Best reward: {cco_results['best_reward']:.3f}")
        
        logger.info("Step 3: Generating predictions...")
        predictions = app.predict_el_deg()
        logger.info(f"Predictions generated for {len(predictions['predictions'])} cells")
        
        # Print some results
        pred_df = predictions['predictions']
        logger.info(f"Sample predictions:\n{pred_df.head()}")
        
        if predictions['recommendations'] is not None and len(predictions['recommendations']) > 0:
            rec_df = predictions['recommendations']
            logger.info(f"Recommendations for {len(rec_df)} cells:\n{rec_df.head()}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")


def example_cli_usage():
    """Example of CLI usage commands."""
    logger.info("=== CLI Usage Examples ===")
    
    commands = [
        "# Run full pipeline",
        "python main_app.py --topology data/topology.csv --training data/ue_training_data.csv --prediction data/ue_data.csv --config data/config.csv --model-id example_model",
        "",
        "# Run with custom parameters",
        "python main_app.py --topology data/topology.csv --training data/ue_training_data.csv --prediction data/ue_data.csv --config data/config.csv --model-id example_model --epochs 50 --container my_radp_container",
        "",
        "# Preprocess data only",
        "python main_app.py --preprocess --base-dir data --days 1 2 3",
        "",
        "# Run with verbose logging",
        "python main_app.py --topology data/topology.csv --training data/ue_training_data.csv --prediction data/ue_data.csv --config data/config.csv --model-id example_model --verbose"
    ]
    
    for command in commands:
        print(command)


def example_data_processing():
    """Example of data preprocessing."""
    logger.info("=== Data Preprocessing Example ===")
    
    try:
        from apps.coverage_capacity_optimization.data_preprocessing import UEDataPreprocessor
        
        # Initialize preprocessor
        base_data_dir = "data"
        preprocessor = UEDataPreprocessor(base_data_dir)
        
        # Process data for specific days
        days_to_process = [1, 2, 3]
        preprocessor.run(days_to_process)
        
        logger.info(f"Data preprocessing completed for days: {days_to_process}")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")


def example_individual_components():
    """Example of using individual components."""
    logger.info("=== Individual Components Example ===")
    
    try:
        import pandas as pd
        from apps.coverage_capacity_optimization.cco_env import CCOEnvironment
        from apps.coverage_capacity_optimization.cco_trainer import CCOTrainer
        from apps.coverage_capacity_optimization.cco_prediction import CCOPredictor
        
        # Load data
        topology = pd.read_csv("data/topology.csv")
        ue_data = pd.read_csv("data/ue_data.csv")
        config = pd.read_csv("data/config.csv")
        
        # Set valid configuration values
        valid_configuration_values = {
            "cell_el_deg": [i * 1.0 for i in range(21)]  # 0 to 20 degrees
        }
        
        # Initialize environment
        environment = CCOEnvironment(
            topology=topology,
            ue_data=ue_data,
            config=config,
            bayesian_digital_twin_id="example_model",
            valid_configuration_values=valid_configuration_values
        )
        
        # Calculate initial metric
        rf_df, coverage_df, cco_obj = environment.calc_metric()
        logger.info(f"Initial CCO metric: {cco_obj:.3f}")
        
        # Initialize trainer
        trainer = CCOTrainer(environment)
        
        # Train model
        results = trainer.train(num_epochs=10, epsilon=0.1, seed=42)
        logger.info(f"Training completed. Best reward: {results['best_reward']:.3f}")
        
        # Initialize predictor
        predictor = CCOPredictor()
        predictor.set_environment(environment)
        
        # Generate predictions
        predictions = predictor.predict_el_deg()
        logger.info(f"Generated predictions for {len(predictions)} cells")
        
    except Exception as e:
        logger.error(f"Individual components example failed: {e}")


def main():
    """Main function to run examples."""
    logger.info("CCO Application Usage Examples")
    logger.info("=" * 50)
    
    # Show CLI usage examples
    example_cli_usage()
    
    # Check if data files exist for programmatic examples
    data_files_exist = all([
        os.path.exists("data/topology.csv"),
        os.path.exists("data/ue_training_data.csv"),
        os.path.exists("data/ue_data.csv"),
        os.path.exists("data/config.csv")
    ])
    
    if data_files_exist:
        logger.info("\nData files found. Running programmatic examples...")
        
        # Run programmatic example
        example_programmatic_usage()
        
        # Run individual components example
        example_individual_components()
        
        # Run data preprocessing example
        example_data_processing()
        
    else:
        logger.info("\nData files not found. Please ensure the following files exist:")
        logger.info("- data/topology.csv")
        logger.info("- data/ue_training_data.csv")
        logger.info("- data/ue_data.csv")
        logger.info("- data/config.csv")
        logger.info("\nThen run this example again to see programmatic usage.")


if __name__ == "__main__":
    main()
