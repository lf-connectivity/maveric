# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Coverage and Capacity Optimization (CCO) Prediction Module.

This module handles prediction and inference for CCO optimization,
providing optimized cell elevation angles (el_deg) as output.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from apps.coverage_capacity_optimization.cco_env import CCOEnvironment
from apps.coverage_capacity_optimization.constants import CELL_EL_DEG, CELL_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CCOPredictor:
    """
    Predictor for Coverage and Capacity Optimization.
    
    This class provides optimized cell elevation angles (el_deg) based on
    trained models and current network conditions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize CCO Predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.trained_model: Optional[Dict] = None
        self.environment: Optional[CCOEnvironment] = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from file.
        
        Args:
            model_path: Path to model file
        """
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                self.trained_model = pickle.load(f)
            
            logger.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def set_environment(self, environment: CCOEnvironment) -> None:
        """
        Set CCO environment for predictions.
        
        Args:
            environment: CCO Environment instance
        """
        self.environment = environment
        logger.info("Environment set for predictions")
    
    def predict_el_deg(self, config: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict optimized elevation angles (el_deg) for cells.
        
        Args:
            config: Optional configuration DataFrame to optimize
            
        Returns:
            DataFrame with optimized el_deg values
        """
        if self.environment is None:
            raise ValueError("Environment not set. Call set_environment() first.")
        
        if config is None:
            config = self.environment.config.copy()
        
        logger.info("Predicting optimized el_deg values")
        
        # Use best configuration from trained model if available
        if self.trained_model and 'best_config' in self.trained_model:
            optimized_config = self.trained_model['best_config'].copy()
            logger.info("Using best configuration from trained model")
        else:
            # Run optimization to get best configuration
            optimized_config = self._optimize_configuration(config)
            logger.info("Generated optimized configuration")
        
        # Extract el_deg values
        el_deg_predictions = optimized_config[[CELL_ID, CELL_EL_DEG]].copy()
        
        logger.info(f"Generated el_deg predictions for {len(el_deg_predictions)} cells")
        return el_deg_predictions
    
    def _optimize_configuration(self, initial_config: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize configuration using greedy approach.
        
        Args:
            initial_config: Starting configuration
            
        Returns:
            Optimized configuration
        """
        from apps.coverage_capacity_optimization.cco_trainer import CCOTrainer
        
        # Create temporary trainer for optimization
        temp_trainer = CCOTrainer(self.environment)
        
        # Run short training to get optimized configuration
        results = temp_trainer.train(
            num_epochs=10,  # Short optimization
            epsilon=0.0,    # No exploration, just greedy
            seed=42
        )
        
        return results['best_config']
    
    def predict_single_cell_el_deg(self, cell_id: str) -> float:
        """
        Predict optimized el_deg for a single cell.
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            Optimized el_deg value
        """
        if self.trained_model and 'best_config' in self.trained_model:
            best_config = self.trained_model['best_config']
            cell_data = best_config[best_config[CELL_ID] == cell_id]
            
            if not cell_data.empty:
                return cell_data[CELL_EL_DEG].iloc[0]
        
        # Fallback to current configuration
        if self.environment:
            current_config = self.environment.get_config(cell_id)
            return current_config.get(CELL_EL_DEG, 0.0)
        
        raise ValueError(f"Could not predict el_deg for cell {cell_id}")
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of optimization results.
        
        Returns:
            Optimization summary dictionary
        """
        if not self.trained_model:
            return {"error": "No trained model available"}
        
        summary = {
            "algorithm": self.trained_model.get("algorithm", "unknown"),
            "best_reward": self.trained_model.get("best_reward", 0.0),
            "num_cells": len(self.trained_model.get("best_config", [])),
            "training_epochs": len(self.trained_model.get("training_history", []))
        }
        
        return summary
    
    def export_predictions(self, predictions: pd.DataFrame, output_path: str) -> None:
        """
        Export predictions to CSV file.
        
        Args:
            predictions: DataFrame with predictions
            output_path: Output file path
        """
        predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions exported to: {output_path}")
    
    def compare_with_current(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Compare predictions with current configuration.
        
        Args:
            predictions: DataFrame with predicted el_deg values
            
        Returns:
            DataFrame with comparison
        """
        if self.environment is None:
            raise ValueError("Environment not set")
        
        current_config = self.environment.config[[CELL_ID, CELL_EL_DEG]].copy()
        
        # Merge predictions with current config
        comparison = current_config.merge(
            predictions, 
            on=CELL_ID, 
            suffixes=('_current', '_predicted')
        )
        
        # Calculate differences
        comparison['el_deg_diff'] = (
            comparison[f'{CELL_EL_DEG}_predicted'] - comparison[f'{CELL_EL_DEG}_current']
        )
        
        logger.info(f"Comparison generated for {len(comparison)} cells")
        return comparison
    
    def get_cell_recommendations(self, predictions: pd.DataFrame, 
                               threshold: float = 2.0) -> pd.DataFrame:
        """
        Get recommendations for cells that need significant changes.
        
        Args:
            predictions: DataFrame with predicted el_deg values
            threshold: Minimum change threshold for recommendations
            
        Returns:
            DataFrame with recommendations
        """
        comparison = self.compare_with_current(predictions)
        
        # Filter cells with significant changes
        recommendations = comparison[
            abs(comparison['el_deg_diff']) >= threshold
        ].copy()
        
        recommendations['recommendation'] = recommendations['el_deg_diff'].apply(
            lambda x: 'Increase' if x > 0 else 'Decrease'
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def validate_predictions(self, predictions: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate prediction results.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            "has_required_columns": CELL_ID in predictions.columns and CELL_EL_DEG in predictions.columns,
            "no_missing_values": not predictions.isnull().any().any(),
            "valid_el_deg_range": True,
            "all_cells_present": True
        }
        
        # Check el_deg range
        if CELL_EL_DEG in predictions.columns:
            el_deg_values = predictions[CELL_EL_DEG]
            validation_results["valid_el_deg_range"] = (
                el_deg_values.min() >= 0 and el_deg_values.max() <= 20
            )
        
        # Check if all cells are present
        if self.environment:
            expected_cells = set(self.environment.config[CELL_ID].unique())
            predicted_cells = set(predictions[CELL_ID].unique())
            validation_results["all_cells_present"] = expected_cells == predicted_cells
        
        return validation_results