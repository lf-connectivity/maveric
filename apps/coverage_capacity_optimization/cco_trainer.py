# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Coverage and Capacity Optimization (CCO) Training Module.

This module implements training logic for CCO optimization using the dGPCO algorithm.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from apps.coverage_capacity_optimization.cco_env import CCOEnvironment
from apps.coverage_capacity_optimization.constants import CELL_EL_DEG, CELL_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CCOTrainer:
    """
    Trainer for Coverage and Capacity Optimization using dGPCO algorithm.
    """
    
    def __init__(self, environment: CCOEnvironment):
        """
        Initialize CCO Trainer.
        
        Args:
            environment: CCO Environment instance
        """
        self.environment = environment
        self.training_history: List[Dict[str, Any]] = []
        
        # Training parameters
        self.num_epochs = 100
        self.epsilon = 0.1  # For epsilon-greedy exploration
        self.seed = 42
        
        # Metrics tracking
        self.current_epoch = 0
        self.best_reward = -np.inf
        self.best_config = None
        
        logger.info("Initialized CCO Trainer")
    
    def train(self, num_epochs: int = 100, epsilon: float = 0.1, seed: int = 42,
              lambda_: float = 0.5, weak_coverage_threshold: float = -90,
              over_coverage_threshold: float = 0) -> Dict[str, Any]:
        """
        Train the CCO optimization model using dGPCO algorithm.
        
        Args:
            num_epochs: Number of training epochs
            epsilon: Exploration rate for epsilon-greedy
            seed: Random seed
            lambda_: Weight parameter for weak vs over coverage
            weak_coverage_threshold: Threshold for weak coverage
            over_coverage_threshold: Threshold for over coverage
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting dGPCO training for {num_epochs} epochs")
        
        # Set training parameters
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        np.random.seed(seed)
        
        # Initialize training
        epoch_rewards = []
        
        # Calculate initial metric
        _, _, initial_reward = self.environment.calc_metric(
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )
        epoch_rewards.append(initial_reward)
        
        logger.info(f"Initial CCO metric = {initial_reward:.3f}")
        
        # Run dGPCO training
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Run one epoch of dGPCO
            epoch_reward = self._run_dgpco_epoch(
                lambda_=lambda_,
                weak_coverage_threshold=weak_coverage_threshold,
                over_coverage_threshold=over_coverage_threshold,
            )
            epoch_rewards.append(epoch_reward)
            
            # Update best configuration
            if epoch_reward > self.best_reward:
                self.best_reward = epoch_reward
                self.best_config = self.environment.config.copy()
            
            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'reward': epoch_reward,
                'config': self.environment.config.copy()
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: CCO metric = {epoch_reward:.3f}")
        
        # Final results
        results = {
            'algorithm': 'dgpco',
            'total_epochs': len(epoch_rewards),
            'initial_reward': initial_reward,
            'final_reward': epoch_rewards[-1],
            'best_reward': self.best_reward,
            'average_reward': np.mean(epoch_rewards),
            'training_history': self.training_history,
            'best_config': self.best_config,
            'epoch_rewards': epoch_rewards
        }
        
        logger.info(f"Training completed. Best reward: {self.best_reward:.3f}")
        return results
    
    def _run_dgpco_epoch(self, lambda_: float, weak_coverage_threshold: float,
                        over_coverage_threshold: float) -> float:
        """
        Run one epoch of dGPCO algorithm.
        
        Args:
            lambda_: Weight parameter
            weak_coverage_threshold: Weak coverage threshold
            over_coverage_threshold: Over coverage threshold
            
        Returns:
            Epoch reward
        """
        num_cells = self.environment.num_cells
        epoch_rewards = []
        
        for cell_idx in range(num_cells):
            # Get current cell configuration
            cell_id = self.environment.config.iloc[cell_idx][CELL_ID]
            current_tilt = self.environment.config.iloc[cell_idx][CELL_EL_DEG]
            
            # Calculate current metric
            _, _, current_reward = self.environment.calc_metric(
                lambda_=lambda_,
                weak_coverage_threshold=weak_coverage_threshold,
                over_coverage_threshold=over_coverage_threshold,
            )
            
            # Try different tilt adjustments
            tilt_adjustments = [-4, -3, -2, -1, 1, 2, 3, 4]
            rewards = []
            tilts_tried = []
            
            for adjustment in tilt_adjustments:
                new_tilt = self._get_adjusted_tilt(current_tilt, adjustment)
                if new_tilt != current_tilt:
                    # Temporarily update configuration
                    original_tilt = self.environment.config.iloc[cell_idx][CELL_EL_DEG]
                    self.environment.config.iloc[cell_idx][CELL_EL_DEG] = new_tilt
                    
                    # Calculate reward
                    _, _, reward = self.environment.calc_metric(
                        lambda_=lambda_,
                        weak_coverage_threshold=weak_coverage_threshold,
                        over_coverage_threshold=over_coverage_threshold,
                    )
                    rewards.append(reward)
                    tilts_tried.append(new_tilt)
                    
                    # Restore original tilt
                    self.environment.config.iloc[cell_idx][CELL_EL_DEG] = original_tilt
            
            # Choose best tilt (with epsilon-greedy exploration)
            if rewards and np.random.random() > self.epsilon:
                best_idx = np.argmax(rewards)
                best_tilt = tilts_tried[best_idx]
                best_reward = rewards[best_idx]
            else:
                # Random exploration
                if rewards:
                    best_idx = np.random.randint(len(rewards))
                    best_tilt = tilts_tried[best_idx]
                    best_reward = rewards[best_idx]
                else:
                    best_tilt = current_tilt
                    best_reward = current_reward
            
            # Apply best tilt
            self.environment.config.iloc[cell_idx][CELL_EL_DEG] = best_tilt
            epoch_rewards.append(best_reward)
            
            logger.debug(f"Cell {cell_id}: {current_tilt} -> {best_tilt}, reward: {best_reward:.3f}")
        
        return np.mean(epoch_rewards) if epoch_rewards else 0
    
    def _get_adjusted_tilt(self, current_tilt: float, adjustment: int) -> float:
        """
        Get adjusted tilt value.
        
        Args:
            current_tilt: Current tilt value
            adjustment: Adjustment amount
            
        Returns:
            Adjusted tilt value
        """
        tilt_values = self.environment.get_valid_tilt_values()
        if not tilt_values:
            return current_tilt
        
        try:
            current_idx = tilt_values.index(current_tilt)
            new_idx = current_idx + adjustment
            
            if 0 <= new_idx < len(tilt_values):
                return tilt_values[new_idx]
            else:
                return current_tilt
        except ValueError:
            return current_tilt
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        model_data = {
            'best_config': self.best_config,
            'best_reward': self.best_reward,
            'training_history': self.training_history,
            'algorithm': 'dgpco'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_config = model_data.get('best_config')
        self.best_reward = model_data.get('best_reward', -np.inf)
        self.training_history = model_data.get('training_history', [])
        
        logger.info(f"Model loaded from: {filepath}")
    
    def get_training_summary(self) -> str:
        """
        Get training summary as string.
        
        Returns:
            Training summary
        """
        if not self.training_history:
            return "No training history available"
        
        summary = f"""
Training Summary:
  Algorithm: dGPCO
  Total Epochs: {len(self.training_history)}
  Best Reward: {self.best_reward:.3f}
  Final Reward: {self.training_history[-1]['reward']:.3f}
"""
        
        return summary