import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from notebooks.radp_library import find_sim_boundary, get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class BayesianMRO(MobilityRobustnessOptimization):
    """Mobility Robustness Optimization using Bayesian optimization techniques.

    This class implements a Bayesian optimization approach to find optimal hysteresis
    and Time-to-Trigger (TTT) parameters for mobility robustness in cellular networks.
    It uses either Gaussian Process Regression (GPR) or XGBoost as the surrogate model
    to efficiently explore the parameter space and maximize the MRO metric.

    The optimization uses Expected Improvement (EI) as the acquisition function to
    balance exploration and exploitation during the search process.
    """

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict[str, Any]],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
        model_type: str = "gpr",
        suppress_warnings: bool = False,
    ):
        """Initialize the BayesianMRO optimizer.

        Args:
            mobility_model_params: Dictionary containing mobility model configuration
                parameters including UE track generation settings and simulation boundaries.
            topology: DataFrame containing network topology information with cell locations
                and configurations.
            bdt: Optional dictionary of pre-trained Bayesian Digital Twin models indexed
                by cell identifiers. Required for making predictions.
            model_type: Surrogate model type to use for Bayesian optimization.
                Options are 'gpr' (Gaussian Process Regression) or 'xgboost'.
                Defaults to 'gpr'.
            suppress_warnings: If True, suppresses convergence warnings during model
                training. Defaults to False.

        Raises:
            ImportError: If model_type is 'xgboost' but the xgboost package is not installed.
        """
        super().__init__(mobility_model_params, topology, new_data=None, bdt=bdt)
        self.model_type = model_type
        self.suppress_warnings = suppress_warnings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _expected_improvement(self, X: np.ndarray, model, best_y: float) -> np.ndarray:
        """Calculate the Expected Improvement (EI) acquisition function.

        The Expected Improvement measures the expected increase in the objective
        function value at candidate points, balancing exploitation of known good
        regions with exploration of uncertain regions.

        For XGBoost models, a simplified version is used that compares predictions
        directly to the best observed value. For Gaussian Process models, the full
        EI formula incorporating both mean and uncertainty is computed.

        Args:
            X: Candidate points to evaluate, shape (n_samples, n_features).
                Each row contains [hysteresis, TTT] parameters.
            model: Trained surrogate model (XGBRegressor or GaussianProcessRegressor).
            best_y: Best objective function value observed so far.

        Returns:
            Array of expected improvement values for each candidate point,
            shape (n_samples,). Higher values indicate more promising candidates.
        """
        if self.model_type == "xgboost":
            y_pred = model.predict(X)
            return y_pred - best_y
        else:
            mu, std = model.predict(X, return_std=True)
            std = np.maximum(std, 1e-9)
            Z = (mu - best_y) / std
            return (mu - best_y) * norm.cdf(Z) + std * norm.pdf(Z)

    def _init_model(self):
        """Initialize and configure the surrogate model for Bayesian optimization.

        Creates either a Gaussian Process Regressor or XGBoost regressor based on
        the model_type specified during initialization.

        For Gaussian Process:
            - Uses a composite kernel: ConstantKernel * Matern + WhiteKernel
            - Matern kernel with nu=2.5 for smoothness
            - WhiteKernel for noise modeling
            - Normalizes target values (normalize_y=True)

        For XGBoost:
            - Uses squared error regression objective
            - Default hyperparameters from XGBRegressor

        Returns:
            Initialized surrogate model ready for training.

        Raises:
            ImportError: If model_type is 'xgboost' but xgboost package is not installed.
        """
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor  # type: ignore
            except ImportError as e:
                raise ImportError("xgboost is required for model_type='xgboost'") from e
            return XGBRegressor(objective="reg:squarederror")
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(
                noise_level=1e-5, noise_level_bounds=(1e-10, 1e1)
            )
            return GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    def solve(self, n_epochs=20, init_samples: int = 5):
        """Optimize hysteresis and TTT parameters using Bayesian optimization.

        This method performs the complete optimization workflow:
        1. Validates that Bayesian Digital Twins are trained
        2. Sets up simulation boundaries and generates UE track data
        3. Makes predictions using the digital twins
        4. Initializes the optimization with random samples
        5. Iteratively refines parameters using Expected Improvement
        6. Logs progress and returns optimal parameters

        The optimization uses a two-phase approach:
        - Initialization: Random sampling to explore the parameter space
        - Refinement: Expected Improvement-guided search for optimal values

        Args:
            n_epochs: Number of Bayesian optimization iterations to perform.
                More epochs allow for better convergence but increase runtime.
                Defaults to 20.
            init_samples: Number of random initial samples to collect before
                starting Bayesian optimization. These provide an initial
                understanding of the objective function landscape. Defaults to 5.

        Returns:
            tuple: A tuple (best_hyst, best_ttt) containing:
                - best_hyst (float): Optimal hysteresis value in dB
                - best_ttt (int): Optimal Time-to-Trigger value in ticks

        Raises:
            ValueError: If Bayesian Digital Twins are not trained before calling solve.

        Notes:
            - The hysteresis range is automatically determined based on signal differences
            - The TTT range is constrained by the number of simulation ticks
            - All evaluated configurations are stored in self.score DataFrame
            - Optimization progress is logged showing epoch, parameters, and metrics
        """
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Determine simulation boundaries
        bounds = find_sim_boundary(self.topology, self.new_data)
        if "ue_tracks_generation" in self.mobility_model_params:
            if "params" in self.mobility_model_params["ue_tracks_generation"]:
                if "lat_lon_boundaries" in self.mobility_model_params["ue_tracks_generation"]["params"]:
                    self.mobility_model_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"].update(bounds)

        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        _, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = self._preprocess_simulation_data(full_prediction_df)

        rlf_threshold = RLF_THRESHOLD
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, max(3, num_ticks + 1)]  # Ensure ttt_range[1] > ttt_range[0]

        self.score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        X, y = [], []
        for _ in range(init_samples):
            hyst = np.random.uniform(hyst_range[0], hyst_range[1])
            ttt = np.random.randint(ttt_range[0], ttt_range[1])
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            metric = calculate_mro_metric(attached_df)
            X.append([hyst, ttt])
            y.append(metric)
            self.score.loc[len(self.score)] = [hyst, ttt, metric]

        X = np.array(X)
        y = np.array(y)
        model = self._init_model()
        best_y = y.max()
        best_idx = y.argmax()

        header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
        self.logger.info(header)
        self.logger.info("-" * len(header))

        for i in range(n_epochs):
            if self.suppress_warnings:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X, y)
            else:
                model.fit(X, y)
            cand_hyst = np.random.uniform(hyst_range[0], hyst_range[1], size=100)
            cand_ttt = np.random.randint(ttt_range[0], ttt_range[1], size=100)
            candidates = np.column_stack([cand_hyst, cand_ttt])
            scores = self._expected_improvement(candidates, model, best_y)
            idx = int(np.argmax(scores))
            hyst, ttt = candidates[idx]
            ttt = int(round(ttt))
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            metric = calculate_mro_metric(attached_df)
            self.logger.info(f"{i:<6} {hyst:<14.10f} {ttt:<6} {metric:<12.6f}")
            X = np.vstack([X, [hyst, ttt]])
            y = np.append(y, metric)
            self.score.loc[len(self.score)] = [hyst, ttt, metric]
            if metric > best_y:
                best_y = metric
                best_idx = len(y) - 1

        best_hyst = float(X[best_idx, 0])
        best_ttt = int(round(X[best_idx, 1]))
        self.logger.info(f"\nOptimized Hyst: {best_hyst},\nOptimized TTT: {best_ttt}")
        return best_hyst, best_ttt
