import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from notebooks.radp_library import find_sim_boundary, get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class BayesianMRO(MobilityRobustnessOptimization):
    """Optimize hysteresis and TTT using Bayesian optimization."""

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict[str, Any]],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
        model_type: str = "gpr",
    ):
        super().__init__(mobility_model_params, topology, new_data=None, bdt=bdt)
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _expected_improvement(self, X: np.ndarray, model, best_y: float) -> np.ndarray:
        if self.model_type == "xgboost":
            y_pred = model.predict(X)
            return y_pred - best_y
        else:
            mu, std = model.predict(X, return_std=True)
            std = np.maximum(std, 1e-9)
            Z = (mu - best_y) / std
            return (mu - best_y) * norm.cdf(Z) + std * norm.pdf(Z)

    def _init_model(self):
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
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Determine simulation boundaries
        bounds = find_sim_boundary(self.topology, self.new_data)
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

        for _ in range(n_epochs):
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
