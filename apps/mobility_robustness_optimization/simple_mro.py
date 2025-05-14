import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from gpytorch.utils.warnings import NumericalWarning

from notebooks.radp_library import get_ue_data
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.cell_selection import find_hyst_diff, perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD

from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric


class SimpleMRO(MobilityRobustnessOptimization):
    """
    Iteratively optimizes the cell attachment strategy to find the best MRO metric.
    Currently, 'perform_attachment' has no parameters to optimize, so this function
    will focus on evaluating its current implementation. This setup is ready to be
    expanded for parameter optimization in future developments.
    """

    def __init__(
        self,
        mobility_model_params: Dict[str, Dict[str, Any]],
        topology: pd.DataFrame,
        bdt: Optional[Dict[str, BayesianDigitalTwin]] = None,
    ):
        super().__init__(mobility_model_params, topology, bdt)

    def solve(self, n_epochs=100):
        """
        Solve the mobility robustness optimization problem.
        """
        # Ensure Bayesian Digital Twins are trained before proceeding
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")

        # Generate and preprocess simulation data
        self.simulation_data = get_ue_data(self.mobility_model_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        if self.topology["cell_id"].dtype == int:
            self.topology["cell_id"] = self.topology["cell_id"].apply(lambda x: f"cell_{int(x)}")

        # Predict power and perform attachment
        predictions, full_prediction_df = self._predictions(self.simulation_data)
        self.simulation_data = full_prediction_df
        self.simulation_data = self._preprocess_simulation_data(self.simulation_data)

        # epochs = 100
        epochs = n_epochs
        hyst = 0.01
        ttt = 5
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
        max_diff = find_hyst_diff(self.simulation_data)
        num_ticks = self.simulation_data["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks + 1]

        # Suppress the specific NumericalWarning from gpytorch
        warnings.filterwarnings("ignore", category=NumericalWarning)

        self.score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
        print(header)
        print("-" * len(header))
        self.score.loc[len(self.score)] = [hyst, ttt, calculate_mro_metric(attached_df)]
        for i in range(epochs):
            while True:
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                if ttt not in self.score["ttt"].values or hyst not in self.score["hyst"].values:
                    break
            # Perform attachment and calculate MRO Metric
            attached_df = perform_attachment_hyst_ttt(self.simulation_data, hyst, ttt, rlf_threshold)
            mro_metric = calculate_mro_metric(attached_df)

            # Store the data in the score DataFrame
            self.score.loc[len(self.score)] = [hyst, ttt, mro_metric]
            print(f"{i:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")

        print(
            f"""\nOptimized Hyst: {self.score.loc[self.score['score'].idxmax(), 'hyst']},
            Optimized TTT: {int(self.score.loc[self.score['score'].idxmax(), 'ttt'])}"""
        )
        return self.score.loc[self.score["score"].idxmax(), "hyst"], int(
            self.score.loc[self.score["score"].idxmax(), "ttt"]
        )
