from .mobility_robustness_optimization import MobilityRobustnessOptimization, calculate_mro_metric
from radp.digital_twin.utils.constants import RLF_THRESHOLD
from notebooks.radp_library import get_ue_data
from radp.digital_twin.utils.cell_selection import (perform_attachment_hyst_ttt, find_hyst_diff)
import pandas as pd
import numpy as np
import warnings
from gpytorch.utils.warnings import NumericalWarning

class SimpleMRO(MobilityRobustnessOptimization):
    """
    Iteratively optimizes the cell attachment strategy to find the best MRO metric.
    Currently, 'perform_attachment' has no parameters to optimize, so this function
    will focus on evaluating its current implementation. This setup is ready to be
    expanded for parameter optimization in future developments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization can be done here if needed
    
    def solve(self):
        """
        Solve the mobility robustness optimization problem.
        """
        # Ensure Bayesian Digital Twins are trained before proceeding
        if not self.bayesian_digital_twins:
            raise ValueError("Bayesian Digital Twins are not trained. Train the models before calculating metrics.")
        
        # Generate and preprocess simulation data
        self.simulation_data = get_ue_data(self.mobility_params)
        self.simulation_data = self.simulation_data.rename(columns={"lat": "latitude", "lon": "longitude"})

        # Predict power and perform attachment
        predictions, full_prediction_df = self._predictions(self.simulation_data)
        df = full_prediction_df
        df = self._preprocess_simulation_data(df)

        epochs = 100
        hyst = 0.01
        ttt = 5
        rlf_threshold = RLF_THRESHOLD

        attached_df = perform_attachment_hyst_ttt(df, hyst, ttt, rlf_threshold)
        max_diff = find_hyst_diff(df)
        num_ticks = df["tick"].nunique()
        hyst_range = [0, max_diff]
        ttt_range = [2, num_ticks+1]

        # Suppress the specific NumericalWarning from gpytorch
        warnings.filterwarnings("ignore", category=NumericalWarning)

        score = pd.DataFrame(columns=["hyst", "ttt", "score"])

        header = f"{'Epoch':<6} {'Hyst':<14} {'TTT':<6} {'MRO Metric':<12}"
        print(header)
        print("-" * len(header))
        score.loc[len(score)] = [hyst, ttt, calculate_mro_metric(attached_df)]
        for i in range(epochs):
            while True:
                hyst = np.random.uniform(hyst_range[0], hyst_range[1])
                ttt = np.random.randint(ttt_range[0], ttt_range[1])
                if ttt not in score["ttt"].values or hyst not in score["hyst"].values:
                    break
            # Perform attachment and calculate MRO Metric
            attached_df = perform_attachment_hyst_ttt(df, hyst, ttt, rlf_threshold)
            mro_metric = calculate_mro_metric(attached_df)

            # Store the data in the score DataFrame
            score.loc[len(score)] = [hyst, ttt, mro_metric]
            print(f"{i:<6} {hyst:<14.10f} {ttt:<6} {mro_metric:<12.6f}")
        
        print(f"\nOptimized Hyst: {score.loc[score['score'].idxmax(), 'hyst']}, Optimized TTT: {int(score.loc[score['score'].idxmax(), 'ttt'])}")        
        return score.loc[score["score"].idxmax(), "hyst"], int(score.loc[score["score"].idxmax(), "ttt"])
