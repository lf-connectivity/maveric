# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from apps.coverage_capacity_optimization import constants
from apps.coverage_capacity_optimization.cco_engine import CcoEngine, CcoMetric
from radp.digital_twin.utils.cell_selection import perform_attachment

load_dotenv()


from radp.client.client import RADPClient  # noqa: E402
from radp.client.helper import RADPHelper, SimulationStatus  # noqa: E402

# instantiate RADP client
radp_client = RADPClient()
radp_helper = RADPHelper(radp_client)


class DgpcoCCO:
    def __init__(
        self,
        topology: pd.DataFrame,
        valid_configuration_values: Dict[str, List[float]],
        bayesian_digital_twin_id: str,
        ue_data: pd.DataFrame,
        config: pd.DataFrame,
    ):
        """

        @args
            `topology_df` : 1 unique cell per row, contains at least the columns
                [cell_lat, cell_lon, cell_el_deg, cell_az_deg, cell_id]
            `valid_configuration_values`: dictionary containing valid values for settings,
                keyed by attribute id that must correspond to a column in site_config_df,
                for e.g. `cell_el_deg`

        """
        # TODO (paulvarkey) : ensure consistencies :
        # 1. that bayesian_digital_twin has x_columns
        #    [CELL_LAT, CELL_LON, CELL_EL_DEG, "distance", RELATIVE_BEARING]
        # 2. that bayesian_digital_twin.num_cells == len(site_config_df)

        self.topology = topology
        self.num_cells = len(self.topology)
        self.valid_configuration_values = valid_configuration_values
        self.ue_data = ue_data
        self.config = config

        # TODO: refactor to have a helper generate this
        # set up simulation event
        self.simulation_event: Dict[str, Any] = {
            "simulation_time_interval_seconds": 1,
            "ue_tracks": {"ue_data_id": "ue_data_1"},
            "rf_prediction": {"model_id": bayesian_digital_twin_id, "config_id": 1},
        }

    def _calc_metric(
        self,
        lambda_: float = 0.5,
        weak_coverage_threshold: float = -90,
        over_coverage_threshold: float = 0,
    ):
        """Calculate the CCO metric given the current state of the config."""
        # update the simulation event
        self.simulation_event["rf_prediction"]["config_id"] += 1

        # run simulation
        simulation_id = radp_client.simulation(
            simulation_event=self.simulation_event,
            ue_data=self.ue_data,
            config=self.config,
        )["simulation_id"]

        simulation_status: SimulationStatus = radp_helper.resolve_simulation_status(
            simulation_id,
            wait_interval=1,
            max_attempts=100,
            verbose=False,
        )
        if not simulation_status.success:
            raise Exception(
                f"Exception occurred while running simulation '{simulation_id}': {simulation_status.error_message}"
            )

        # consume simulation results
        rf_dataframe = radp_client.consume_simulation_output(simulation_id)

        print(rf_dataframe)

        # run cell attachment
        cell_selected_rf_dataframe = perform_attachment(rf_dataframe, self.topology)

        print(cell_selected_rf_dataframe)

        # get CCO coverage dataframe
        coverage_dataframe = CcoEngine.rf_to_coverage_dataframe(
            rf_dataframe=cell_selected_rf_dataframe,
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )

        # get CCO objective
        cco_objective = CcoEngine.get_cco_objective_value(
            coverage_dataframe=coverage_dataframe,
            active_ids_list=coverage_dataframe[constants.CELL_ID].unique(),
            cco_metric=CcoMetric.PIXEL,
        )
        return cell_selected_rf_dataframe, coverage_dataframe, cco_objective

    def run(
        self,
        num_epochs: int = 0,
        lambda_: float = 0.5,
        weak_coverage_threshold: float = -90,
        over_coverage_threshold: float = 0,
        seed: int = 0,
        epsilon: float = 0,
        opt_delta: Tuple = (-4, -3, -2, -1, 0, 1, 2, 3, 4),
    ):
        """Run the dGPCO algorithm"""

        def _single_step(
            cell_id: str,
            orig_el_deg: float,
            cur_el_deg: float,
        ):
            """Single step of DGPCO."""

            # calculate new metric
            (
                current_rf_dataframe,
                current_coverage_dataframe,
                current_cco_objective,
            ) = self._calc_metric(
                lambda_=lambda_,
                weak_coverage_threshold=weak_coverage_threshold,
                over_coverage_threshold=over_coverage_threshold,
            )

            rf_dataframe_vs_delta = []
            coverage_dataframe_vs_delta = []
            cco_objective_value_vs_delta = []
            elevs_tried = []

            # pull the cell config index
            cell_config_index = self.config.index[self.config["cell_id"] == cell_id][0]

            orig_el_idx = self.valid_configuration_values[constants.CELL_EL_DEG].index(
                orig_el_deg
            )
            cur_el_idx = self.valid_configuration_values[constants.CELL_EL_DEG].index(
                cur_el_deg
            )

            for d in opt_delta:
                new_el_idx = orig_el_idx + d

                if new_el_idx == cur_el_idx:
                    # we do not want to check current value
                    continue

                if new_el_idx < 0 or new_el_idx >= len(
                    self.valid_configuration_values[constants.CELL_EL_DEG]
                ):
                    # we do not want to wrap around, since that would not be a neighboring tilt
                    continue

                new_el = self.valid_configuration_values[constants.CELL_EL_DEG][
                    new_el_idx
                ]

                # update the cell config el_degree
                self.config.loc[cell_config_index, constants.CELL_EL_DEG] = new_el

                # predict CCO metric for changed configuration
                new_rf_dataframe, coverage_dataframe, cco_objective = self._calc_metric(
                    lambda_=lambda_,
                    weak_coverage_threshold=weak_coverage_threshold,
                    over_coverage_threshold=over_coverage_threshold,
                )

                elevs_tried.append(new_el)
                rf_dataframe_vs_delta.append(new_rf_dataframe)
                coverage_dataframe_vs_delta.append(coverage_dataframe)
                cco_objective_value_vs_delta.append(cco_objective)

            if np.random.uniform() < epsilon:
                # pick randomly
                best_delta_index = np.random.randint(0, len(elevs_tried))
            else:
                # pick best
                best_delta_index = int(np.argmax(cco_objective_value_vs_delta))

            best_el = elevs_tried[best_delta_index]
            best_rf_dataframe = rf_dataframe_vs_delta[best_delta_index]
            best_coverage_dataframe = coverage_dataframe_vs_delta[best_delta_index]
            best_cco_objective_value = cco_objective_value_vs_delta[best_delta_index]

            # if existing was best, don't change
            if current_cco_objective >= best_cco_objective_value:
                best_el = cur_el_deg
                best_rf_dataframe = current_rf_dataframe
                best_coverage_dataframe = current_coverage_dataframe
                best_cco_objective_value = current_cco_objective

            logging.info(
                f"...cell_id={cell_id}, orig_el_deg={orig_el_deg}, cur_el_deg={cur_el_deg}, "
                f"elevs_tried={elevs_tried}, best_el_tried={elevs_tried[best_delta_index]}, best_el={best_el}"
            )

            # update the cell config el_degree
            self.config.iloc[cell_config_index].cell_el_deg = best_el

            return (
                best_el,
                best_rf_dataframe,
                best_coverage_dataframe,
                best_cco_objective_value,
            )

        # TODO (paulvarkey) : remove, and set externally!
        np.random.seed(seed)

        rf_dataframe_per_epoch = []
        coverage_dataframe_per_epoch = []
        cco_objective_per_epoch = []

        # predict CCO metric for initial configuration
        rf_dataframe, coverage_dataframe, cco_objective = self._calc_metric(
            lambda_=lambda_,
            weak_coverage_threshold=weak_coverage_threshold,
            over_coverage_threshold=over_coverage_threshold,
        )

        rf_dataframe_per_epoch.append(rf_dataframe)
        coverage_dataframe_per_epoch.append(coverage_dataframe)
        cco_objective_per_epoch.append(cco_objective)

        logging.info(f"Before dGPCO starts, CCO metric = {cco_objective_per_epoch[0]}")

        opt_per_epoch = np.zeros([num_epochs + 1, self.num_cells])

        for idx in range(self.num_cells):
            opt_per_epoch[0, idx] = self.config.iloc[[idx]].cell_el_deg

        # Save original tilts
        orig_cell_el_deg = self.config.cell_el_deg.copy()
        continuously_unchanged = 0

        for epoch in range(1, num_epochs + 1):
            # Roundrobin "site" (cell-sector) selection
            cell_idx = (epoch - 1) % self.num_cells

            orig_el_deg = orig_cell_el_deg.iloc[cell_idx]

            # current elevation
            cur_el_deg = self.config.iloc[cell_idx].cell_el_deg
            cell_id = self.config.iloc[cell_idx].cell_id

            logging.info(f"\nIn epoch: {epoch:02}/{num_epochs}...")

            # Perform one step of DGPCO
            (
                new_opt_el,
                new_rf_dataframe,
                new_coverage_dataframe,
                new_cco_objective_value,
            ) = _single_step(
                cell_id=cell_id,
                orig_el_deg=orig_el_deg,
                cur_el_deg=cur_el_deg,
            )

            # update elevation tilt
            self.config.loc[cell_idx, constants.CELL_EL_DEG] = new_opt_el

            if new_opt_el != cur_el_deg:
                logging.info(
                    f"...changing elevaton tilt change for cell {cell_id} "
                    f"from {cur_el_deg} to {new_opt_el}, to achieve new "
                    f"CCO metric = {new_cco_objective_value:.3f}"
                )
                continuously_unchanged = 0
            else:
                logging.info(f"...keeping same tilt for cell {cell_id} at {cur_el_deg}")
                continuously_unchanged += 1

            for idx in range(self.num_cells):
                opt_per_epoch[epoch, idx] = self.config.iloc[idx].cell_el_deg

            rf_dataframe_per_epoch.append(new_rf_dataframe)
            coverage_dataframe_per_epoch.append(new_coverage_dataframe)
            cco_objective_per_epoch.append(new_cco_objective_value)

            if continuously_unchanged == self.num_cells:
                logging.info(
                    "\nno change for any cell after 1 full round robin (optimization converged)...."
                    f"\ntrimming {len(opt_per_epoch) - len(cco_objective_per_epoch)} values from opt_per_epoch...."
                    f"\nexiting...."
                )
                # if exiting before epochs done, trim opt_per_epoch
                opt_per_epoch = np.delete(
                    opt_per_epoch,
                    slice(len(cco_objective_per_epoch), len(opt_per_epoch)),
                    0,  # axis 0 represents epochs
                )
                break

        return (
            rf_dataframe_per_epoch,
            coverage_dataframe_per_epoch,
            cco_objective_per_epoch,
            opt_per_epoch,
        )
