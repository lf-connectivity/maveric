# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd

from radp.digital_twin.utils.constants import CELL_ID, LOC_X, LOC_Y  # noqa: E402
from radp.digital_twin.utils.gis_tools import GISTools  # noqa: E402


class optParams(Enum):
    PWR = "enb_tx_power"
    EL = "el_boresight_angle"
    AZ = "az_boresight_angle"


class CcoMetric(Enum):
    PIXEL = "pixel"
    CELL = "cell"


class CcoEngine:
    @staticmethod
    def rf_to_coverage_dataframe(
        rf_dataframe: pd.DataFrame,
        id_field: str = CELL_ID,
        loc_x_field: str = LOC_X,
        loc_y_field: str = LOC_Y,
        rx_dbm_field: str = "rsrp_dbm",
        sinr_db_field: str = "sinr_db",
        lambda_: float = 0.5,
        weak_coverage_threshold: float = -100,
        over_coverage_threshold: float = 0,
        growth_rate: float = 1,
    ) -> pd.DataFrame:

        if lambda_ <= 0 or lambda_ >= 1:
            raise ValueError("lambda_ must be between 0 and 1 (noninclusive)")

        coverage_dataframe = pd.DataFrame(
            data={
                loc_x_field: rf_dataframe[loc_x_field],
                loc_y_field: rf_dataframe[loc_y_field],
                rx_dbm_field: rf_dataframe[rx_dbm_field],
                sinr_db_field: rf_dataframe[sinr_db_field],
                id_field: rf_dataframe[id_field],
                "weakly_covered": False,
                "weak_coverage": 0.0,
                "overly_covered": False,
                "over_coverage": 0.0,
            }
        )
        h = rf_dataframe[rx_dbm_field] - weak_coverage_threshold
        g = rf_dataframe[sinr_db_field] - over_coverage_threshold

        # weakly covered: rsrp <= weak_coverage_threshold
        # overly covered: (rsrp > weak_coverage_threshold) & (sinr < over_coverage_threshold)
        coverage_dataframe["weakly_covered"] = h <= 0
        coverage_dataframe["weak_coverage"] = np.minimum(0, h)
        coverage_dataframe["overly_covered"] = (h > 0) & (g <= 0)
        coverage_dataframe["over_coverage"] = np.minimum(0, g)
        coverage_dataframe["covered"] = ~coverage_dataframe["weakly_covered"] & ~coverage_dataframe["overly_covered"]

        # TODO : deprecate the below notion
        # soft_weak_coverage = sigmoid(h, growth_rate)
        # soft_over_coverage = sigmoid(g, growth_rate)
        coverage_dataframe["soft_weak_coverage"] = 1000 * np.tanh(0.05 * growth_rate * h)
        coverage_dataframe["soft_over_coverage"] = 1000 * np.tanh(0.05 * growth_rate * g)
        coverage_dataframe["network_coverage_utility"] = (
            lambda_ * coverage_dataframe["soft_weak_coverage"]
            + (1 - lambda_) * coverage_dataframe["soft_over_coverage"]
        )
        return coverage_dataframe

    @staticmethod
    def get_weak_over_coverage_percentages(
        coverage_dataframe: pd.DataFrame,
    ) -> Tuple[float, float]:
        n_points = len(coverage_dataframe.index)
        weak_coverage_percent = 100 * coverage_dataframe["weakly_covered"].sum() / n_points
        over_coverage_percent = 100 * coverage_dataframe["overly_covered"].sum() / n_points
        return weak_coverage_percent, over_coverage_percent

    @staticmethod
    def get_cco_objective_value(
        coverage_dataframe: pd.DataFrame,
        active_ids_list: List = None,
        id_field: str = CELL_ID,
        cco_metric: CcoMetric = CcoMetric.PIXEL,
        traffic_model_df: pd.DataFrame = None,
    ) -> float:
        """Computes the CCO metric (single value)

        Technical outline: https://fb.quip.com/GxXoA9SGavWU

        Args:
            coverage_dataframe: Columns include: 'latitude', 'longitude', 'pci', 'weakly_covered',
                 'overly_covered', 'covered', 'network_coverage_utility'.
            active_ids_list: The CCO metric does not need ot be computed over all sectors.
                This list is a subset of the pci list obtained from sites.
            cco_metric: The CCO metric can be computed at different levels. At a PIXEL level
                each pixel (ie UE) contributes directly to the output. At a CELL level, the CCO
                metric is first aggregated to the cells and subsequently a single value is computed
                from the cell-level values.
            traffic_model_df:Dataframe containing the Aggregated columns(Average,maximum and minimum) of Traffic.

        Returns:
            A single CCO metric (aka network coverage utility) representing the "coverage" of a
            region represented by coverage_dataframe. This value should be between -1 and +1.
        """
        if traffic_model_df is not None:
            augmented_coverage_df_with_normalized_traffic_model = (
                CcoEngine.augment_coverage_df_with_normalized_traffic_model(
                    traffic_model_df,
                    "avg_of_average_egress_kbps_across_all_time",
                    coverage_dataframe,
                )
            )
            augmented_coverage_df_with_normalized_traffic_model["network_coverage_utility"] = (
                augmented_coverage_df_with_normalized_traffic_model["normalized_traffic_statistic"]
                * coverage_dataframe["network_coverage_utility"]
            )
            coverage_dataframe["network_coverage_utility"] = augmented_coverage_df_with_normalized_traffic_model[
                "network_coverage_utility"
            ]

        if active_ids_list is None:
            return -math.inf

        active_df = coverage_dataframe[coverage_dataframe[id_field].isin(active_ids_list)]
        active_sector_metric = active_df.groupby(id_field)["network_coverage_utility"]

        if cco_metric == CcoMetric.PIXEL:
            weights = active_sector_metric.count()
        elif cco_metric == CcoMetric.CELL:
            weights = np.ones(len(active_sector_metric))

        cco_objective_value = weights.dot(active_sector_metric.mean()) / sum(weights)
        return cco_objective_value

    @staticmethod
    def add_tile_x_and_tile_y(
        coverage_dataframe: pd.DataFrame,
        loc_x_field: str = LOC_X,
        loc_y_field: str = LOC_Y,
        level: int = 18,  # level : int =18 # TODO(sajalkaushik17) : THIS IS BROKEN, FIX
    ) -> pd.DataFrame:
        """Method to add Bing tile_x and tile_y columns (at specified resolution level)
        corresponding to the loc_x_field and loc_y_field in input coverage dataframe.

        Args:
            coverage_dataframe: coverage dataframe with loc_x_field and loc_y_field

        Returns:
            Dataframe with tile_x and tile_y columns appended

        """
        tile_coords = list(zip(coverage_dataframe[loc_x_field], coverage_dataframe[loc_y_field]))

        coverage_dataframe["tile_x"], coverage_dataframe["tile_y"] = zip(
            *map(
                GISTools.make_tile,
                tile_coords,
            )
        )
        return coverage_dataframe

    @staticmethod
    def augment_coverage_df_with_normalized_traffic_model(
        traffic_model_df: pd.DataFrame,
        desired_traffic_statistic_col: str,
        coverage_df: pd.DataFrame,
        loc_x_field: str = LOC_X,
        loc_y_field: str = LOC_Y,
    ) -> pd.DataFrame:
        """Given a traffic_model_df containing desired traffic statistics,
        we augment the given coverage_df with normalized statistics over its region.

        Args:
            traffic_model_df: Dataframe containing the Aggregated columns(Average,maximum and minimum) of Traffic.
            desired_traffic_statistics: The column name on which the user wants to augment the given coverage_df.
            coverage_df: Dataframe having the columns LOC_X(longitude), LOC_Y(latitude), "weak_coverage",
                "over_coverage" and other columns.


        Returns:
                A dataframe with columns
                "tile_x",
                "tile_y",
                loc_x_field,
                loc_y_field,
                "normalized_traffic_statistic",
                "weak_coverage",
                "over_coverage",
        """

        sum_of_desired_traffic_statistic_across_all_tiles = traffic_model_df[desired_traffic_statistic_col].sum()
        traffic_model_df["normalized_traffic_statistic"] = (
            traffic_model_df[desired_traffic_statistic_col] / sum_of_desired_traffic_statistic_across_all_tiles
        )
        coverage_dataframe_with_bing_tiles = CcoEngine.add_tile_x_and_tile_y(coverage_df)
        augmented_coverage_df_with_normalized_traffic_model = pd.merge(
            traffic_model_df,
            coverage_dataframe_with_bing_tiles,
            on=["tile_x", "tile_y"],
            how="right",
        )
        return augmented_coverage_df_with_normalized_traffic_model[
            [
                "tile_x",
                "tile_y",
                loc_x_field,
                loc_y_field,
                "normalized_traffic_statistic",
                "weak_coverage",
                "over_coverage",
            ]
        ]

    @staticmethod
    def traffic_normalized_cco_metric(coverage_dataframe: pd.DataFrame) -> float:
        """Returns the sum of the dot products of the column "normalized_traffic_statistic" with "weak_coverage"
        and "over_coverage" columns.

        Args:
            coverage_dataframe: pandas dataframe

        """
        # only one of weak_coverage and over_coverage can be simultaneously 1
        # so, the logic below does not double count
        return (
            coverage_dataframe["normalized_traffic_statistic"] * coverage_dataframe["weak_coverage"]
            + coverage_dataframe["normalized_traffic_statistic"] * coverage_dataframe["over_coverage"]
        ).sum()
