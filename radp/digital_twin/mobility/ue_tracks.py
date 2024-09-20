# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, Generator, List, Any
import itertools

import numpy as np
import pandas as pd

from radp.digital_twin.mobility.mobility import gauss_markov
from radp.common import constants
from radp.digital_twin.utils.gis_tools import GISTools


class MobilityClass(Enum):
    stationary = "stationary"
    pedestrian = "pedestrian"
    cyclist = "cyclist"
    car = "car"


class UETracksGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
        lon_x_dims: int = 100,
        lon_y_dims: int = 100,
        num_ticks: int = 2,
        num_UEs: int = 2,
        alpha: float = 0.5,
        variance: float = 0.8,
        min_lat: float = -90,
        max_lat: float = 90,
        min_lon: float = -180,
        max_lon: float = 180,
        anchor_loc: np.ndarray = None,
        cov_around_anchor: np.ndarray = None,
    ):
        """Arguments:

        lon_x_dims:     The x dimension of the simulation area

        lon_y_dims:     The y dimension of the simulation area

        num_ticks :     The number of simulation ticks to run

        num_UEs   :     The number of users which we want to simulate

        alpha     :     The tuning parameter used to vary the randomness.
        Totally random values are obtained by setting it to 0 and
        linear motion is obtained by setting it to 1

        variance  :     The randomness variance

        min_lat   :     Minimum latitude for conversion of ue tracks into longitude and latitude

        max_lat   :     Maximum latitude for conversion of ue tracks into longitude and latitude

        max_lon   :     Maximum longitude for conversion of ue tracks into longitude and latitude

        min_lon   :     Minimum longitude for conversion of ue tracks into longitude and latitude

        anchor_loc:     The anchoring or concentration points, if seeking to initialize users in clusters.
        Provided as Nx2 array, where N must be divisible by `num_users`.

        cov_around_anchor: The covariance for sampling around anchor points,
        used only if `anchor_loc` is given.

        mobility_class_distribution: Distribution of number of users across mobility classes

        mobility_class_velocities: Average velocities across mobility classes

        mobility_class_velocity_variances: Variances in the average velocities across mobility classes

        """

        self.rng = rng
        self.lon_x_dims = lon_x_dims
        self.lon_y_dims = lon_y_dims
        self.num_ticks = num_ticks
        self.num_UEs = num_UEs
        self.alpha = alpha
        self.variance = variance
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.anchor_loc = anchor_loc
        self.cov_around_anchor = cov_around_anchor
        self.mobility_class_distribution = mobility_class_distribution
        self.mobility_class_velocities = mobility_class_velocities
        self.mobility_class_velocity_variances = mobility_class_velocity_variances

        self.sampled_users_per_mobility_class = self.rng.choice(
            [mobility_class.value for mobility_class in list(self.mobility_class_distribution.keys())],
            size=(self.num_UEs),
            replace=True,
            p=list(self.mobility_class_distribution.values()),
        )

        self.num_users_per_mobility_class: Dict[MobilityClass, int] = {}
        self.velocity_range: Dict[MobilityClass, List[float]] = {}
        self.gauss_markov_models: Dict[MobilityClass, Generator] = {}

        # mapping the count of users and the velocity ranges
        # across for different mobility classes
        for k in self.mobility_class_distribution.keys():
            self.num_users_per_mobility_class[k] = np.count_nonzero(self.sampled_users_per_mobility_class == k.value)
            low = self.mobility_class_velocities[k] - self.mobility_class_velocity_variances[k]
            high = self.mobility_class_velocities[k] + self.mobility_class_velocity_variances[k]
            self.velocity_range[k] = [low, high]

        # mapping the gauss_markov models to their respective mobility classes
        for k in self.mobility_class_distribution.keys():
            self.gauss_markov_models[k] = gauss_markov(
                rng=self.rng,
                num_users=self.num_users_per_mobility_class[k],
                dimensions=(self.lon_x_dims, self.lon_y_dims),
                velocity_mean=self.rng.uniform(
                    low=self.velocity_range[k][0],
                    high=self.velocity_range[k][1],
                    size=self.num_users_per_mobility_class[k],
                ),
                alpha=self.alpha,
                variance=self.variance,
                anchor_loc=self.anchor_loc,
                cov_around_anchor=self.cov_around_anchor,
            )

    def generate(
        self,
    ) -> Generator:
        """
        This method uses the Gauss-Markov Mobility Model to yield a batch of tracks for UEs,
        corresponding to `num_ticks` number of simulation ticks, and the number of UEs
        the user wants to simulate.

        Returns: A List of List of tracks (track is defined as a sequence of (x, y) points).
        The outer list corresponds to ticks, and the inner list corresponds to UEs.
        """
        while True:
            tracks = []
            # Iterating over ticks
            for _ in range(0, self.num_ticks):
                xy_lonlat = []
                # Iterating over UEs, one mobility class at a time
                for k in self.gauss_markov_models.keys():
                    xy = next(self.gauss_markov_models[k])
                    xy_lonlat.extend(xy)
                tracks.append(xy_lonlat)
            yield (tracks)
        # yield converts it to a generator object which is iterable

    def close(self):
        for k in self.gauss_markov_models.keys():
            self.gauss_markov_models[k].close()

    def generate_as_lon_lat_points(
        self,
        rng_seed: int,
        lon_x_dims: int,
        lon_y_dims: int,
        num_ticks: int,
        num_batches: int,
        num_UEs: int,
        alpha: int,
        variance: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
    ) -> pd.DataFrame:
        """
        The mobility data generation method takes in all the parameters required to generate UE tracks
        for a specified number of batches

        The UETracksGenerator uses the Gauss-Markov Mobility Model to yields batch of tracks for UEs,
        corresponding to `num_ticks` number of simulation ticks, and the number of UEs
        the user wants to simulate.

        Using the UETracksGenerator, the UE tracks are returned in form of a dataframe
        The Dataframe is arranged as follows:

        +------------+------------+-----------+------+
        | mock_ue_id | lon        | lat       | tick |
        +============+============+===========+======+
        |   0        | 102.219377 | 33.674572 |   0  |
        |   1        | 102.415954 | 33.855534 |   0  |
        |   2        | 102.545935 | 33.878075 |   0  |
        |   0        | 102.297766 | 33.575942 |   1  |
        |   1        | 102.362725 | 33.916477 |   1  |
        |   2        | 102.080675 | 33.832793 |   1  |
        +------------+------------+-----------+------+
        """

        ue_tracks_generator = UETracksGenerator(
            rng=np.random.default_rng(rng_seed),
            lon_x_dims=lon_x_dims,
            lon_y_dims=lon_y_dims,
            num_ticks=num_ticks,
            num_UEs=num_UEs,
            alpha=alpha,
            variance=variance,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            mobility_class_distribution=mobility_class_distribution,
            mobility_class_velocities=mobility_class_velocities,
            mobility_class_velocity_variances=mobility_class_velocity_variances,
        )

        for _num_batches, xy_batches in enumerate(ue_tracks_generator.generate()):
            ue_tracks_dataframe_dict: Dict[Any, Any] = {}

            # Extract the xy (lon, lat) points from each batch to use it in the mobility dataframe
            # mock_ue_id, tick, lat, lon
            mock_ue_id = []
            ticks = []
            lon: List[float] = []
            lat: List[float] = []

            tick = 0
            for xy_batch in xy_batches:
                lon_lat_pairs = GISTools.converting_xy_points_into_lonlat_pairs(
                    xy_points=xy_batch,
                    x_dim=lon_x_dims,
                    y_dim=lon_y_dims,
                    min_longitude=min_lon,
                    max_longitude=max_lon,
                    min_latitude=min_lat,
                    max_latitude=max_lat,
                )

                # Build list for each column/row for the UE Tracks dataframe
                lon.extend(xy_points[0] for xy_points in lon_lat_pairs)
                lat.extend(xy_points[1] for xy_points in lon_lat_pairs)
                mock_ue_id.extend([i for i in range(num_UEs)])
                ticks.extend(list(itertools.repeat(tick, num_UEs)))
                tick += 1

            # Build dict for each column/row for the UE Tracks dataframe
            ue_tracks_dataframe_dict[constants.MOCK_UE_ID] = mock_ue_id
            ue_tracks_dataframe_dict[constants.LONGITUDE] = lon
            ue_tracks_dataframe_dict[constants.LATITUDE] = lat
            ue_tracks_dataframe_dict[constants.TICK] = ticks

            # Yield each batch as a dataframe
            yield pd.DataFrame(ue_tracks_dataframe_dict)

            num_batches -= 1
            if num_batches == 0:
                break
