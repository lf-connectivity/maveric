# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, Generator, List

import numpy as np

from radp.digital_twin.mobility.mobility import gauss_markov


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
