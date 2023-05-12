# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# coding: utf-8
#
#  Copyright (C) 2008-2010 Istituto per l'Interscambio Scientifico I.S.I.
#  You can contact us by email (isi@isi.it) or write to:
#  ISI Foundation, Viale S. Severo 65, 10133 Torino, Italy.
#
#  This program was written by André Panisson <panisson@gmail.com>
#
"""
Created on Jan 24, 2012

@author: André Panisson
@contact: panisson@gmail.com
@organization: ISI Foundation, Torino, Italy
@url : https://github.com/panisson/pymobility/blob/master/src/pymobility/models/mobility.py
"""
import logging
from typing import Tuple

import numpy as np


# define a Uniform Distribution
def U(rng, MIN, MAX, SAMPLES):
    return rng.random(SAMPLES.shape) * (MAX - MIN) + MIN


# define a Truncated Power Law Distribution
def P(rng, ALPHA, MIN, MAX, SAMPLES):
    return ((MAX ** (ALPHA + 1.0) - 1.0) * rng.random(SAMPLES.shape) + 1.0) ** (1.0 / (ALPHA + 1.0))


# *************** Palm state probability **********************
def pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions):
    alpha1 = ((pause_high + pause_low) * (speed_high - speed_low)) / (2 * np.log(speed_high / speed_low))
    delta1 = np.sqrt(np.sum(np.square(dimensions)))
    return alpha1 / (alpha1 + delta1)


# *************** Palm residual ******************************
def residual_time(rng, mean, delta, shape=(1,)):
    t1 = mean - delta
    t2 = mean + delta
    u = rng.random(*shape)
    residual = np.zeros(shape)
    if delta != 0.0:
        case_1_u = u < (2.0 * t1 / (t1 + t2))
        residual[case_1_u] = u[case_1_u] * (t1 + t2) / 2.0
        residual[np.logical_not(case_1_u)] = t2 - np.sqrt((1.0 - u[np.logical_not(case_1_u)]) * (t2 * t2 - t1 * t1))
    else:
        residual = u * mean
    return residual


# *********** Initial speed ***************************
def initial_speed(rng, speed_mean, speed_delta, shape=(1,)):
    v0 = speed_mean - speed_delta
    v1 = speed_mean + speed_delta
    u = rng.random(shape)
    return pow(v1, u) / pow(v0, u - 1)


def init_random_waypoint(rng, nr_nodes, dimensions, speed_low, speed_high, pause_low, pause_high):
    ndim = len(dimensions)
    positions = np.empty((nr_nodes, ndim))
    waypoints = np.empty((nr_nodes, ndim))
    speed = np.empty(nr_nodes)
    pause_time = np.empty(nr_nodes)

    speed_low = float(speed_low)
    speed_high = float(speed_high)

    moving = np.ones(nr_nodes)
    speed_mean, speed_delta = (speed_low + speed_high) / 2.0, (speed_high - speed_low) / 2.0
    pause_mean, pause_delta = (pause_low + pause_high) / 2.0, (pause_high - pause_low) / 2.0

    # steady-state pause probability for Random Waypoint
    q0 = pause_probability_init(pause_low, pause_high, speed_low, speed_high, dimensions)

    for i in range(nr_nodes):
        while True:
            z1 = rng.random(ndim) * np.array(dimensions)
            z2 = rng.random(ndim) * np.array(dimensions)

            if rng.random() < q0:
                moving[i] = 0.0
                break
            else:
                # r is a ratio of the length of the randomly chosen path over
                # the length of a diagonal across the simulation area
                r = np.sqrt(np.sum((z2 - z1) ** 2) / np.sum(np.array(dimensions) ** 2))
                if rng.random() < r:
                    moving[i] = 1.0
                    break

        positions[i] = z1
        waypoints[i] = z2

    # steady-state positions
    # initially the node has traveled a proportion u2 of the path from (x1,y1) to (x2,y2)
    u2 = rng.random(positions.shape)
    positions = u2 * positions + (1 - u2) * waypoints

    # steady-state speed and pause time
    paused_bool = moving == 0.0
    paused_idx = np.where(paused_bool)[0]
    pause_time[paused_idx] = residual_time(rng, pause_mean, pause_delta, paused_idx.shape)
    speed[paused_idx] = 0.0

    moving_bool = np.logical_not(paused_bool)
    moving_idx = np.where(moving_bool)[0]
    pause_time[moving_idx] = 0.0
    speed[moving_idx] = initial_speed(rng, speed_mean, speed_delta, moving_idx.shape)

    return positions, waypoints, speed, pause_time


class RandomWaypoint(object):
    def __init__(self, rng, nr_nodes, dimensions, velocity=(0.1, 1.0), wt_max=None):
        """
        Random Waypoint model.

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

        keyword arguments:

          *velocity*:
            Tuple of Integers, the minimum and maximum values for node velocity.

          *wt_max*:
            Integer, the maximum wait time for node pauses.
            If wt_max is 0 or None, there is no pause time.
        """

        self.rng = rng
        self.nr_nodes = nr_nodes
        self.dimensions = dimensions
        self.velocity = velocity
        self.wt_max = wt_max
        self.init_stationary = True

    def __iter__(self):
        ndim = len(self.dimensions)
        MIN_V, MAX_V = self.velocity

        wt_min = 0.0

        if self.init_stationary:
            positions, waypoints, velocity, wt = init_random_waypoint(
                self.rng,
                self.nr_nodes,
                self.dimensions,
                MIN_V,
                MAX_V,
                wt_min,
                (self.wt_max if self.wt_max is not None else 0.0),
            )
        else:
            NODES = np.arange(self.nr_nodes)
            positions = U(
                self.rng,
                np.zeros(ndim),
                np.array(self.dimensions),
                np.dstack((NODES,) * ndim)[0],
            )
            waypoints = U(
                self.rng,
                np.zeros(ndim),
                np.array(self.dimensions),
                np.dstack((NODES,) * ndim)[0],
            )
            wt = np.zeros(self.nr_nodes)
            velocity = U(self.rng, MIN_V, MAX_V, NODES)

        # assign nodes' movements (direction * node velocity)
        direction = waypoints - positions
        direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]

        while True:
            # update node position
            positions += direction * velocity[:, np.newaxis]
            # calculate distance to waypoint
            d = np.sqrt(np.sum(np.square(waypoints - positions), axis=1))
            # update info for arrived nodes
            arrived = np.where(np.logical_and(d <= velocity, wt <= 0.0))[0]

            # step back for nodes that surpassed waypoint
            positions[arrived] = waypoints[arrived]

            if self.wt_max:
                velocity[arrived] = 0.0
                wt[arrived] = U(self.rng, 0, self.wt_max, arrived)
                # update info for paused nodes
                wt[np.where(velocity == 0.0)[0]] -= 1.0
                # update info for moving nodes
                arrived = np.where(np.logical_and(velocity == 0.0, wt < 0.0))[0]

            if arrived.size > 0:
                waypoints[arrived] = U(
                    self.rng,
                    np.zeros(ndim),
                    np.array(self.dimensions),
                    np.zeros((arrived.size, ndim)),
                )
                velocity[arrived] = U(self.rng, MIN_V, MAX_V, arrived)

                new_direction = waypoints[arrived] - positions[arrived]
                direction[arrived] = new_direction / np.linalg.norm(new_direction, axis=1)[:, np.newaxis]

            self.velocity = velocity
            self.wt = wt
            yield positions


class StochasticWalk(object):
    def __init__(
        self,
        rng,
        nr_nodes,
        dimensions,
        FL_DISTR,
        VELOCITY_DISTR,
        WT_DISTR=None,
        border_policy="reflect",
    ):
        """
        Base implementation for models with direction uniformly chosen from [0,pi]:
        random_direction, random_walk, truncated_levy_walk

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

          *FL_DISTR*:
            A function that, given a set of samples,
             returns another set with the same size of the input set.
            This function should implement the distribution of flight lengths
             to be used in the model.

          *VELOCITY_DISTR*:
            A function that, given a set of flight lengths,
             returns another set with the same size of the input set.
            This function should implement the distribution of velocities
             to be used in the model, as random or as a function of the flight lengths.

        keyword arguments:

          *WT_DISTR*:
            A function that, given a set of samples,
             returns another set with the same size of the input set.
            This function should implement the distribution of wait times
             to be used in the node pause.
            If WT_DISTR is 0 or None, there is no pause time.

          *border_policy*:
            String, either 'reflect' or 'wrap'. The policy that is used when the node arrives to the border.
            If 'reflect', the node reflects off the border.
            If 'wrap', the node reappears at the opposite edge (as in a torus-shaped area).
        """
        self.rng = rng
        self.collect_fl_stats = False
        self.collect_wt_stats = False
        self.border_policy = border_policy
        self.dimensions = dimensions
        self.nr_nodes = nr_nodes
        self.FL_DISTR = FL_DISTR
        self.VELOCITY_DISTR = VELOCITY_DISTR
        self.WT_DISTR = WT_DISTR

    def __iter__(self):
        def reflect(xy):
            # node bounces on the margins
            for dim, max_d in enumerate(self.dimensions):
                b = np.where(xy[:, dim] < 0)[0]
                if b.size > 0:
                    xy[b, dim] = -xy[b, dim]
                    movement[b, dim] = -movement[b, dim]
                b = np.where(xy[:, dim] > max_d)[0]
                if b.size > 0:
                    xy[b, dim] = 2 * max_d - xy[b, dim]
                    movement[b, dim] = -movement[b, dim]

        def wrap(xy):
            for dim, max_d in enumerate(self.dimensions):
                b = np.where(xy[:, dim] < 0)[0]
                if b.size > 0:
                    xy[b, dim] += max_d
                b = np.where(xy[:, dim] > max_d)[0]
                if b.size > 0:
                    xy[b, dim] -= max_d

        if self.border_policy == "reflect":
            borderp = reflect
        elif self.border_policy == "wrap":
            borderp = wrap
        else:
            borderp = self.border_policy

        ndim = len(self.dimensions)
        NODES = np.arange(self.nr_nodes)

        # assign node's positions, flight lengths and velocities
        xy = U(
            self.rng,
            np.zeros(ndim),
            np.array(self.dimensions),
            np.dstack((NODES,) * ndim)[0],
        )
        fl = self.FL_DISTR(NODES)
        velocity = self.VELOCITY_DISTR(fl)

        # assign nodes' movements (direction * node velocity)
        direction = U(self.rng, 0.0, 1.0, np.zeros((self.nr_nodes, ndim))) - 0.5
        direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
        movement = direction * velocity[:, np.newaxis]

        # starts with no wating time
        wt = np.zeros(self.nr_nodes)

        if self.collect_fl_stats:
            self.fl_stats = list(fl)
        if self.collect_wt_stats:
            self.wt_stats = list(wt)

        while True:
            xy += movement
            fl -= velocity

            # step back for nodes that surpassed fl
            arrived = np.where(np.logical_and(velocity > 0.0, fl <= 0.0))[0]
            if arrived.size > 0:
                diff = fl.take(arrived) / velocity.take(arrived)
                xy[arrived] += np.dstack((diff,) * ndim)[0] * movement[arrived]

            # apply border policy
            borderp(xy)

            if self.WT_DISTR:
                velocity[arrived] = 0.0
                wt[arrived] = self.WT_DISTR(arrived)
                if self.collect_wt_stats:
                    self.wt_stats.extend(wt[arrived])
                # update info for paused nodes
                wt[np.where(velocity == 0.0)[0]] -= 1.0
                arrived = np.where(np.logical_and(velocity == 0.0, wt < 0.0))[0]

            # update info for moving nodes
            if arrived.size > 0:
                fl[arrived] = self.FL_DISTR(arrived)
                if self.collect_fl_stats:
                    self.fl_stats.extend(fl[arrived])
                velocity[arrived] = self.VELOCITY_DISTR(fl[arrived])
                v = velocity[arrived]
                direction = U(self.rng, 0.0, 1.0, np.zeros((arrived.size, ndim))) - 0.5
                direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
                movement[arrived] = v[:, np.newaxis] * direction

            yield xy


class RandomWalk(StochasticWalk):
    def __init__(
        self,
        rng,
        nr_nodes,
        dimensions,
        velocity=1.0,
        distance=1.0,
        border_policy="reflect",
    ):
        """
        Random Walk mobility model.
        This model is based in the Stochastic Walk, but both the flight length
        and node velocity distributions are in fact constants, set to the
        *distance* and *velocity* parameters. The waiting time is set to None.

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

        keyword arguments:

          *velocity*:
            Double, the value for the constant node velocity. Default is 1.0

          *distance*:
            Double, the value for the constant distance traveled in each step. Default is 1.0

          *border_policy*:
            String, either 'reflect' or 'wrap'. The policy that is used when the node arrives to the border.
            If 'reflect', the node reflects off the border.
            If 'wrap', the node reappears at the opposite edge (as in a torus-shaped area).
        """
        if velocity > distance:
            # In this implementation, each step is 1 second,
            # it is not possible to have a velocity larger than the distance
            raise Exception("Velocity must be <= Distance")

        fl = np.zeros(nr_nodes) + distance
        vel = np.zeros(nr_nodes) + velocity

        def FL_DISTR(SAMPLES):
            return np.array(fl[: len(SAMPLES)])

        def VELOCITY_DISTR(FD):
            return np.array(vel[: len(FD)])

        StochasticWalk.__init__(
            self,
            rng,
            nr_nodes,
            dimensions,
            FL_DISTR,
            VELOCITY_DISTR,
            border_policy=border_policy,
        )


class RandomDirection(StochasticWalk):
    def __init__(
        self,
        rng,
        nr_nodes,
        dimensions,
        wt_max=None,
        velocity=(0.1, 1.0),
        border_policy="reflect",
    ):
        """
        Random Direction mobility model.
        This model is based in the Stochastic Walk. The flight length is chosen from a uniform distribution,
        with minimum 0 and maximum set to the maximum dimension value.
        The velocity is also chosen from a uniform distribution, with boundaries set by the *velocity* parameter.
        If wt_max is set, the waiting time is chosen from a uniform distribution with values between 0 and wt_max.
        If wt_max is not set, waiting time is set to None.

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

        keyword arguments:

          *wt_max*:
            Double, maximum value for the waiting time distribution.
            If wt_max is set, the waiting time is chosen from a uniform distribution with values between 0 and wt_max.
            If wt_max is not set, the waiting time is set to None.
            Default is None.

          *velocity*:
            Tuple of Doubles, the minimum and maximum values for node velocity.

          *border_policy*:
            String, either 'reflect' or 'wrap'. The policy that is used when the node arrives to the border.
            If 'reflect', the node reflects off the border.
            If 'wrap', the node reappears at the opposite edge (as in a torus-shaped area).
        """

        MIN_V, MAX_V = velocity
        FL_MAX = max(dimensions)

        def FL_DISTR(SAMPLES):
            return U(rng, 0, FL_MAX, SAMPLES)

        def WT_DISTR(SAMPLES):
            if wt_max:
                return U(rng, 0, wt_max, SAMPLES)
            else:
                return None

        def VELOCITY_DISTR(FD):
            return U(rng, MIN_V, MAX_V, FD)

        StochasticWalk.__init__(
            self,
            rng,
            nr_nodes,
            dimensions,
            FL_DISTR,
            VELOCITY_DISTR,
            WT_DISTR=WT_DISTR,
            border_policy=border_policy,
        )


class TruncatedLevyWalk(StochasticWalk):
    def __init__(
        self,
        rng,
        nr_nodes,
        dimensions,
        FL_EXP=-2.6,
        FL_MAX=50.0,
        WT_EXP=-1.8,
        WT_MAX=100.0,
        border_policy="reflect",
    ):
        """
        Truncated Levy Walk mobility model, based on the following paper:
        Injong Rhee, Minsu Shin, Seongik Hong, Kyunghan Lee, and Song Chong.
        On the Levy-Walk Nature of Human Mobility.
        In 2008 IEEE INFOCOM - Proceedings of the 27th Conference on Computer
        Communications, pages 924-932. April 2008.

        The implementation is a special case of the more generic Stochastic Walk,
        in which both the flight length and waiting time distributions are truncated power laws,
        with exponents set to FL_EXP and WT_EXP and truncated at FL_MAX and WT_MAX.
        The node velocity is a function of the flight length.

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

        keyword arguments:

          *FL_EXP*:
            Double, the exponent of the flight length distribution. Default is -2.6

          *FL_MAX*:
            Double, the maximum value of the flight length distribution. Default is 50

          *WT_EXP*:
            Double, the exponent of the waiting time distribution. Default is -1.8

          *WT_MAX*:
            Double, the maximum value of the waiting time distribution. Default is 100

          *border_policy*:
            String, either 'reflect' or 'wrap'. The policy that is used when the node arrives to the border.
            If 'reflect', the node reflects off the border.
            If 'wrap', the node reappears at the opposite edge (as in a torus-shaped area).
        """

        def FL_DISTR(SAMPLES):
            return P(rng, FL_EXP, 1.0, FL_MAX, SAMPLES)

        WT_DISTR = None
        if WT_EXP and WT_MAX:

            def WT_DISTR(SAMPLES):
                return P(rng, WT_EXP, 1.0, WT_MAX, SAMPLES)

        def VELOCITY_DISTR(FD):
            return np.sqrt(FD) / 10.0

        StochasticWalk.__init__(
            self,
            rng,
            nr_nodes,
            dimensions,
            FL_DISTR,
            VELOCITY_DISTR,
            WT_DISTR=WT_DISTR,
            border_policy=border_policy,
        )


class HeterogeneousTruncatedLevyWalk(StochasticWalk):
    def __init__(
        self,
        rng,
        nr_nodes,
        dimensions,
        WT_EXP=-1.8,
        WT_MAX=100.0,
        FL_EXP=-2.6,
        FL_MAX=50.0,
        border_policy="reflect",
    ):
        """
        This is a variant of the Truncated Levy Walk mobility model.
        This model is based in the Stochastic Walk.
        The waiting time distribution is a truncated power law with exponent set to WT_EXP and truncated WT_MAX.
        The flight length is a uniform distribution, different for each node. These uniform distributions are
        created by taking both min and max values from a power law with exponent set to FL_EXP and truncated FL_MAX.
        The node velocity is a function of the flight length.

        Required arguments:

          *rng*:
            numpy.random.Generator object

          *nr_nodes*:
            Integer, the number of nodes.

          *dimensions*:
            Tuple of Integers, the x and y dimensions of the simulation area.

        keyword arguments:

          *WT_EXP*:
            Double, the exponent of the waiting time distribution. Default is -1.8

          *WT_MAX*:
            Double, the maximum value of the waiting time distribution. Default is 100

          *FL_EXP*:
            Double, the exponent of the flight length distribution. Default is -2.6

          *FL_MAX*:
            Double, the maximum value of the flight length distribution. Default is 50

          *border_policy*:
            String, either 'reflect' or 'wrap'. The policy that is used when the node arrives to the border.
            If 'reflect', the node reflects off the border.
            If 'wrap', the node reappears at the opposite edge (as in a torus-shaped area).
        """

        NODES = np.arange(nr_nodes)
        FL_MAX = P(rng, -1.8, 10.0, FL_MAX, NODES)
        FL_MIN = FL_MAX / 10.0

        def FL_DISTR(SAMPLES):
            return rng.random(len(SAMPLES)) * (FL_MAX[SAMPLES] - FL_MIN[SAMPLES]) + FL_MIN[SAMPLES]

        def WT_DISTR(SAMPLES):
            return P(rng, WT_EXP, 1.0, WT_MAX, SAMPLES)

        def VELOCITY_DISTR(FD):
            return np.sqrt(FD) / 10.0

        StochasticWalk.__init__(
            self,
            rng,
            nr_nodes,
            dimensions,
            FL_DISTR,
            VELOCITY_DISTR,
            WT_DISTR=WT_DISTR,
            border_policy=border_policy,
        )


def gauss_markov(
    rng: np.random.Generator,
    num_users: int,
    dimensions: Tuple[int, int],
    velocity_mean: np.ndarray,
    alpha: float = 1.0,
    variance: float = 1.0,
    anchor_loc: np.ndarray = None,
    cov_around_anchor: np.ndarray = None,
):
    """
    Gauss-Markov Mobility Model, as proposed in
    Camp, T., Boleng, J. & Davies, V. A survey of mobility models
    for ad hoc network research.
    Wireless Communications and Mobile Computing 2, 483-502 (2002).

    Required arguments:

      *num_users*:
        Integer, the number of users.

      *dimensions*:
        Tuple of Integers, the x and y dimensions of the simulation area.

      *velocity_mean*:
        The mean velocity

    keyword arguments:

      *alpha*:
        The tuning parameter used to vary the randomness

      *variance*:
        The randomness variance

      *anchor_loc*:
        Anchoring or concentration points, if seeking to initialize users in clusters.
        Provided as Nx2 array, where N must be divisible by `num_users`.

      *cov_around_anchor*:
        Covariance for sampling around anchor points, used only if `anchor_loc` is given.

        **Grid dimensions and tick time**

        The main concept for all the mobility models in that module is that the region of
        interest is specified as a uniform grid world (where x and y dimensions are
        provided as a tuple in the dimensions parameter in the constructors). The interpretation
        of the grid dimension (in real world units like meters) is up to the user.

        Likewise, each step of the trajectory, or a simulation tick, is interpreted in
        real world terms (1 s, 500 ms, 100 ms, etc.), by the user. _Once the user selects
        a tick unit and a grid unit, the velocity of the users must be parametrized in terms
        of *grid_distance per tick*_.

        *Example 1*. Say that the user wants a sample every 500 ms. Say that the desired
        velocity is 15 - 30 meters per second, or 7.5 to 15 meters per tick (500 ms). Then,
        the user may select any grid unit. For example, if they select grid unit = 1 meter,
        the final velocity parameter is *7.5 to 15 grid units per tick*. If, instead, the
        grid unit is selected to be 100 meters, then, the final velocity parameter is
        *0.075 to 0.15 grid units per tick*. The client code is able to interpret the resultant
        trajectory points, since the client code controls the interpretation.

    """

    MAX_X, MAX_Y = dimensions
    USERS = np.arange(num_users)

    if anchor_loc is None:
        x = U(rng, 0, MAX_X, USERS)
        y = U(rng, 0, MAX_Y, USERS)

    else:
        old_num_users = num_users
        num_users = int(num_users / len(anchor_loc)) * len(anchor_loc)
        if old_num_users != num_users:
            logging.info("len(anchor_loc) must evenly divide num_users.....terminating....")
            return

        num_users_per_anchor = int(num_users / len(anchor_loc))
        if cov_around_anchor is None:
            cov_around_anchor = np.array([[1, 0], [0, 1]])
        x, y = non_homogeneous_drop(
            rng=rng,
            anchor_loc=anchor_loc,
            num_users_per_anchor=num_users_per_anchor,
            cov_around_anchor=cov_around_anchor,
        ).transpose()

    velocity = np.zeros(num_users) + velocity_mean
    theta = U(rng, 0, 2 * np.pi, USERS)
    angle_mean = theta

    alpha2 = 1.0 - alpha
    alpha3 = np.sqrt(1.0 - alpha * alpha) * variance

    while True:
        x = x + velocity * np.cos(theta)
        y = y + velocity * np.sin(theta)

        # node bounces on the margins
        b = np.where(x < 0)[0]
        x[b] = -x[b]
        theta[b] = np.pi - theta[b]
        angle_mean[b] = np.pi - angle_mean[b]
        b = np.where(x > MAX_X)[0]
        x[b] = 2 * MAX_X - x[b]
        theta[b] = np.pi - theta[b]
        angle_mean[b] = np.pi - angle_mean[b]
        b = np.where(y < 0)[0]
        y[b] = -y[b]
        theta[b] = -theta[b]
        angle_mean[b] = -angle_mean[b]
        b = np.where(y > MAX_Y)[0]
        y[b] = 2 * MAX_Y - y[b]
        theta[b] = -theta[b]
        angle_mean[b] = -angle_mean[b]

        # calculate new speed and direction based on the model
        velocity = alpha * velocity + alpha2 * velocity_mean + alpha3 * rng.normal(0.0, 1.0, num_users)

        theta = alpha * theta + alpha2 * angle_mean + alpha3 * rng.normal(0.0, 1.0, num_users)

        yield np.dstack((x, y))[0]


def non_homogeneous_drop(
    rng: np.random.Generator,
    anchor_loc: np.ndarray,
    num_users_per_anchor: int,
    cov_around_anchor: np.ndarray,
):
    """
    rng: numpy.random.Generator object
    anchor_loc: Nx2 numpy array of anchor points locations.
      Users are (Gaussian) clustered around these anchor points.
    num_users_per_anchor: cluster size around these anchor points.
    cov_around_anchor: 2D (2x2) covariance for the Gaussian cluster.
    """
    num_anchors = anchor_loc.shape[0]
    user_loc = []
    for anchor_it in range(num_anchors):
        anchor_mean = anchor_loc[anchor_it, :]
        user_loc.append(rng.multivariate_normal(mean=anchor_mean, cov=cov_around_anchor, size=num_users_per_anchor))

    user_loc = np.concatenate(user_loc, axis=0)

    return user_loc
