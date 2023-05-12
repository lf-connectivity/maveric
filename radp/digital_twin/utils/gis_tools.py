# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# pyre-strict
# Contains generic GIS utility methods

from __future__ import annotations

import math
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd

from radp.digital_twin.utils.constants import CIRC_KM_TO_DEG_LAT


class GISTools:
    """GIS utilities"""

    # radius of the earth in kms
    R: float = 6378.1
    OneDegree: float = R * 2 * math.pi / 360 * 1000  # 1° latitude in meters

    # Bing tile Zoom Level --> Ground Resolution in Meters Per Pixel
    # Note : 1 Bing Tile is 256 x 256 pixels
    bing_tile_zoom_to_ground_resolution_meters_dict = {
        1: 78271.5170,
        2: 39135.7585,
        3: 19567.8792,
        4: 9783.9396,
        5: 4891.9698,
        6: 2445.9849,
        7: 1222.9925,
        8: 611.4962,
        9: 305.7481,
        10: 152.8741,
        11: 76.4370,
        12: 38.2185,
        13: 19.1093,
        14: 9.5546,
        15: 4.7773,
        16: 2.3887,
        17: 1.1943,
        18: 0.5972,
        19: 0.2986,
        20: 0.1493,
    }

    @staticmethod
    def get_tile_side_length_meters(bing_tile_zoom: int) -> float:
        """Returns equatorial ground length (in meters) for the specified
        Bing Tile zoom level."""
        assert bing_tile_zoom <= 20, "Only Bing Tile Zoom Level 20 and coarser are supported!"
        return GISTools.bing_tile_zoom_to_ground_resolution_meters_dict[bing_tile_zoom]

    @staticmethod
    def get_tile_side_length_km(lat: float, zoom: int) -> float:
        """
        Given a latitude and zoom level, return the side length of a Bing tile
        at that zoom level.
        """
        return float(math.cos(lat * math.pi / 180) * 2 * math.pi * GISTools.R / (2**zoom))

    @staticmethod
    def isclose(A: Tuple[float, float], B: Tuple[float, float], abs_tol: float = 0.0002) -> bool:
        try:
            return math.isclose(A[0], B[0], abs_tol=abs_tol) and math.isclose(A[1], B[1], abs_tol=abs_tol)
        except AttributeError:
            return abs(A[0] - B[0]) <= max(1e-9 * max(abs(A[0]), abs(B[0])), abs_tol) and abs(A[1] - B[1]) <= max(
                1e-9 * max(abs(A[1]), abs(B[1])), abs_tol
            )

    @staticmethod
    def dist(l1: Tuple[float, float], l2: Tuple[float, float], abs_tol: float = 0.0002) -> float:
        """Returns distance (in kms) between two points on the earth.

        Utlizes haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)

        Args:
            l1 and l2 are the coordinates of the two points in (lat,lng) format
            abs_tol is specification on tolerance for GISTools.isclose

        TODO(paulvarkey) : compare with different formula used here :
                           http://www.movable-type.co.uk/scripts/latlong.html
        """
        if GISTools.isclose(l1, l2, abs_tol=abs_tol):
            return 0.0

        [phi1, lam1] = [math.radians(l1[0]), math.radians(l1[1])]
        [phi2, lam2] = [math.radians(l2[0]), math.radians(l2[1])]
        d = (
            2
            * GISTools.R
            * math.asin(
                math.sqrt(
                    math.sin((phi2 - phi1) / 2) ** 2
                    + math.cos(phi1) * math.cos(phi2) * math.sin((lam2 - lam1) / 2) ** 2
                )
            )
        )
        return d

    @staticmethod
    def get_bearing(l1: Tuple[float, float], l2: Tuple[float, float]) -> float:
        """Find bearing of point `l2` from `l1`, both specified in (lat, lon).

        This returns bearing in the range of (-180, 180], where 0 is N and 180 is S.

        To change the result into the range [0, 360), where 0 is N and 180 is S,
        do the following :
            b = get_bearing()
            b = (b + 360) % 360
        """

        [phi1, lam1] = [math.radians(l1[0]), math.radians(l1[1])]
        [phi2, lam2] = [math.radians(l2[0]), math.radians(l2[1])]
        y = math.sin(lam2 - lam1) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lam2 - lam1)
        return math.degrees(math.atan2(y, x))

    @staticmethod
    def get_destination(
        origin: Union[List[int], List[float], Tuple[Union[int, float], Union[int, float]]],
        brng: float,
        d: float,
    ) -> Tuple[float, float]:
        """Given an origin, bearing and distance, return destination.

        Args:
            origin : (lat, lon)
            brng : bearing, in degrees
            d : distance, in kms
        """
        R = GISTools.R
        brng_r = math.radians(brng)
        [phi1, lam1] = [math.radians(origin[0]), math.radians(origin[1])]
        phi2 = math.asin(math.sin(phi1) * math.cos(d / R) + math.cos(phi1) * math.sin(d / R) * math.cos(brng_r))
        lam2 = lam1 + math.atan2(
            math.sin(brng_r) * math.sin(d / R) * math.cos(phi1),
            math.cos(d / R) - (math.sin(phi1) * math.sin(phi2)),
        )
        return (math.degrees(phi2), math.degrees(lam2))

    @staticmethod
    def convert_bearing_0_to_360(bearing_minus180_to_180: float) -> float:
        """Given bearing in degrees, in range (-180, 180], convert to [0, 360)."""
        return (bearing_minus180_to_180 + 360) % 360

    @staticmethod
    def rel_bearing(
        heading_0_to_360: float,
        target_bearing_0_to_360: float,
    ) -> float:
        """Given heading and target bearing in degrees, in range [0, 360),
        return the relative bearing of target w.r.t. heading, also,
        in degrees in range [0, 360)."""
        return (target_bearing_0_to_360 - heading_0_to_360) % 360

    @staticmethod
    def _random_point_in_disk(
        radius: float,
        rng: Any,  # pyre-ignore[2]
    ) -> Tuple[float, float]:
        """Gets a random point in disc centered around (0, 0) with radius `radius`."""
        assert type(rng) == np.random.Generator, "`rng` must be numpy.random.Generator"
        r = radius * math.sqrt(rng.uniform())
        theta = rng.uniform() * 2 * math.pi
        return (r * math.cos(theta), r * math.sin(theta))

    @staticmethod
    def random_location(
        lon: float,
        lat: float,
        radius: float,
        rng: Any,  # pyre-ignore[2]
    ) -> Tuple[float, float]:
        """Gets a random point in disc centered around (lat, lon) with radius `radius`.

        @params `lat`, `lon` : latitude and longitude in decimal degrees
        @param `radius` : radius (in meters)

        NOTE : At or near edges (latitudes 90 or -90 and longitudes 180 and -180),
               there will be wrong warp-around behavior
        """
        assert type(rng) == np.random.Generator, "`rng` must be numpy.random.Generator"
        assert -90 < lat < 90, f"Latitude {lat} should be in (-90, 90)"
        assert -180 < lon < 180, f"Longitude {lon} should be in (-180, 180)"
        dx, dy = GISTools._random_point_in_disk(radius=radius, rng=rng)
        random_lat = lat + dy / GISTools.OneDegree
        random_lat = max(-90, min(90, random_lat))
        random_lon = lon + dx / (GISTools.OneDegree * math.cos(lat * math.pi / 180))
        random_lon = max(-180, min(180, random_lon))
        return (random_lon, random_lat)

    @staticmethod
    def snap_align_lower_left(pt: Tuple[float, float], tile_discretization_resolution: int) -> Tuple[float, float]:
        lat = int(pt[0])
        lon = int(pt[1])
        inc = float(1 / float(tile_discretization_resolution - 1))
        frac_lat = pt[0] - lat
        frac_lon = pt[1] - lon
        frac_lat_inc = math.floor(frac_lat / inc)
        frac_lon_inc = math.floor(frac_lon / inc)
        # TODO(paulvarkey) : if almost idempotent, make it so!
        return (lat + (frac_lat_inc * inc), lon + (frac_lon_inc * inc))

    @staticmethod
    def get_grid_idx(
        pt: Tuple[float, float],
        SW: Tuple[float, float],
        tile_discretization_resolution: int,
        coarse_factor: int = 1,
    ) -> Tuple[int, int]:
        """Returns the index of a (lat, lon) point on a grid.

        The grid is supposed to be a 2D matrix (e.g. list of lists) in lat-lon
        order. The first row corresponds to the southern most latitude and the
        first column corresponds to the western most longitude. The grid is
        anchored at the South-West reference point SW, such that::
          get_grid_idx(SW, SW, tile_discretization_resolution) = [0, 0]
        >>> GISTools.get_grid_idx((-30.11, 24.67), (-30.11, 24.67), 3601)
        [0, 0]

        Args:
            pt : the coordinates of the point in (lat, lon) format
            SW : the South-West point -- (0,0) reference point for the grid
            tile_discretization_resolution : the number of grid divisions for
                                             1 latitude X 1 longitude
        """
        inc = 1.0 / float(tile_discretization_resolution - 1)

        # esnure that all pts are snap-aligned
        # pt = GISTools.snap_align_lower_left(pt, tile_discretization_resolution)
        # SW = GISTools.snap_align_lower_left(SW, tile_discretization_resolution)

        inc *= coarse_factor

        lat_idx = int(round((pt[0] - SW[0]) / inc))
        lon_idx = int(round((pt[1] - SW[1]) / inc))

        return (lat_idx, lon_idx)

    @staticmethod
    def get_latlon(
        grid_idx: Tuple[int, int],
        SW: Tuple[float, float],
        tile_discretization_resolution: int,
        coarse_factor: int = 1,
    ) -> Tuple[float, float]:
        """Returns (lat, lon) given the index of a point on a grid.

        The grid is supposed to be a 2D matrix (e.g. list of lists) in lat-lon
        order. The first row corresponds to the southern most latitude and the
        first column corresponds to the western most longitude. The grid is
        anchored at the South-West reference point SW, such that::
          get_latlon([0, 0], SW, tile_discretization_resolution) = SW
        >>> GISTools.get_latlon([0, 0], (-30.11, 24.67), 3601)
        (-30.11, 24.67)

        Args:
            grid_idx : the coordinates of the point on the grid
            SW : the South-West point -- (0,0) reference point for the grid
            tile_discretization_resolution : the number of grid divisions for
                                             1 latitude X 1 longitude
        """
        inc = 1.0 / float(tile_discretization_resolution - 1)

        lat = SW[0] + inc * grid_idx[0] * coarse_factor
        lon = SW[1] + inc * grid_idx[1] * coarse_factor

        return (lat, lon)

    @staticmethod
    def mk_grid_params(
        SW: Tuple[float, float],
        NE: Tuple[float, float],
        tile_discretization_resolution: int,
        coarse_factor: int = 1,
    ) -> Tuple[float, Tuple[float, float], Tuple[float, float], int, int]:
        """Create aligned and adjusted grid params"""
        inc = 1.0 / float(tile_discretization_resolution - 1)
        SW = GISTools.snap_align_lower_left(SW, tile_discretization_resolution)
        NE = GISTools.snap_align_lower_left((NE[0] + inc, NE[1] + inc), tile_discretization_resolution)
        inc *= coarse_factor
        num_rows = int(math.ceil((NE[0] - SW[0]) / inc)) + 1
        num_cols = int(math.ceil((NE[1] - SW[1]) / inc)) + 1
        return (inc, SW, NE, num_rows, num_cols)

    @staticmethod
    def get_bounding_box(
        lat: float,
        lon: float,
        radius: float,
        SW: Tuple[float, float],
        num_rows: int,
        num_cols: int,
        tile_discretization_resolution: int,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        aligned = GISTools.snap_align_lower_left((lat, lon), tile_discretization_resolution)
        idx = GISTools.get_grid_idx(aligned, SW, tile_discretization_resolution)
        box_radius = math.floor(radius / 30.0)
        if math.isclose(box_radius, 0):
            return (idx, idx)
        boxLL = (max(0, idx[0] - box_radius), max(0, idx[1] - box_radius))
        boxUR = (
            min(num_rows - 1, idx[0] + box_radius),
            min(num_cols - 1, idx[1] + box_radius),
        )
        return (boxLL, boxUR)

    @staticmethod
    def str_bbox(SW: Tuple[float, float], NE: Tuple[float, float]) -> str:
        return "From_" + str(SW).replace(" ", "") + "_To_" + str(NE).replace(" ", "")

    @staticmethod
    def latlon_to_wkt(lat: float, lon: float, precision: int = 12) -> str:
        """
        Defintion of a a unique keyable lnglat WKT string for a (lat, lon) tuple.
        Trailing zeros are stripped using fixed, given precision in order to,
        for example, match relational (SQL-like) datastores using VALUES.
        """
        lat = round(lat, precision)
        lon = round(lon, precision)
        precision_str_format: str = f"%.{precision}f"

        def coord_str(coord: float) -> str:
            if abs(coord) < 1e-04:  # special case to match ST_Point
                str_coord = str(coord)
                str_coord_parts = str_coord.split("e")
                mantissa = str_coord_parts[0]
                mantissa_decimal = ""
                if len(mantissa.split(".")) > 1:
                    mantissa_decimal = mantissa.split(".")[1]
                trailing_zeros_to_add = "0" * (16 - len(mantissa_decimal))
                return (
                    mantissa.split(".")[0] + "." + mantissa_decimal + trailing_zeros_to_add + "e" + str_coord_parts[1]
                )
            else:  # normal case
                return (precision_str_format % coord).rstrip("0").rstrip(".")

        lon_str = coord_str(lon)
        lat_str = coord_str(lat)
        return "POINT (" + lon_str + " " + lat_str + ")"

    @staticmethod
    def get_bbox_km_around_point(lat: float, lon: float, d: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Given a latlon point and a distance d (in km), return the SW and NE corners
        of a box whose side lengths are 2d with lat, lon as the center.
        """
        deg = d * CIRC_KM_TO_DEG_LAT
        scale_fac = 1 / math.cos(math.pi * lat / 180.0)
        minlat = lat - deg
        maxlat = lat + deg
        minlon = lon - deg * scale_fac
        maxlon = lon + deg * scale_fac
        return (minlat, minlon), (maxlat, maxlon)

    @staticmethod
    def extend_bbox(
        SW: Tuple[float, float],
        NE: Tuple[float, float],
        n_dist: float,
        s_dist: float,
        w_dist: float,
        e_dist: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Given a bounding box, extend the bounds by the desired distances outwards.
        SW and NE are (lat, lon) tuples for the southwest and northeast coordinates
        of the bbox.
        n/s/w/e_dist need to all be in km.
        Returns the SW and NE cooordinates of the resulting bounding box.
        """
        minlat, minlon = SW
        maxlat, maxlon = NE
        midlat = (minlat + maxlat) / 2
        scale_fac = 1 / math.cos(math.pi * midlat / 180.0)
        n_deg = n_dist * CIRC_KM_TO_DEG_LAT
        s_deg = s_dist * CIRC_KM_TO_DEG_LAT
        w_deg = w_dist * CIRC_KM_TO_DEG_LAT * scale_fac
        e_deg = e_dist * CIRC_KM_TO_DEG_LAT * scale_fac
        return (minlat - s_deg, minlon - w_deg), (maxlat + n_deg, maxlon + e_deg)

    @staticmethod
    def lon_lat_to_bing_tile(longitude: float, latitude: float, level: int = 18) -> Tuple[int, int]:
        """Convert the given pair of longitude and latitude to Bing tile, at specified resolution.

        Technical Outline:- https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system

        Args:
            latitude: Latitude in Degrees.
            longitude: Longitude in Degrees.

        Returns:
                Tuple containing (x, y) coordinates for corresponding Bing tile at specified level.

        """

        min_latitude = -85.05112878
        max_latitude = 85.05112878
        min_longitude = -180
        max_longitude = 180

        latitude = min(max(latitude, min_latitude), max_latitude)
        longitude = min(max(longitude, min_longitude), max_longitude)

        x = (longitude + 180) / 360
        sin_latitude = math.sin(latitude * math.pi / 180)
        y = 0.5 - (math.log(((1 + sin_latitude) / (1 - sin_latitude))) / (4 * math.pi))

        map_size = 256 << level

        """pixel_x and pixel_y is calculated by taking maximum of x* map_size and 0 and then taking
        by taking minimum with map_size -1."""

        pixel_x = int(min(max(x * map_size + 0.5, 0), map_size - 1))
        pixel_y = int(min(max(y * map_size + 0.5, 0), map_size - 1))

        tile_x = int(pixel_x / 256)
        tile_y = int(pixel_y / 256)

        return tile_x, tile_y

    @staticmethod
    def make_tile(
        lon_lat_tuple: Tuple[float, float],
        level: int = 18,
    ) -> Tuple[int, int]:
        """Convert (lon, lat) tuple to Bing tile (tile_x, tile_y) tuple, at specified resolution level.

        Args:
            lon_lat_tuple: Tuple of longitude(loc_x) and latitude(loc_y)
            level: Specified Resolution Level, by default is 18.

        Returns:
                tuple of tile_x(loc_x="longitude") and tile_y(loc_y="latitude")
        """

        tile_x, tile_y = GISTools.lon_lat_to_bing_tile(lon_lat_tuple[0], lon_lat_tuple[1], level=level)
        return (tile_x, tile_y)

    @staticmethod
    def get_all_covering_tiles(
        coverage_dataframe: pd.DataFrame,
        loc_x_field: str = "loc_x",
        loc_y_field: str = "loc_y",
        level: int = 18,
    ) -> List[List[int]]:
        """
        Get the list of Bing tiles, at specified resolution, that span
        all pixels in the coverage dataframe.

        Technical Outline:- https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system

        Args:
            coverage_dataframe: coverage dataframe containing columns with
            names that are given by the values of `loc_x_field` and `loc_y_field


        Returns:
                List of tuples containing (x, y) coordinates for spanning Bing tile (at resolution level 18)

        """
        list_of_pairs_of_tile_x_and_tile_y_coordinates = []

        """
        In the lat-lon space, the y dimension increases from bottom to top but
        it is the opposite in the Bing tile Space; therefore, tile_y_max corresponds
        to the Bing tile that spans the min value of loc_y, and vice versa
        """
        tile_x_min, tile_y_max = GISTools.lon_lat_to_bing_tile(
            coverage_dataframe[loc_x_field].min(),
            coverage_dataframe[loc_y_field].min(),
            level=level,
        )
        tile_x_max, tile_y_min = GISTools.lon_lat_to_bing_tile(
            coverage_dataframe[loc_x_field].max(),
            coverage_dataframe[loc_y_field].max(),
            level=level,
        )
        for i in range(tile_x_min, tile_x_max + 1):
            for j in range(tile_y_min, tile_y_max + 1):
                list_of_pairs_of_tile_x_and_tile_y_coordinates.append([i, j])
        return list_of_pairs_of_tile_x_and_tile_y_coordinates

    @staticmethod
    def get_relative_bearing(cell_az_deg, cell_lat, cell_lon, lat, lon):
        """Get relative bearing of lat/lon point from cell,
        with respect to the azimuthal direction of the cell."""
        return GISTools.rel_bearing(
            cell_az_deg,
            GISTools.convert_bearing_0_to_360(
                GISTools.get_bearing(
                    (cell_lat, cell_lon),
                    (
                        lat,
                        lon,
                    ),
                )
            ),
        )

    @staticmethod
    def get_log_distance(lat1, lon1, lat2, lon2, epsilon=1.0):
        """Get log of Haversine distance (in meters) between two points on earth."""
        return np.log(
            epsilon
            + 1000.0
            * GISTools.dist(
                (
                    lat1,
                    lon1,
                ),
                (lat2, lon2),
            )
        )

    @staticmethod
    def get_antenna_gain(hTx, hRx, log_distance, tilt_deg, theta_3db=10):
        """
        Reference: https://fb.quip.com/b2YrACm2FiOq
        hTx: height of Tx antenna
        hRx: height of Rx antenna
        log_distance: natural log of distance betwen Tx and Rx
        tilt_deg: downtilt
        theta_3db=3dB bandwidth of Tx antenna
        """
        relative_tilt = np.degrees(np.arctan((hTx - hRx) / np.exp(log_distance))) - tilt_deg
        G_db = -12 * np.power(relative_tilt / theta_3db, 2)
        return G_db

    @staticmethod
    def converting_xy_points_into_lonlat_pairs(
        xy_points: List,
        x_dim: int,
        y_dim: int,
        min_longitude: float = -180,
        max_longitude: float = 180,
        min_latitude: float = -90,
        max_latitude: float = 90,
    ) -> List[Tuple[float, float]]:
        """Converting the list of arrays of xy points in a given dimension of "x_dim"*"y_dim" to a List containing
        longitude("loc_x") and latitude("loc_y") corresponding to those points in a given latitude and longitude space.
        """

        lon_lat_points = []
        for point in xy_points:
            lon = min_longitude + ((max_longitude - min_longitude) / x_dim) * point[0]
            lat = min_latitude + ((max_latitude - min_latitude) / y_dim) * point[1]

            lon_lat_points.append((lon, lat))

        return lon_lat_points
