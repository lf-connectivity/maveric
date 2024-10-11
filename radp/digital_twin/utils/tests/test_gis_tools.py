# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest

import numpy as np
import pandas as pd
from shapely import wkt

from radp.digital_twin.utils.gis_tools import GISTools


class GISToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wkt_path = os.path.join(os.path.dirname(__file__), "../data/hungary_wkt.txt")
        with open(wkt_path, "r") as wkt_file:
            # pyre-ignore[16]
            cls.hungary_poly = wkt.load(wkt_file)

    def test_latlon_to_wkt(self) -> None:
        # coordinates greater than or equal to 1e-5 (in absolute value)
        self.assertEqual(GISTools.latlon_to_wkt(0.0004, 32), "POINT (32 0.0004)")
        self.assertEqual(GISTools.latlon_to_wkt(-0.0004, 32.000), "POINT (32 -0.0004)")
        self.assertEqual(GISTools.latlon_to_wkt(3.456, 0.0001), "POINT (0.0001 3.456)")
        self.assertEqual(
            GISTools.latlon_to_wkt(3.456000, -0.0001), "POINT (-0.0001 3.456)"
        )
        # coordinates lesser than 1e-5
        self.assertEqual(
            GISTools.latlon_to_wkt(0.00004, 32), "POINT (32 4.0000000000000000e-05)"
        )
        self.assertEqual(
            GISTools.latlon_to_wkt(-0.00004, 32.000),
            "POINT (32 -4.0000000000000000e-05)",
        )
        self.assertEqual(
            GISTools.latlon_to_wkt(3.456, 0.00001),
            "POINT (1.0000000000000000e-05 3.456)",
        )
        self.assertEqual(
            GISTools.latlon_to_wkt(3.456000, -0.00001),
            "POINT (-1.0000000000000000e-05 3.456)",
        )
        # from a known previously failing case
        self.assertEqual(
            GISTools.latlon_to_wkt(51.619369, 4.6e-05),
            "POINT (4.6000000000000000e-05 51.619369)",
        )

    def test_get_tile_side_length_km(self) -> None:
        lat = 0
        zoom = 18
        equator_tile_length_18 = 0.5972 * 256 / 1000
        length = GISTools.get_tile_side_length_km(lat, zoom)
        self.assertAlmostEqual(length, equator_tile_length_18, places=4)

    def test_get_bbox_km_around_point(self) -> None:
        lat, lon = 41.819782, -71.830141
        radius = 7
        result = GISTools.get_bbox_km_around_point(lat, lon, radius)
        minlat, minlon = result[0]
        maxlat, maxlon = result[1]
        self.assertAlmostEqual(minlat, 41.75689993011163, places=3)
        self.assertAlmostEqual(minlon, -71.91451862086333, places=3)
        self.assertAlmostEqual(maxlat, 41.882664069888364, places=3)
        self.assertAlmostEqual(maxlon, -71.74576337913666, places=3)

    def test_extend_bbox(self) -> None:
        lat, lon = 41.819782, -71.830141
        radius = 7
        SW, NE = GISTools.get_bbox_km_around_point(lat, lon, radius)
        _, NE_ext = GISTools.extend_bbox(SW, NE, 1, 0, 0, 0)
        self.assertGreater(NE_ext[0], NE[0])
        _, NE_ext = GISTools.extend_bbox(SW, NE, 0, 0, 0, 1)
        self.assertGreater(NE_ext[1], NE[1])
        SW_ext, _ = GISTools.extend_bbox(SW, NE, 0, 1, 0, 0)
        self.assertLess(SW_ext[0], SW[0])
        SW_ext, _ = GISTools.extend_bbox(SW, NE, 0, 0, 1, 0)
        self.assertLess(SW_ext[1], SW[1])

    def test_random_location(self) -> None:
        rng = np.random.default_rng(42)

        lat, lon = 41.819782, -71.830141
        radius = 7

        rand_lat_1_seed1, rand_lon_1_seed1 = GISTools.random_location(
            lat=lat, lon=lon, radius=radius, rng=rng
        )
        rand_lat_2_seed1, rand_lon_2_seed1 = GISTools.random_location(
            lat=lat, lon=lon, radius=radius, rng=rng
        )

        # Seed with a different seed and observe that results are different
        rng = np.random.default_rng(43)
        rand_lat_1_seed2, rand_lon_1_seed2 = GISTools.random_location(
            lat=lat, lon=lon, radius=radius, rng=rng
        )
        self.assertNotEqual(rand_lat_1_seed1, rand_lat_1_seed2)
        self.assertNotEqual(rand_lon_1_seed1, rand_lon_1_seed2)

        # Re-seed and observe the same points
        rng = np.random.default_rng(42)
        rand_lat_1_seed1_again, rand_lon_1_seed1_again = GISTools.random_location(
            lat=lat, lon=lon, radius=radius, rng=rng
        )
        rand_lat_2_seed1_again, rand_lon_2_seed1_again = GISTools.random_location(
            lat=lat, lon=lon, radius=radius, rng=rng
        )
        self.assertEqual(rand_lat_1_seed1, rand_lat_1_seed1_again)
        self.assertEqual(rand_lon_1_seed1, rand_lon_1_seed1_again)
        self.assertEqual(rand_lat_2_seed1, rand_lat_2_seed1_again)
        self.assertEqual(rand_lon_2_seed1, rand_lon_2_seed1_again)

    def test_bearing_utils(self) -> None:
        # for get_bearing, the first parameter in the tuples corresponds to lat (y axis)

        self.assertEqual(
            GISTools.get_bearing((0, 0), (0, 0)),
            0,
        )
        self.assertEqual(
            GISTools.get_bearing((0, 0), (1, 0)),  # go North
            0,
        )
        self.assertEqual(
            GISTools.get_bearing((0, 0), (0, 1)),  # go East
            90,
        )
        self.assertAlmostEqual(
            GISTools.get_bearing((0, 0), (1, 1)),  # go North-East
            45,
            places=0,
        )
        self.assertEqual(
            GISTools.get_bearing((0, 0), (-1, 0)),  # go South
            180,
        )
        self.assertAlmostEqual(
            GISTools.get_bearing((0, 0), (-1, 1)),  # go South-East
            135,
            places=0,
        )
        self.assertEqual(
            GISTools.get_bearing((0, 0), (0, -1)),  # go West
            -90,
        )
        self.assertAlmostEqual(
            GISTools.get_bearing((0, 0), (-1, -1)),  # go South-West
            -135,
            places=0,
        )
        self.assertAlmostEqual(
            GISTools.get_bearing((0, 0), (1, -1)),  # go North-West
            -45,
            places=0,
        )

        # convert_bearing_0_to_360

        self.assertEqual(
            GISTools.convert_bearing_0_to_360(45),  # go West
            45,
        )
        self.assertEqual(
            GISTools.convert_bearing_0_to_360(135),  # go West
            135,
        )
        self.assertEqual(
            GISTools.convert_bearing_0_to_360(180),  # go West
            180,
        )
        self.assertEqual(
            GISTools.convert_bearing_0_to_360(-179),  # go West
            181,
        )
        self.assertEqual(
            GISTools.convert_bearing_0_to_360(-90),  # go West
            270,
        )

        # rel_bearing

        self.assertEqual(
            GISTools.rel_bearing(heading_0_to_360=0, target_bearing_0_to_360=1),
            1,
        )
        self.assertEqual(
            GISTools.rel_bearing(heading_0_to_360=0, target_bearing_0_to_360=359),
            359,
        )
        self.assertEqual(
            GISTools.rel_bearing(heading_0_to_360=1, target_bearing_0_to_360=0),
            359,
        )
        self.assertEqual(
            GISTools.rel_bearing(heading_0_to_360=359, target_bearing_0_to_360=0),
            1,
        )

    def test_lat_and_lon_to_tile_x_and_tile_y(self):
        tile_x, tile_y = GISTools.lon_lat_to_bing_tile(80, 80)
        self.assertEqual((tile_x == 189326 and tile_y == 29428), True)

    def test_make_tile(self):
        tile_x, tile_y = GISTools.make_tile((80, 80))
        self.assertEqual((tile_x == 189326 and tile_y == 29428), True)

    def test_get_all_covering_tiles(self):
        d = {"loc_x": [-0.00056, 0.00056], "loc_y": [0.00056, -0.00056]}
        df = pd.DataFrame(data=d)
        tiles = GISTools.get_all_covering_tiles(df)
        expected = [
            [131071, 131071],
            [131071, 131072],
            [131072, 131071],
            [131072, 131072],
        ]
        self.assertTrue(tiles == expected)

    def test_converting_xy_points_into_lonlat_pairs(self):
        s = [
            np.array([85.8649796, 9.34373949]),
            np.array([69.74819822, 97.51281031]),
            np.array([85.88763201, 9.27841924]),
            np.array([69.65711548, 97.69667864]),
        ]
        expected_lat_lon_points = GISTools.converting_xy_points_into_lonlat_pairs(
            xy_points=s,
            x_dim=100,
            y_dim=100,
            min_longitude=0,
            max_longitude=60,
            min_latitude=-15,
            max_latitude=50,
        )
        actual_lat_lon_points = [
            (51.518987759999995, -8.9265693315),
            (41.848918932000004, 48.3833267015),
            (51.532579206, -8.969027493999999),
            (41.794269288, 48.502841116000006),
        ]

        self.assertTrue(expected_lat_lon_points == actual_lat_lon_points)
