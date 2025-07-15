# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import zipfile
from typing import Any, Container, Dict, List, Optional, Tuple, Union

import fastkml
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio.features
from dotenv import load_dotenv
from rasterio.transform import Affine
from shapely import geometry

from radp.digital_twin.mobility.mobility import gauss_markov
from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
from radp.digital_twin.mobility.ue_tracks_params import UETracksGenerationParams
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin, NormMethod
from radp.digital_twin.utils.constants import RLF_THRESHOLD, TXPWR_DBM
from radp.digital_twin.utils.gis_tools import GISTools

Boundary = Union[geometry.Polygon, geometry.MultiPolygon]
KML_NS = "{http://www.opengis.net/kml/2.2}"

SRTM_STEP = 1.0 / 3600.0

load_dotenv()
# Ensure to install ffmpeg on your machine and replace the FFMPEG_PATH
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", None)


def _kml_obj_to_string(kml_obj: fastkml.KML) -> bytes:
    kml_str = kml_obj.to_string(prettyprint=False).replace("kml:", "")
    """
    Converts a FastKML object to its string representation.
    @return str: string representation of the KML object
    """
    # Encode to 'utf-8' to eliminate unicode code points
    return kml_str.encode()


def write_kml_as_kmz_file(kml_obj: fastkml.KML, kmz_filename: str) -> None:
    """
    Writes a FastKML object as a KMZ file.
    @param kml_obj: a FastKML object
    @param kmz_filename str: filename to write
    """
    # Strip out redundant "kml:" prefix
    kml_str = _kml_obj_to_string(kml_obj)

    with zipfile.ZipFile(kmz_filename, mode="w") as zfh:
        zfh.writestr("doc.kml", kml_str, compress_type=zipfile.ZIP_DEFLATED)


class ShapesKMLWriter(object):
    @classmethod
    def _doc_name(cls, kmz_name: str) -> str:
        """
        Extracts the document name from a given KMZ file path.
        @params kmz_name: the file path of the KMZ file.
        @returns: the name of the document without the file extension.
        """

        doc_name = os.path.basename(kmz_name)
        return os.path.splitext(doc_name)[0]

    @classmethod
    def _setup_kml_obj(
        cls,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        styles: Optional[List[fastkml.Style]] = None,
    ) -> Tuple[fastkml.KML, fastkml.Document]:
        """
        Set up a KML object and its root document.
        @param name: name of the document
        @param desc: description of the document
        @param styles: list of styles
        @returns: (kml_obj, root_document) pair.
        """
        if not styles:
            styles = []

        k = fastkml.KML()
        doc = fastkml.Document(ns=KML_NS, name=(name or "Shapes"), description=(desc or ""), styles=styles)
        doc = fastkml.Document(ns=KML_NS, name=(name or "Shapes"), description=(desc or ""), styles=styles)
        k.append(doc)

        return k, doc

    @staticmethod
    def _build_description_from_prop_dict(
        prop_dict: Dict[str, Any], props_to_exclude: Optional[Container[str]] = None
    ) -> str:
        """
        Take a dict and return a string of the formatted key-value pairs.
        Optionally specify keys to ignore (for example, NODE_ALTITUDE_ATTR, which is
        automatically added by the ANP graph library).
        """
        props_to_exclude = {} if props_to_exclude is None else props_to_exclude
        s = [f"{k}:  {v}" for k, v in prop_dict.items() if k not in props_to_exclude]
        return ",\n".join(s)

    @classmethod
    def _add_shape_to_folder(
        cls,
        folder: Union[fastkml.Folder, fastkml.Document],
        shape: geometry.base.BaseGeometry,
        name: str,
        styles: Optional[List[fastkml.Style]] = None,
        desc: Optional[str] = None,
    ) -> None:
        """
        Add a geometric shape to a KML folder or document.
        @param folder: The KML folder or document to which the shape will be added.
        @param shape: The geometric shape to add, represented as a BaseGeometry object.
        @param name: The name of the placemark representing the shape.
        @param styles: Optional list of styles to apply to the placemark.
        @param desc: Optional description of the placemark. Defaults to the value of `name` if not provided.
        @returns: None.
        """
        if desc is None:
            desc = name
        shape_placemark = fastkml.Placemark(ns=KML_NS, name=name, description=desc, styles=styles)
        shape_placemark.geometry = shape
        folder.append(shape_placemark)

    @classmethod
    def placemark_list_to_kmz(
        cls,
        placemarks: List[fastkml.Placemark],
        kmz_filename: str,
        folder_name: str = "Places",
        styles: Optional[List[fastkml.Style]] = None,
    ) -> None:
        """
        Added a list of placemarks to a KMZ file
        @param placemarks [kml.Placemarks]: list of placemarks to write
        @param kmz_filename str: name of file to write
        @param kml_filename str: name of the internal KML file
        @param folder_name str: name of the folder to make and put placemarks in
        @param styles [styles.Style]: list of style objects
        """

        doc_name = cls._doc_name(kmz_filename)
        kml_pair = cls._setup_kml_obj(name=doc_name, styles=styles)
        kml_obj, doc = kml_pair

        folder = fastkml.Folder(ns=KML_NS, name=folder_name)
        doc.append(folder)

        for place in placemarks:
            folder.append(place)

        # Write as zip
        write_kml_as_kmz_file(kml_obj, kmz_filename)

    @classmethod
    def shape_dict_to_kmz(
        cls,
        shape_dict: Dict[str, Boundary],
        filename: str,
        styles: Optional[List[fastkml.Style]] = None,
        styles_dict: Optional[Dict[str, List[fastkml.Style]]] = None,
        zipped: bool = True,
        descriptions_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Add shapes in a (nested) directory structure to a KMZ (or KML) file.
        @param shape_dict: (nested) dict of shapes. For example,
            {'A': SHAPE_1, 'F' : {'B' : SHAPE_2, 'C' : SHAPE_3}} describes three
            shapes and one subfolder
            DOC_ROOT -> A
                -> F -> B
                    -> C
        @param filename str: name of the kmz or kml file
        @param styles: list of fastkml styles to apply to all shapes
        @param styles_dict: dict of lists of styles applied to individual shapes
            (matched by keys of styles_dict and shape_dict)
        @param zipped bool: if true, output in kmz format. Otherwise, kml.
        @param descriptions_dict: dict of descriptions associated with each shape.
            keys of shape_dict will be used for name and description otherwise.
        """
        doc_name = cls._doc_name(filename)
        kml_pair = cls._setup_kml_obj(name=doc_name)
        kml_obj, doc = kml_pair

        fringe = [(doc, shape_dict)]

        while len(fringe) > 0:
            cur_folder, cur_dict = fringe.pop()
            for name, obj in cur_dict.items():
                if styles_dict is not None and name in styles_dict:
                    obj_style = styles_dict[name]
                else:
                    obj_style = styles

                if descriptions_dict is not None and name in descriptions_dict:
                    obj_desc = ShapesKMLWriter._build_description_from_prop_dict(descriptions_dict[name])
                else:
                    obj_desc = name

                if isinstance(obj, geometry.base.BaseGeometry):
                    cls._add_shape_to_folder(cur_folder, obj, name, styles=obj_style, desc=obj_desc)
                else:
                    # isinstance(obj, dict))
                    child_folder = fastkml.Folder(ns=KML_NS, name=name, styles=obj_style)
                    cur_folder.append(child_folder)
                    fringe.append((child_folder, obj))

        # Write as zip
        if zipped:
            write_kml_as_kmz_file(kml_obj, filename)
        else:
            with open(filename, "w+") as f:
                f.write(_kml_obj_to_string(kml_obj).decode())


def get_percell_data(
    data_in: pd.DataFrame,
    n_samples: int,
    choose_strongest_samples_percell: bool = False,
    invalid_value: float = -500.0,
    seed: int = 0,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Prediction dataframe cleanup
    Dataframe should contain ['cell_id', 'log_distance', 'relative_bearing', 'cell_rxpwr_dbm'] cloumns.
    
    
    +---------+--------------+------------------+----------------+
    | cell_id | log_distance | relative_bearing | cell_rxpwr_dbm |
    +=========+==============+==================+================+
    |   1     | 102.22       | 33.67            | -85.4          |
    |   2     | 102.42       | 33.85            | -90.1          |
    |   3     | 102.54       | 33.87            | -88.9          |
    |   1     | 102.29       | 33.57            | -75.3          |
    |   2     | 102.36       | 33.91            | -72.0          |
    |   3     | 102.08       | 33.83            | -78.7          |
    +---------+--------------+------------------+----------------+
    
    """
    data_out = []
    data_stats = []
    data_in_sampled = data_in

    data_in_sampled.columns = [col.replace("_1", "") if col.endswith("_1") else col for col in data_in_sampled.columns]

    # filter out invalid values
    data_cell_valid = data_in_sampled[data_in_sampled.cell_rxpwr_dbm != invalid_value]
    if choose_strongest_samples_percell:
        data_cell_sampled = data_cell_valid.sort_values("cell_rxpwr_dbm", ascending=False).head(
            n=min(n_samples, len(data_cell_valid))
        )

    else:
        # get n_samples independent random samples inside training groups
        data_cell_sampled = data_cell_valid.sample(n=min(n_samples, len(data_cell_valid)), random_state=(seed))

    data_out.append(data_cell_sampled.reset_index(drop=True))

    stats_cell = data_cell_sampled.describe(include="all")
    data_stats.append(stats_cell)

    return data_out, data_stats


def y_to_latitude(lower_bound: bool, y: float, zoom_factor: int, tile_pixels: int = 256) -> float:
    """
    Convert a y tile coordinate to a latitude coordinate.

    Arguments:
        lower_bound: A bool whether to use the lower edge of the tile.
        y: The y tile coordinate.
        zoom_factor: The zoom factor for the tile's zoom level.

    Returns:
        Degrees latitude for either the upper or lower edge of the tile.
    """
    if lower_bound:
        y = y + 1
    yt = (y * 1.0) / zoom_factor
    k = np.exp((yt - 0.5) * 4 * np.pi)
    sin_latitude = (k - 1) / (k + 1)
    latitude = np.arcsin(sin_latitude)
    latitude_degrees = latitude * 180.0 / np.pi
    return -latitude_degrees


def bing_tile_to_center(x: float, y: float, level: int, tile_pixels: int = 256) -> float:
    """Get the center coordinate as [latitude, longitude]
    for a given tile.

    Arguments:
        x: The tile x coordinate.
        y: The tile y coordinate.
        level: The zoom level of the tile.

    Returns:
        An array with [latitude, longitude] coordinates
        of the center of the given tile.
    """
    zoom_factor = 1 << level
    xwidth = 360.0 / zoom_factor
    out = []
    out.append(
        (y_to_latitude(True, y, zoom_factor, tile_pixels) + y_to_latitude(False, y, zoom_factor, tile_pixels)) / 2
    )
    out.append(xwidth * (x + 0.5) - 180)
    return out


def bing_tile_to_center_df_row(row: int, level: int) -> int:
    """
    Convert Bing tile coordinates in a DataFrame row to their center coordinates.
    @param row: A DataFrame row containing Bing tile coordinates with attributes `loc_x` and `loc_y`.
    @param level: The zoom level of the Bing tile.
    @returns: The modified DataFrame row with updated `loc_x` and `loc_y` values representing the center coordinates.
    """

    y, x = bing_tile_to_center(row.loc_x, row.loc_y, level)
    row.loc_x = x
    row.loc_y = y
    return row


def longitude_to_world_pixel(longitude: float, zoom_factor: int, tile_pixels: int = 256) -> float:
    """Convert degrees longitude to world pixel coordinates.

    World pixel coordinates span the whole range of longitude
    coordinates, whereas tile pixel coordinates mark a position
    within a tile. Longitude pixels have their origin coordinate at the
    westernmost valid longitude coordinate: -180 deg,
    and extend to their easternmost coordinate: 180 deg.

    Arguments:
        longitude: Longitude coordinate in degrees.
        zoom_factor: Zoom factor for the given zoom level.

    Returns:
        A world pixel coordinate in longitudinal direction, a floating-point
    """
    longitude = map_clip(longitude, -180.0, 180.0)
    pixel_x = ((longitude + 180.0) / 360.0) * (tile_pixels * zoom_factor)
    return pixel_x


def latitude_to_world_pixel(latitude: float, zoom_factor: int, tile_pixels: int = 256) -> float:
    """Convert degrees latitude to world pixel coordinates.

    World pixel coordinates span the whole range of latitude
    coordinates, whereas tile pixel coordinates mark a position
    within a tile. Latitude pixels have their origin coordinate at the
    northernmost valid latitude coordinate: 85.05112878 deg,
    and extent to their southernmost coordinate: -85.05112878 deg.

    Arguments:
        latitude: Latitude in degrees.
        zoom_factor: Zoom factor for the given zoom level.

    Returns:
        A world pixel coordinate in latitudinal direction, a floating-point
    """
    latitude = map_clip(latitude, -85.05112878, 85.05112878)
    sin_latitude = np.sin(latitude * np.pi / 180.0)

    pixel_y = (0.5 - np.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * np.pi)) * (tile_pixels * zoom_factor)
    return pixel_y


def map_clip(val: float, min_val: float, max_val: float) -> float:
    """Clip val to [min_val, max_val].

    Arguments:
        val: A value to be clipped.
        min_val: The minimum allowed value.
        max_val: The maximum allowed value.

    Returns:
        Value clipped to the range of [min_val, max_val].
    """
    return np.max([min_val, np.min([val, max_val])])


def lon_lat_to_bing_tile(longitude: float, latitude: float, level: int, tile_pixels: int = 256) -> float:
    """
    Convert longitude and latitude to Bing tile coordinates.
    @param longitude: Longitude of the location in degrees.
    @param latitude: Latitude of the location in degrees.
    @param level: Zoom level for the Bing tile.
    @param tile_pixels: Size of the tile in pixels (default is 256).
    @returns: (x, y) pair representing the Bing tile coordinates.
    """
    zoom_factor = 1 << level
    pixel_x = longitude_to_world_pixel(longitude, zoom_factor)
    x = int(map_clip(np.floor(pixel_x / tile_pixels), 0, tile_pixels * zoom_factor - 1))
    pixel_y = latitude_to_world_pixel(latitude, zoom_factor)
    y = int(map_clip(np.floor(pixel_y / tile_pixels), 0, zoom_factor - 1))
    return x, y


def lon_lat_to_bing_tile_df_row(row: int, level: int) -> int:
    """
    Convert longitude and latitude in a DataFrame row to Bing tile coordinates.
    @param row: A DataFrame row containing 'loc_x' (longitude) and 'loc_y' (latitude) attributes.
    @param level: The zoom level for the Bing tile conversion.
    @returns: The modified DataFrame row with 'loc_x' and 'loc_y' updated to Bing tile coordinates.
    """

    x, y = lon_lat_to_bing_tile(row.loc_x, row.loc_y, level)
    row.loc_x = x
    row.loc_y = y
    return row


def get_lonlat_from_xy_idxs(xy: np.ndarray, lower_left: Tuple[float, float]) -> np.ndarray:
    """
    Convert x and y indices to longitude and latitude coordinates.
    @param xy: A numpy array of x and y indices.
    @param lower_left: A tuple representing the longitude and latitude of the lower-left corner.
    @returns: A numpy array of longitude and latitude coordinates.
    """
    return xy * SRTM_STEP + lower_left


def find_closest(data_df: pd.DataFrame, lat: float, lon: float) -> Optional[int]:
    """
    Find the closest point in a DataFrame to a given latitude and longitude.

    @param data_df: A pandas DataFrame containing location data with columns 'loc_y' (latitude) and 'loc_x' (longitude).
    @param lat: Latitude of the target point.
    @param lon: Longitude of the target point.
    @returns: The index of the closest point if the minimum distance is less than 100, otherwise None.
   
    +------------+------------+
    |  loc_x     |  loc_y     |
    +============+============+
    |  90.412521 | 23.810331  |
    |  90.413000 | 23.811000  |
    |  90.410000 | 23.809000  |
    |  90.420000 | 23.820000  |
    |  90.405000 | 23.800000  |
    +------------+------------+
    
    """
    dist = data_df.apply(lambda row: GISTools.dist((row.loc_y, row.loc_x), (lat, lon)), axis=1)
    if dist.min() < 100:
        return dist.idxmin()
    else:
        return None


def get_track_samples(
    data_df: pd.DataFrame,
    num_UEs: int,
    ticks: int,
) -> pd.DataFrame:
    """
    Generate track samples based on a Gauss-Markov mobility model
    and map them to the closest points in the given dataset.

    @param data_df: Input DataFrame containing location data with columns 'loc_x' and 'loc_y'.
    @param num_UEs: Number of user equipment (UE) tracks to simulate.
    @param ticks: Number of time steps to simulate for the mobility model.
    @returns: A DataFrame containing the sampled track points mapped to the closest points in the input dataset.
    
    +------------+------------+
    |  loc_x     |  loc_y     |
    +============+============+
    |  90.412521 | 23.810331  |
    |  90.413000 | 23.811000  |
    |  90.410000 | 23.809000  |
    |  90.420000 | 23.820000  |
    |  90.405000 | 23.800000  |
    +------------+------------+

    """

    alpha = 0.8
    variance = 0.5

    # initialize random seed
    rng = np.random.default_rng(0)

    min_lon = min(data_df.loc_x)
    min_lat = min(data_df.loc_y)
    max_lon = max(data_df.loc_x)
    max_lat = max(data_df.loc_y)

    lon_x_dims = math.ceil((max_lon - min_lon) / SRTM_STEP)
    lat_y_dims = math.ceil((max_lat - min_lat) / SRTM_STEP)

    mobility_model = gauss_markov(
        rng=rng,
        num_users=num_UEs,
        dimensions=[lon_x_dims, lat_y_dims],
        velocity_mean=rng.uniform(low=0.09, high=0.1, size=num_UEs),
        alpha=alpha,
        variance=variance,
    )

    xy_lonlat_ue_tracks = []
    for _t, xy in enumerate(mobility_model):
        if _t >= ticks:
            break
        xy_lonlat = get_lonlat_from_xy_idxs(xy, (min_lon, min_lat))
        xy_lonlat_ue_tracks.extend(xy_lonlat)

    all_track_pts_df = pd.DataFrame(columns=["loc_x", "loc_y"], data=xy_lonlat_ue_tracks)
    all_track_pts_sampled_df = all_track_pts_df.apply(lambda row: find_closest(data_df, row.loc_y, row.loc_x), axis=1)

    return data_df.loc[all_track_pts_sampled_df]


def bdt(
    sim_idx_folders: List[str],
    bucket_path: str,
    sim_data_path: str,
    test_idx: int,
    p_train=20,
    p_test=100,
    maxiter=20,
    # load_model=False,
    # save_model=False,
    # model_path="",
    # model_name="",
    plot_loss_vs_iter=False,
    cmap="PuBuGn",
    choose_strongest_samples_percell=False,
    include_test_in_training=False,
    filter_out_samples_dbm_threshold=-np.inf,
    filter_out_samples_kms_threshold=np.inf,
    track_sampling=False,
    num_UEs=10,
    ticks=100,
):
    """
    Train and test Bayesian Digital Twins (BDT) for cellular network data.
    """
    site_config_path = sim_idx_folders[0] + "/site_config.csv"
    site_config_df = pd.read_csv(f"/{bucket_path}/{sim_data_path}/{site_config_path}")

    n_sample_train = (p_train * 0.01 * site_config_df["nRx"]).astype(int)
    n_sample_test = (p_test * 0.01 * site_config_df["nRx"]).astype(int)
    n_cell = len(site_config_df.index)

    metadata_df = pd.DataFrame(
        {
            "cell_id": [cell_id for cell_id in site_config_df.cell_id],
            "idx": [i + 1 for i in range(n_cell)],
        }
    )
    idx_cell_id_mapping = dict(zip(metadata_df.idx, metadata_df.cell_id))
    desired_idxs = [1 + r for r in range(n_cell)]

    # train

    percell_data_list = []

    for idx in range(len(sim_idx_folders)):
        # sim_idx = s + 1
        if idx == test_idx and not include_test_in_training:
            continue
        sim_idx_folder = sim_idx_folders[idx]
        logging.info(f"loading training set : {sim_idx_folder}")
        save_path = f"/{bucket_path}/{sim_data_path}/{sim_idx_folder}"
        tilt_df = pd.read_csv(f"{save_path}/full_data.csv")

        # drop redundant columns
        clean_tilt_df = tilt_df.drop(columns=["cell_rxpwr_dbm", "cell_el_deg"])

        # get data by groups
        tilt_per_cell_df = [x for _, x in clean_tilt_df.groupby("cell_id")]

        tilt_per_cell_df_processed = []

        for i in range(n_cell):
            tilt_per_cell_df_processed.append(
                get_percell_data(
                    data_in=tilt_per_cell_df[i],
                    choose_strongest_samples_percell=choose_strongest_samples_percell,
                    n_samples=n_sample_train[i],
                )[0][0]
            )

        percell_data_list.append(tilt_per_cell_df_processed)

    training_data = {}
    if plot_loss_vs_iter:
        _, axs = plt.subplots(1, 2, figsize=(16, 12))
        axs[0].set_title("Training points")
        axs[0].set_aspect("equal", "box")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_title("Training points after filtering distant weak points")
        axs[1].set_aspect("equal", "box")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
    for i in range(len(desired_idxs)):
        train_cell_id = idx_cell_id_mapping[i + 1]
        training_data[train_cell_id] = pd.concat([tilt_per_cell_df[i] for tilt_per_cell_df in percell_data_list])

        if track_sampling:
            training_data[train_cell_id] = get_track_samples(
                training_data[train_cell_id],
                num_UEs=num_UEs,
                ticks=ticks,
            )
        if plot_loss_vs_iter:
            axs[0].scatter(
                training_data[train_cell_id].loc_x,
                training_data[train_cell_id].loc_y,
                c=training_data[train_cell_id].cell_rxpwr_dbm,
                cmap=cmap,
                s=25,
            )
    for train_cell_id, training_data_idx in training_data.items():
        training_data_idx["cell_id"] = train_cell_id
        training_data_idx["cell_lat"] = site_config_df[site_config_df["cell_id"] == train_cell_id]["cell_lat"].values[0]
        training_data_idx["cell_lon"] = site_config_df[site_config_df["cell_id"] == train_cell_id]["cell_lon"].values[0]
        training_data_idx["cell_az_deg"] = site_config_df[site_config_df["cell_id"] == train_cell_id][
            "cell_az_deg"
        ].values[0]
        training_data_idx["cell_txpwr_dbm"] = site_config_df[site_config_df["cell_id"] == train_cell_id][
            "cell_txpwr_dbm"
        ].values[0]
        training_data_idx["hTx"] = site_config_df[site_config_df["cell_id"] == train_cell_id]["hTx"].values[0]
        training_data_idx["hRx"] = site_config_df[site_config_df["cell_id"] == train_cell_id]["hRx"].values[0]
        training_data_idx["cell_carrier_freq_mhz"] = site_config_df[site_config_df["cell_id"] == train_cell_id][
            "cell_carrier_freq_mhz"
        ].values[0]

        training_data_idx["log_distance"] = [
            GISTools.get_log_distance(
                training_data_idx["cell_lat"].values[0],
                training_data_idx["cell_lon"].values[0],
                lat,
                lon,
            )
            for lat, lon in zip(training_data_idx.loc_y, training_data_idx.loc_x)
        ]
        # filter out "too far" readings that are "too weak"
        # print(len(training_data_idx))
        # training_data_idx = training_data_idx.loc[
        #     ~(
        #         (training_data_idx['cell_rxpwr_dbm'] < filter_out_samples_dbm_threshold)
        #         & (training_data_idx['log_distance'] > np.log(1000 * filter_out_samples_kms_threshold))
        #     )
        # ]
        # print(len(training_data_idx))
        training_data_idx["relative_bearing"] = [
            GISTools.get_relative_bearing(
                training_data_idx["cell_az_deg"].values[0],
                training_data_idx["cell_lat"].values[0],
                training_data_idx["cell_lon"].values[0],
                lat,
                lon,
            )
            for lat, lon in zip(training_data_idx.loc_y, training_data_idx.loc_x)
        ]
        # training_data_idx["antenna_gain"] = GISTools.get_antenna_gain(
        #     training_data_idx["hTx"].values[0],
        #     training_data_idx["hRx"].values[0],
        #     training_data_idx["log_distance"],
        #     training_data_idx["cell_el_deg"],
        # )

    # do training

    bayesian_digital_twins = {}
    loss_vs_iters = []
    for train_cell_id, training_data_idx in training_data.items():
        # filter out "too far" readings that are "too weak"
        training_data_idx = training_data_idx.drop(
            training_data_idx[
                (training_data_idx["cell_rxpwr_dbm"] < filter_out_samples_dbm_threshold)
                & (training_data_idx["log_distance"] > np.log(1000 * filter_out_samples_kms_threshold))
                & (training_data_idx["log_distance"] > np.log(1000 * filter_out_samples_kms_threshold))
            ].index
        )
        if plot_loss_vs_iter:
            axs[1].set_aspect("equal", "box")
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].scatter(
                training_data_idx.loc_x,
                training_data_idx.loc_y,
                c=training_data_idx.cell_rxpwr_dbm,
                cmap=cmap,
                s=25,
            )
        logging.info(f"training cell =  : {train_cell_id}")

        bayesian_digital_twins[train_cell_id] = BayesianDigitalTwin(
            data_in=[training_data_idx],
            x_columns=["log_distance", "relative_bearing", "cell_el_deg"],
            y_columns=["cell_rxpwr_dbm"],
            norm_method=NormMethod.MINMAX,
        )
        loss_vs_iters.append(
            bayesian_digital_twins[train_cell_id].train_distributed_gpmodel(
                maxiter=maxiter,
            )
        )

    # test

    sim_idx_folder = sim_idx_folders[test_idx]
    logging.info(f"loading for testing : {sim_idx_folder}")
    save_path = f"/{bucket_path}/{sim_data_path}/{sim_idx_folder}"
    tilt_test_df = pd.read_csv(f"{save_path}/full_data.csv")

    # drop redundant columns
    tilt_test_df = tilt_test_df.drop(columns=["cell_rxpwr_dbm", "cell_el_deg"])

    # get data by groups
    tilt_test_per_cell_df_list = [x for _, x in tilt_test_df.groupby("cell_id")]

    test_data = {}

    for i in range(len(tilt_test_per_cell_df_list)):
        tilt_test_per_cell_df_list[i] = get_percell_data(
            data_in=tilt_test_per_cell_df_list[i],
            choose_strongest_samples_percell=True,
            n_samples=n_sample_test[i],
        )[0][0]

    for i in range(len(desired_idxs)):
        test_cell_id = idx_cell_id_mapping[i + 1]
        test_data[test_cell_id] = pd.concat(
            [
                tilt_test_per_cell_df_list[i],
            ]
        )

    for test_cell_id, test_data_idx in test_data.items():
        test_data_idx["cell_id"] = test_cell_id
        test_data_idx["cell_lat"] = site_config_df[site_config_df["cell_id"] == test_cell_id]["cell_lat"].values[0]
        test_data_idx["cell_lon"] = site_config_df[site_config_df["cell_id"] == test_cell_id]["cell_lon"].values[0]
        test_data_idx["cell_az_deg"] = site_config_df[site_config_df["cell_id"] == test_cell_id]["cell_az_deg"].values[
            0
        ]
        test_data_idx["cell_txpwr_dbm"] = site_config_df[site_config_df["cell_id"] == test_cell_id][
            "cell_txpwr_dbm"
        ].values[0]
        test_data_idx["hTx"] = site_config_df[site_config_df["cell_id"] == test_cell_id]["hTx"].values[0]
        test_data_idx["hRx"] = site_config_df[site_config_df["cell_id"] == test_cell_id]["hRx"].values[0]
        test_data_idx["cell_carrier_freq_mhz"] = site_config_df[site_config_df["cell_id"] == test_cell_id][
            "cell_carrier_freq_mhz"
        ].values[0]

        test_data_idx["log_distance"] = [
            GISTools.get_log_distance(
                test_data_idx["cell_lat"].values[0],
                test_data_idx["cell_lon"].values[0],
                lat,
                lon,
            )
            for lat, lon in zip(test_data_idx.loc_y, test_data_idx.loc_x)
        ]
        test_data_idx["relative_bearing"] = [
            GISTools.get_relative_bearing(
                test_data_idx["cell_az_deg"].values[0],
                test_data_idx["cell_lat"].values[0],
                test_data_idx["cell_lon"].values[0],
                lat,
                lon,
            )
            for lat, lon in zip(test_data_idx.loc_y, test_data_idx.loc_x)
        ]
        # test_data_idx["antenna_gain"] = GISTools.get_antenna_gain(
        #     test_data_idx["hTx"],
        #     test_data_idx["hRx"],
        #     test_data_idx["log_distance"],
        #     test_data_idx["cell_el_deg"],
        # )

    # predict & merge
    full_prediction_frame = pd.DataFrame()
    bing_tile_level = 22
    for idx, test_data_percell in test_data.items():
        logging.info(f"predicting cell at idx =  : {idx}")
        # filter out "too far" readings that are "too weak"
        # test_data_idx = test_data_idx.loc[
        #     ~(
        #         (test_data_idx['cell_rxpwr_dbm'] < filter_out_samples_dbm_threshold)
        #         & (test_data_idx['log_distance'] > np.log(1000 * filter_out_samples_kms_threshold))
        #     )
        # ]
        test_data_percell = test_data_percell.drop(
            test_data_percell[
                (test_data_percell["cell_rxpwr_dbm"] < filter_out_samples_dbm_threshold)
                & (test_data_percell["log_distance"] > np.log(1000 * filter_out_samples_kms_threshold))
            ].index
        )
        (pred_means_percell, _,) = bayesian_digital_twins[idx].predict_distributed_gpmodel(
            prediction_dfs=[test_data_percell],
        )
        logging.info(f"merging cell at idx =  : {idx}")
        test_data_percell["pred_means"] = pred_means_percell[0]
        # convert to Bing tile, before merging
        test_data_percell_bing_tile = test_data_percell.apply(
            lon_lat_to_bing_tile_df_row, level=bing_tile_level, axis=1
        )
        full_prediction_frame = (
            pd.concat([full_prediction_frame, test_data_percell_bing_tile])
            .groupby(["loc_x", "loc_y"], as_index=False)[["cell_rxpwr_dbm", "pred_means"]]
            .max()
        )
    # re-convert to lat/lon
    full_prediction_frame = full_prediction_frame.apply(bing_tile_to_center_df_row, level=bing_tile_level, axis=1)

    # compute RSRP as maximum over predicted rx powers
    pred_rsrp = np.array(full_prediction_frame.pred_means)
    # extract true (actual) RSRP from test set
    true_rsrp = np.array(full_prediction_frame.cell_rxpwr_dbm)
    # mean absolute error
    MAE = abs(true_rsrp - pred_rsrp).mean()
    # mean square error
    MSE = (abs(true_rsrp - pred_rsrp) ** 2).mean()
    # mean absolute percentage error
    MAPE = 100 * abs((true_rsrp - pred_rsrp) / true_rsrp).mean()
    # 85th percentile error
    Percentile85Error = np.percentile(abs(true_rsrp - pred_rsrp), 85)

    logging.info("==========")
    logging.info(f"MSE = {MSE:0.5f}, MAE = {MAE:0.5f} dB, MAPE = {MAPE:0.5f} %")
    logging.info(f"Percentile85Error = {Percentile85Error:0.1f}")
    logging.info("==========")

    lons = list(full_prediction_frame.loc_x)
    lats = list(full_prediction_frame.loc_y)

    if plot_loss_vs_iter:
        f, axs = plt.subplots(1, 2, figsize=(16, 12))

        axs[0].scatter(lons, lats, c=true_rsrp, cmap=cmap, s=25)
        axs[0].set_title(
            r"Actual RSRP",
            fontsize=14,
        )
        axs[0].set_aspect("equal", "box")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].scatter(lons, lats, c=pred_rsrp, cmap=cmap, s=25)
        axs[1].set_title(
            f"Predicted RSRP \n MAE = {MAE:0.1f} dB, 85th percentile error = {Percentile85Error:0.1f} "
            f"| max_training_iterations = {maxiter} "
            f"| training_data_used_percentage = {p_train}",
            fontsize=12,
            wrap=True,
        )
        axs[1].set_aspect("equal", "box")
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.0, hspace=0.1)
        plt.show()

    return (
        bayesian_digital_twins,
        site_config_df,
        test_data,
        loss_vs_iters,
        lons,
        lats,
        true_rsrp,
        pred_rsrp,
        MAE,
        Percentile85Error,
    )


def animate_predictions(
    lats,
    lons,
    true_rsrp,
    pred_rsrp_list,
    MAE_list,
    Percentile85Error_list,
    maxiter_list,
    p_train_list,
    filename,
    cmap="PuBuGn",
):
    """ "
    Create an animation visualizing true and predicted RSRP values over geographical coordinates.
    """
    if not FFMPEG_PATH:
        print("Please provide ffmpeg path to create animation")
        return
    plt.rcParams["animation.ffmpeg_path"] = FFMPEG_PATH

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout(pad=0)

    true_rsrp_points = axs[0].scatter(lons, lats, c=true_rsrp, cmap=cmap, s=25)
    pred_rsrp_points = axs[1].scatter(lons, lats, c=pred_rsrp_list[0], cmap=cmap, s=25)

    plt.show()

    def _init_plt(axs):
        axs[0].set_title(
            r"Actual RSRP",
            fontsize=12,
        )
        axs[0].set_aspect("equal", "box")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_title(
            f"Predicted RSRP \n MAE = {MAE_list[0]:0.1f} dB"
            f"\nmax_training_iterations = {maxiter_list[0]} | "
            f"training_data_used_percentage = {p_train_list[0]}",
            fontsize=12,
            wrap=True,
        )
        axs[1].set_aspect("equal", "box")
        axs[1].set_xticks([])
        axs[1].set_yticks([])

    # fig.colorbar(sinr_points)

    # initialization function: plot the background of each frame
    def init():
        """
        Initialize the plotting environment by clearing the current figure and setting up the axes.
        @returns: A list containing the true RSRP points and predicted RSRP points.
        """
        plt.clf()
        _init_plt(axs)
        # pred_rsrp_points.set_offsets([])
        return [true_rsrp_points, pred_rsrp_points]

    # animation function.  This is called sequentially
    def animate(i):
        """
        Update the animation frame for visualizing predicted RSRP values.
        @param i: Index of the current animation frame.
        @returns: A list containing the scatter plot objects for true RSRP points
                  and predicted RSRP points.
        """

        plt.clf()
        _init_plt(axs)
        pred_rsrp_points = axs[1].scatter(lons, lats, c=pred_rsrp_list[i], cmap=cmap, s=25)
        axs[1].set_title(
            f"Predicted RSRP \n MAE = {MAE_list[i]:0.1f} dB"
            f"\nmax_training_iterations = {maxiter_list[i]} | "
            f"training_data_used_percentage = {p_train_list[i]}",
            fontsize=12,
            wrap=True,
        )
        axs[1].set_aspect("equal", "box")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        return [true_rsrp_points, pred_rsrp_points]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(pred_rsrp_list), blit=True)

    writervideo = animation.FFMpegWriter(fps=4)
    anim.save(filename, writer=writervideo)

    return anim, fig


def rfco_to_best_server_shapes(
    spwr_src_raster: np.ndarray,
    min_lat: float,
    min_lon: float,
    step: float,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Vectorize the RFCoverageObject coverage areas, based on signal power source raster.

    For each site (int in spwr_src_raster), create a geojson-like dict of geometries and
    the value associated with them in spwr_src_raster. Return a list with one dict+value
    per site.
    """
    # transform like GDAL GeoTransform
    tf = Affine.from_gdal(min_lon, 0, step, min_lat, step, 0)
    shapes = sorted(
        rasterio.features.shapes(spwr_src_raster, transform=tf),
        key=lambda x: x[1],
    )
    return shapes


# Mobility Model helper functions


def get_ue_data(params: dict) -> pd.DataFrame:
    """
    Generates user equipment (UE) tracks data using specified simulation parameters.

    This function initializes a UETracksGenerationParams object using the provided parameters
    and then iterates over batches generated by the UETracksGenerator. Each batch of UE tracks
    data is consolidated into a single DataFrame which captures mobility tracks across multiple
    ticks and batches, as per the defined parameters.

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

    # Initialize the UE data
    ue_tracks_params = UETracksGenerationParams(params)

    ue_tracks_generation = pd.DataFrame()  # Initialize an empty DataFrame
    for ue_tracks_generation_batch in UETracksGenerator.generate_as_lon_lat_points(
        rng_seed=ue_tracks_params.rng_seed,
        lon_x_dims=ue_tracks_params.lon_x_dims,
        lon_y_dims=ue_tracks_params.lon_y_dims,
        num_ticks=ue_tracks_params.num_ticks,
        num_UEs=ue_tracks_params.num_UEs,
        num_batches=ue_tracks_params.num_batches,
        alpha=ue_tracks_params.alpha,
        variance=ue_tracks_params.variance,
        min_lat=ue_tracks_params.min_lat,
        max_lat=ue_tracks_params.max_lat,
        min_lon=ue_tracks_params.min_lon,
        max_lon=ue_tracks_params.max_lon,
        mobility_class_distribution=ue_tracks_params.mobility_class_distribution,
        mobility_class_velocities=ue_tracks_params.mobility_class_velocities,
        mobility_class_velocity_variances=ue_tracks_params.mobility_class_velocity_variances,
    ):
        # Append each batch to the main DataFrame
        ue_tracks_generation = pd.concat([ue_tracks_generation, ue_tracks_generation_batch], ignore_index=True)

    return ue_tracks_generation


def calculate_received_power(distance_km: float, frequency_mhz: int) -> float:
    """
    Calculate received power using the Free-Space Path Loss (FSPL) model.
    """
    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    # Calculate Free-Space Path Loss (FSPL) in dB
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_mhz) - 27.55

    # Calculate and return the received power in dBm
    received_power_dbm = TXPWR_DBM - fspl_db
    return received_power_dbm


def plot_ue_tracks(df: pd.DataFrame) -> None:
    """
    Plots the movement tracks of unique UE IDs on a grid of subplots.
    
    +-------------+------+-----------+------------+
    | mock_ue_id  | tick |   lat     |    lon     |
    +=============+======+===========+============+
    |     1       |  0   | 23.8103   | 90.4125    |
    |     1       |  1   | 23.8109   | 90.4130    |
    |     1       |  2   | 23.8115   | 90.4135    |
    |     2       |  0   | 23.8120   | 90.4140    |
    |     2       |  1   | 23.8125   | 90.4145    |
    |     2       |  2   | 23.8130   | 90.4150    |
    +-------------+------+-----------+------------+

    """

    # Initialize an empty list to store batch indices
    batch_indices = []

    # Identify where tick resets and mark the indices
    for i in range(1, len(df)):
        if df.loc[i, "tick"] == 0 and df.loc[i - 1, "tick"] != 0:
            batch_indices.append(i)

    # Add the final index to close the last batch
    batch_indices.append(len(df))

    # Now, iterate over the identified batches
    start_idx = 0
    for batch_num, end_idx in enumerate(batch_indices):
        batch_data = df.iloc[start_idx:end_idx]

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Generate a color map with different colors for each ue_id
        color_map = cm.get_cmap("tab20", len(batch_data["mock_ue_id"].unique()))

        # Plot each ue_id's movement over ticks in this batch
        for idx, ue_id in enumerate(batch_data["mock_ue_id"].unique()):
            ue_data = batch_data[batch_data["mock_ue_id"] == ue_id]
            color = color_map(idx)  # Get a unique color for each ue_id

            # Plot the path with arrows
            for i in range(len(ue_data) - 1):
                x_start = ue_data.iloc[i]["lon"]
                y_start = ue_data.iloc[i]["lat"]
                x_end = ue_data.iloc[i + 1]["lon"]
                y_end = ue_data.iloc[i + 1]["lat"]

                # Calculate the direction vector
                dx = x_end - x_start
                dy = y_end - y_start

                # Plot the line with an arrow with reduced width and unique color
                plt.quiver(
                    x_start,
                    y_start,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=color,
                    width=0.002,
                    headwidth=3,
                    headlength=5,
                )

            # Plot starting points as circles with the same color
            plt.scatter(
                ue_data["lon"].iloc[0],
                ue_data["lat"].iloc[0],
                color=color,
                label=f"Start UE {ue_id}",
            )

        # Set plot title and labels
        plt.title(f"UE Tracks with Direction for Batch {batch_num + 1}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))

        # Display the plot
        plt.show()

        # Update start_idx for the next batch
        start_idx = end_idx


def plot_ue_tracks_side_by_side(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Plots the movement tracks of unique UE IDs from two DataFrames side by side.
    
    df1:
    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8101   | 90.4100    |
    |     2       | 23.8105   | 90.4110    |
    |     3       | 23.8110   | 90.4120    |
    |     4       | 23.8115   | 90.4130    |
    +-------------+-----------+------------+
    
    df2: 
    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8120   | 90.4140    |
    |     2       | 23.8125   | 90.4150    |
    |     3       | 23.8130   | 90.4160    |
    |     4       | 23.8135   | 90.4170    |
    +-------------+-----------+------------+

    """
    # Set up subplots with 2 columns for side by side plots
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # 2 rows, 2 columns (side by side)

    # Plot the first DataFrame
    plot_ue_tracks_on_axis(df1, axes[0], title="DataFrame 1")

    # Plot the second DataFrame
    plot_ue_tracks_on_axis(df2, axes[1], title="DataFrame 2")

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


def plot_ue_tracks_on_axis(df: pd.DataFrame, ax, title: str) -> None:
    """
    Helper function to plot UE tracks on a given axis.
    
    +-------------+-----------+------------+
    | mock_ue_id  |   lat     |    lon     |
    +=============+===========+============+
    |     1       | 23.8103   | 90.4125    |
    |     1       | 23.8109   | 90.4130    |
    |     1       | 23.8115   | 90.4135    |
    |     2       | 23.8120   | 90.4140    |
    |     2       | 23.8125   | 90.4145    |
    |     2       | 23.8130   | 90.4150    |
    +-------------+-----------+------------+

    
    """
    data = df
    unique_ids = data["mock_ue_id"].unique()
    num_plots = len(unique_ids)

    color_map = cm.get_cmap("tab20", num_plots)

    for idx, ue_id in enumerate(unique_ids):
        ue_data = data[data["mock_ue_id"] == ue_id]

        for i in range(len(ue_data) - 1):
            x_start = ue_data.iloc[i]["lon"]
            y_start = ue_data.iloc[i]["lat"]
            x_end = ue_data.iloc[i + 1]["lon"]
            y_end = ue_data.iloc[i + 1]["lat"]

            dx = x_end - x_start
            dy = y_end - y_start
            ax.quiver(
                x_start,
                y_start,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color_map(idx),
            )

        ax.scatter(ue_data["lon"], ue_data["lat"], color=color_map(idx), label=f"UE {ue_id}")

    ax.set_title(title)
    ax.legend()


# MRO app helper functions

# Scatter plot of the Cell towers and UE Locations


def mro_plot_scatter(df: pd.DataFrame, topology: pd.DataFrame) -> None:
    """
    Plot a scatter plot of cell towers and UE (User Equipment) locations.
    @param df: DataFrame containing UE data with columns 'loc_x', 'loc_y', 'cell_id', and 'sinr_db'.
    @param topology: DataFrame containing cell tower data with columns 'cell_lon', 'cell_lat', and 'cell_id'.
    @returns: None. Displays a scatter plot with cell towers and UE locations.
    
    df:
    +---------+--------+--------+----------+
    | cell_id | loc_x  | loc_y  | sinr_db  |
    +=========+========+========+==========+
    |    1    | 90.412 | 23.810 |   15.3   |
    |    2    | 90.413 | 23.811 |   12.1   |
    |    1    | 90.415 | 23.812 |   18.7   |
    |    2    | 90.416 | 23.813 |    5.5   |
    +---------+--------+--------+----------+

    topology:
    +---------+----------+----------+
    | cell_id | cell_lon | cell_lat |
    +=========+==========+==========+
    |    1    | 90.410   | 23.809   |
    |    2    | 90.414   | 23.810   |
    +---------+----------+----------+

    """

    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    plt.scatter([], [], color="grey", label="RLF")

    # Define color mapping based on cell_id for both cells and UEs
    color_map = {1: "red", 2: "green", 3: "blue"}

    # Plot cell towers from the topology dataframe with 'X' markers and corresponding colors
    for _, row in topology.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        plt.scatter(
            row["cell_lon"],
            row["cell_lat"],
            marker="x",
            color=color,
            s=200,
            label=f"Cell {row['cell_id']}",
        )

    # Plot UEs from df without labels but with the same color coding
    for _, row in df.iterrows():
        color = color_map.get(row["cell_id"], "black")  # Default to black if unknown cell_id
        if row["sinr_db"] < RLF_THRESHOLD:  # REMOVE COMMENT WHEN sinr_db IS FIXED
            color = "grey"  # Change to grey if sinr_db < 2

        plt.scatter(row["loc_x"], row["loc_y"], color=color)

    # Add labels and title
    plt.xlabel("Longitude (loc_x)")
    plt.ylabel("Latitude (loc_y)")
    plt.title("Cell Towers and UE Locations")

    # Create a legend for the cells only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.show()


def get_ues_cells_cartesian_df(data: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """
    returns a cartesian dataframe of UE and cell data
    
    df:
    +---------+-----------+------------+----------+
    |  ue_id  | latitude  | longitude  |   tick   |
    +=========+===========+============+==========+
    |    1    | 90.412    | 23.810     |     0    |
    |    2    | 90.413    | 23.811     |     0    |
    |    1    | 90.415    | 23.812     |     1    |
    |    2    | 90.416    | 23.813     |     1    |
    +---------+-----------+------------+----------+

    topology:
    +---------+----------+----------+--------------+------------------------+
    | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +=========+==========+==========+==============+========================+
    |    1    | 90.410   | 23.809   |     120      |         1800           |
    |    2    | 90.414   | 23.810   |     240      |         2100           |
    +---------+----------+----------+--------------+------------------------+

    """
    if topology["cell_id"].dtype == object:
        topology["cell_id"] = topology["cell_id"].str.replace("cell_", "").astype(int)
    data["key"] = 1
    topology["key"] = 1
    cartesian_df = pd.merge(data, topology, on="key").drop("key", axis=1)

    data.drop(columns=["key"], inplace=True)
    topology.drop(columns=["key"], inplace=True)

    return cartesian_df


def calc_log_distance(cartesian_df: pd.DataFrame) -> pd.DataFrame:
    """
    adds a log distance column to the cartesian dataframe based on the lat/lon of the UE and cell
    
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +========+==========+===========+======+=========+==========+==========+==============+========================+
    |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120       |        1800           |
    |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120       |        1800           |
    |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240       |        2100           |
    |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240       |        2100           |
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+

    """
    cartesian_df["log_distance"] = cartesian_df.apply(
        lambda row: GISTools.get_log_distance(row["latitude"], row["longitude"], row["cell_lat"], row["cell_lon"]),
        axis=1,
    )
    return cartesian_df


def calc_rx_power(cartesian_df: pd.DataFrame) -> pd.DataFrame:
    """
    adds a cell_rxpwr_dbm column to the cartesian dataframe,
    based on the log distance and cell frequency using fspl
    
    
    +--------+----------+-----------+------+---------+----------+----------+--------------+--------------+------------------------+
    | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  | log_distance |
    +========+==========+===========+======+=========+==========+==========+==============+========================+==============+
    |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120       |        1800           |   -2.546     |
    |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120       |        1800           |   -2.850     |
    |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240       |        2100           |   -2.268     |
    |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240       |        2100           |   -2.547     |
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+--------------+

    """
    cartesian_df["cell_rxpwr_dbm"] = cartesian_df.apply(
        lambda row: calculate_received_power(row["log_distance"], row["cell_carrier_freq_mhz"]),
        axis=1,
    )
    return cartesian_df


def calc_relative_bearing(cartesian_df: pd.DataFrame) -> pd.DataFrame:
    """
    adds a relative_bearing column to the cartesian dataframe,
    based on the lat/lon of the UE and cell and az_deg of the cell
    
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +========+==========+===========+======+=========+==========+==========+==============+========================+
    |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120      |         1800           |
    |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120      |         1800           |
    |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240      |         2100           |
    |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240      |         2100           |
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    
    """
    cartesian_df["relative_bearing"] = cartesian_df.apply(
        lambda row: GISTools.get_relative_bearing(
            row["cell_az_deg"],
            row["cell_lat"],
            row["cell_lon"],
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )
    return cartesian_df


def preprocess_ue_data(data: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """
    creates a cartesian dataframe of UE and cell data, adds log distance and rx power columns
    
    df:
    +---------+-----------+------------+----------+
    |  ue_id  | latitude  | longitude  |   tick   |
    +=========+===========+============+==========+
    |    1    | 90.412    | 23.810     |     0    |
    |    2    | 90.413    | 23.811     |     0    |
    |    1    | 90.415    | 23.812     |     1    |
    |    2    | 90.416    | 23.813     |     1    |
    +---------+-----------+------------+----------+

    topology:
    +---------+----------+----------+--------------+------------------------+
    | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +=========+==========+==========+==============+========================+
    |    1    | 90.410   | 23.809   |     120      |         1800           |
    |    2    | 90.414   | 23.810   |     240      |         2100           |
    +---------+----------+----------+--------------+------------------------+

    """
    cartesian_df = get_ues_cells_cartesian_df(data, topology)
    cartesian_df = calc_log_distance(cartesian_df)
    return calc_rx_power(cartesian_df)


def normalize_cell_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'cell_id' column in the DataFrame by ensuring all IDs follow the 'cell_<integer>' format.
    
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +========+==========+===========+======+=========+==========+==========+==============+========================+
    |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120      |         1800           |
    |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120      |         1800           |
    |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240      |         2100           |
    |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240      |         2100           |
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    
    """

    df = df.copy()
    df["cell_id"] = df["cell_id"].apply(lambda x: f"cell_{int(float(x))}" if not str(x).startswith("cell_") else str(x))
    return df


def check_cartesian_format(df: pd.DataFrame, topology: pd.DataFrame) -> bool:
    """
    Validates that the DataFrame has the expected cartesian format for cell IDs per pixel.
    
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    | ue_id  | latitude | longitude | tick | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +========+==========+===========+======+=========+==========+==========+==============+========================+
    |   0    | 90.412   | 23.810    |  0   |    1    | 90.410   | 23.809   |     120      |         1800           |
    |   1    | 90.413   | 23.811    |  0   |    1    | 90.414   | 23.810   |     120      |         1800           |
    |   0    | 90.415   | 23.812    |  1   |    2    | 90.410   | 23.809   |     240      |         2100           |
    |   1    | 90.416   | 23.813    |  1   |    2    | 90.414   | 23.810   |     240      |         2100           |
    +--------+----------+-----------+------+---------+----------+----------+--------------+------------------------+
    
    """
    expected_cells = list(topology["cell_id"])
    expected_cell_set = set(expected_cells)
    num_expected_cells = len(expected_cells)

    # Check if the set of cell_ids in the DataFrame matches the expected set
    actual_cell_set = set(df["cell_id"])
    if actual_cell_set != expected_cell_set:
        missing_cells = expected_cell_set - actual_cell_set
        extra_cells = actual_cell_set - expected_cell_set

        raise ValueError(
            f"Cell ID mismatch detected:\n" f"  Missing cells: {missing_cells}\n" f"  Extra cells: {extra_cells}"
        )

    # Group by pixel
    grouped = df.groupby(["latitude", "longitude"])

    for (lat, lon), group in grouped:
        cell_ids = list(group["cell_id"])
        cell_counts = pd.Series(cell_ids).value_counts()

        total_rows = len(cell_ids)

        if total_rows % num_expected_cells != 0:
            raise ValueError(
                f"""
                For Pixel ({lat}, {lon}): total rows = {total_rows} not divisible
                by expected # of cells from topology = {num_expected_cells}, indicating missing or extra cells.
                """
            )

        k = total_rows // num_expected_cells  # number of revisits

        # Check exact counts for each expected cell_id
        extra = []
        wrong_counts = []

        for cell in expected_cell_set:
            count = cell_counts.get(cell, 0)
            if count != k:
                wrong_counts.append((cell, count))

        unexpected_cells = set(cell_counts.index) - expected_cell_set
        if unexpected_cells:
            extra.extend(unexpected_cells)

        if wrong_counts or extra:
            raise ValueError(
                f"Pixel ({lat}, {lon}):\n"
                f"  Expected {k} of each: {expected_cell_set}\n"
                f"  Wrong counts: {wrong_counts}\n"
                f"  Unexpected cells: {extra}"
            )

    return True


def add_cell_info(new_data_with_rx_data: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cell information ['cell_id', 'cell_lat', 'cell_lon', 'cell_az_deg']
    to the DataFrame based on cell_id.

    Converts integer cell_id to string format like 'cell_1' to match topology.
    
    df:
    +---------+-----------+------------+----------+----------------+
    | ue_id   | latitude  | longitude  | tick     | cell_rxpwr_dbm |
    +=========+===========+============+==========+================+
    |    1    | 90.412    | 23.810     |    0     |      -85       |
    |    2    | 90.413    | 23.811     |    0     |      -88       |
    |    1    | 90.415    | 23.812     |    1     |      -80       |
    |    2    | 90.416    | 23.813     |    1     |      -90       |
    +---------+-----------+------------+----------+----------------+

    topology:
    +---------+----------+----------+--------------+------------------------+
    | cell_id | cell_lon | cell_lat | cell_az_deg  | cell_carrier_freq_mhz  |
    +=========+==========+==========+==============+========================+
    |    1    | 90.410   | 23.809   |     120      |         1800           |
    |    2    | 90.414   | 23.810   |     240      |         2100           |
    +---------+----------+----------+--------------+------------------------+

    
    """
    # Convert int to str format matching topology: 'cell_1', 'cell_2', etc.
    if new_data_with_rx_data["cell_id"].dtype == int:
        new_data_with_rx_data["cell_id"] = new_data_with_rx_data["cell_id"].apply(lambda x: f"cell_{x}")

    # Merge using consistent cell_id format
    new_data_topology_merged = new_data_with_rx_data.merge(
        topology[["cell_id", "cell_lat", "cell_lon", "cell_az_deg"]],
        on="cell_id",
        how="left",
    )

    return new_data_topology_merged


def plot_sinr_db_by_ue(df: pd.DataFrame, df2: pd.DataFrame, ue_id: int) -> None:
    """
    Plots SINR (in dB) over ticks for a specific ue_id.

    - Solid bold line: Connected cell_id (from df), color-coded.
    - Dotted lines: All cell_id sinr_db values from df2 for context.
    - RLF events: Drop to bottom with bold black line.
    - RLF_THRESHOLD: Horizontal dashed line.

    Parameters:
    df (pd.DataFrame): Connected cell data: 'ue_id', 'tick', 'sinr_db', 'cell_id' (or 'RLF').
    df2 (pd.DataFrame): All candidate cell data: 'ue_id', 'tick', 'cell_id', 'sinr_db'.
    topology (pd.DataFrame): Not used.
    ue_id (int): UE to plot.
    
    +--------+------+----------+----------+
    | ue_id  | tick | cell_id  | sinr_db  |
    +========+======+==========+==========+
    |   0    |  0   |    1     |  14.0    |
    |   0    |  0   |    2     |  12.5    |
    |   1    |  0   |    1     |  13.2    |
    |   1    |  0   |    2     |  10.8    |
    |   0    |  1   |    1     |  16.7    |
    |   0    |  1   |    2     |  12.3    |
    |   1    |  1   |    1     |  -2.0    |
    |   1    |  1   |    2     |  -4.3    |
    +--------+------+----------+----------+ 
    
    """
    ue_df = df[df["ue_id"] == ue_id].sort_values("tick").reset_index(drop=True)
    ue_df2 = df2[df2["ue_id"] == ue_id].sort_values("tick")

    if ue_df.empty or ue_df2.empty:
        print(f"No data found for ue_id {ue_id}.")
        return

    # Base + dynamic color map
    base_colors = {1.0: "red", 2.0: "green", 3.0: "blue"}
    all_cell_ids = pd.concat([
        ue_df2["cell_id"],
        ue_df[ue_df["cell_id"] != "RLF"]["cell_id"]
    ]).unique()
    missing_ids = [cid for cid in all_cell_ids if cid not in base_colors]
    extra_colors = cm.get_cmap("tab10", len(missing_ids))
    dynamic_colors = {cid: extra_colors(i) for i, cid in enumerate(missing_ids)}
    full_color_map = {**base_colors, **dynamic_colors}

    min_sinr = min(
        ue_df2["sinr_db"].min(),
        ue_df[ue_df["cell_id"] != "RLF"]["sinr_db"].min()
    )
    drop_value = min_sinr - 5

    plt.figure(figsize=(12, 6))
    legend_cells = set()

    # --- Plot all candidate cell SINRs (dotted, bold) ---
    for cell_id, group in ue_df2.groupby("cell_id"):
        label = f"cell_id {cell_id}" if cell_id not in legend_cells else None
        legend_cells.add(cell_id)
        plt.plot(
            group["tick"],
            group["sinr_db"],
            linestyle=":",
            linewidth=2.5,
            color=full_color_map.get(cell_id, "gray"),
            label=label,
            alpha=0.7,
        )

    # --- Plot connected UE SINR as a continuous line, color-coded per cell_id ---
    previous_idx = None
    for i in range(len(ue_df) - 1):
        tick1, tick2 = ue_df.loc[i, "tick"], ue_df.loc[i + 1, "tick"]
        sinr1, sinr2 = ue_df.loc[i, "sinr_db"], ue_df.loc[i + 1, "sinr_db"]
        cell1, cell2 = ue_df.loc[i, "cell_id"], ue_df.loc[i + 1, "cell_id"]

        # If current or next point is RLF, break the line
        if cell1 == "RLF" or cell2 == "RLF":
            continue

        # Draw line from point i to i+1 with color of current cell
        label = f"cell_id {cell1}" if cell1 not in legend_cells else None
        if label:
            legend_cells.add(cell1)
        plt.plot(
            [tick1, tick2],
            [sinr1, sinr2],
            color=full_color_map.get(cell1, "gray"),
            linewidth=3,
            label=label,
        )

    # --- Plot RLFs as vertical drops ---
    rlf_ticks = ue_df[ue_df["cell_id"] == "RLF"]["tick"]
    if not rlf_ticks.empty:
        for rlf_tick in rlf_ticks:
            plt.plot(
                [rlf_tick],
                [drop_value],
                "ko",
                markersize=8,
                label="RLF" if "RLF" not in legend_cells else None
            )
            legend_cells.add("RLF")

    # --- RLF Threshold ---
    plt.axhline(y=RLF_THRESHOLD, color="black", linestyle="--", linewidth=2)

    # --- Final Decorations ---
    plt.title(f"SINR over Time for UE ID {ue_id}")
    plt.xlabel("Tick")
    plt.ylabel("SINR (dB)")
    plt.grid(True)
    plt.legend(title=None, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def mro_score_3d_plot(df: pd.DataFrame) -> None:
    """
    Create an interactive 3D scatter plot using Plotly.

    Parameters:
    - df (pd.DataFrame): A DataFrame with columns ['hyst', 'ttt', 'score']
    """
    # Validate input
    required_cols = {"hyst", "ttt", "score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Create plot
    fig = px.scatter_3d(
        df,
        x="ttt",
        y="hyst",
        z="score",
        color="score",
        color_continuous_scale="Viridis",
        title="Interactive 3D Plot: ttt vs hyst vs score",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    fig.show()
