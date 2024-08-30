from enum import Enum
from typing import Dict, Generator, List, Tuple

from typing import List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from radp.common import constants
from radp.digital_twin.mobility.ue_tracks import UETracksGenerator
from radp.digital_twin.utils.gis_tools import GISTools
from radp.digital_twin.mobility.ue_tracks import MobilityClass




class UETracksGenerationHandler:
    
    """
    A handler for generating and managing User Equipment (UE) tracks based on the Gauss-Markov mobility model.

    This class provides functionality to generate UE tracks, convert them to longitude and latitude coordinates,
    plot them with direction indicators, and save the generated data to a CSV file. The tracks are generated 
    according to specified parameters such as the number of ticks, UEs, batches, and mobility characteristics.

    Attributes:
        rng_seed (int): Seed for the random number generator.
        num_batches (int): Number of batches to generate.
        lon_x_dims (int): Longitudinal dimension for x-coordinates.
        lon_y_dims (int): Longitudinal dimension for y-coordinates.
        num_ticks (int): Number of ticks per batch.
        num_UEs (int): Number of User Equipment (UE) instances.
        alpha (float): Alpha parameter for the Gauss-Markov mobility model.
        variance (float): Variance parameter for the Gauss-Markov mobility model.
        min_lat (float): Minimum latitude boundary.
        max_lat (float): Maximum latitude boundary.
        min_lon (float): Minimum longitude boundary.
        max_lon (float): Maximum longitude boundary.
        mobility_class_distribution (Dict[MobilityClass, float]): Distribution of mobility classes.
        mobility_class_velocities (Dict[MobilityClass, float]): Average velocities for each mobility class.
        mobility_class_velocity_variances (Dict[MobilityClass, float]): Variance of velocities for each mobility class.
    """
    
    def __init__(
        self,
        rng_seed: int,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
        num_batches : int = 1,
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
        ):

        self.rng_seed = rng_seed
        self.num_batches = num_batches
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
        self.mobility_class_distribution = mobility_class_distribution
        self.mobility_class_velocities = mobility_class_velocities
        self.mobility_class_velocity_variances = mobility_class_velocity_variances



    def generate_ue_data(self) -> List[Any]:

        """
        Generates the UE Tracks using the specified Gauss-Markov Mobility Parameters.
        
        return: A list containing the batches of all the UE Tracks data created.
        """

        ue_generator = UETracksGenerator(
            rng=np.random.default_rng(self.rng_seed),
            mobility_class_distribution=self.mobility_class_distribution,
            mobility_class_velocities=self.mobility_class_velocities,
            mobility_class_velocity_variances=self.mobility_class_velocity_variances,
            lon_x_dims=self.lon_x_dims,
            lon_y_dims=self.lon_y_dims,
            num_ticks=self.num_ticks,
            num_UEs=self.num_UEs,
            alpha = self.alpha,
            variance = self.variance,
            min_lat = self.min_lat,
            max_lat = self.max_lat,
            min_lon = self.min_lon,
            max_lon = self.max_lon
            ) 

        all_batches = []
        for batch_no in range(self.num_batches):
            batch = next(ue_generator.generate())
            all_batches.append(batch)
        print("Batches of Data: " , all_batches)
        return all_batches



    def mobility_data_generation(self) -> pd.DataFrame:

        """
        Generates UE Mobility Data, converts X and Y Co-ordinates to Latitude and Longitude respectively,
        and organises everything into dataframes of the following format:
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
            
        return: A pandas DataFrame containing the UE tracks data of all the batches.
        """
        
        ue_data = self.generate_ue_data()
        all_dataframes = []  # List to hold all batch dataframes
<<<<<<< HEAD

        # Iterate through each batch
        for batch_no, xy_batches in enumerate(ue_data):
            batch_dataframe_list = []

          # Iterate through ticks in each batch
            for tick, xy_batch in enumerate(xy_batches):
              # Transforming (x, y) into (lon, lat)
                lon_lat_pairs = converting_xy_points_into_lonlat_pairs(
                  xy_batch,
                  self.lon_x_dims,
                  self.lon_y_dims,
                  self.min_lon,
                  self.max_lon,
                  self.min_lat,
                  self.max_lat
                )

                # Building DataFrame for this tick
                df_tick = pd.DataFrame({
                    'mock_ue_id': range(self.num_UEs),
                    'longitude': [pair[0] for pair in lon_lat_pairs],
                    'latitude': [pair[1] for pair in lon_lat_pairs],
                    'tick': np.full(self.num_UEs, tick)
                })

                batch_dataframe_list.append(df_tick)

            # Concatenate all ticks data for the current batch
            all_dataframes.append(pd.concat(batch_dataframe_list, ignore_index=True))

            if batch_no + 1 >= self.num_batches:  # Check if the specified number of batches has been reached
                break

=======

        # Iterate through each batch
        for batch_no, xy_batches in enumerate(ue_data):
            batch_dataframe_list = []

            # Iterate through ticks in each batch
            for tick, xy_batch in enumerate(xy_batches):
                # Transforming (x, y) into (lon, lat)
                lon_lat_pairs = GISTools.converting_xy_points_into_lonlat_pairs(
                    xy_batch,
                    self.lon_x_dims,
                    self.lon_y_dims,
                    self.min_lon,
                    self.max_lon,
                    self.min_lat,
                    self.max_lat
                )

                # Building DataFrame for this tick
                df_tick = pd.DataFrame({
                    'mock_ue_id': range(self.num_UEs),
                    'longitude': [pair[0] for pair in lon_lat_pairs],
                    'latitude': [pair[1] for pair in lon_lat_pairs],
                    'tick': np.full(self.num_UEs, tick)
                })

                batch_dataframe_list.append(df_tick)

            # Concatenate all ticks data for the current batch
            all_dataframes.append(pd.concat(batch_dataframe_list, ignore_index=True))

            if batch_no + 1 >= self.num_batches:  # Check if the specified number of batches has been reached
                break

>>>>>>> origin/tanzim/mobility-model
        # Concatenate all batches into a single DataFrame and return
        return pd.concat(all_dataframes, ignore_index=True)


    def plot_ue_tracks(self,csv_file) -> None:

        """
        Plot UE tracks from a CSV file in respect to time. 
        Each plotted graph represents a batch of data, and the lines and arrows in the graph
        are used to represent the direction of movement of the respective UE's.
        
        param csv_file: Path to the CSV file containing the UE Tracks data.
        """
        
        # Load the data
        data = pd.read_csv(csv_file)

        # Initialize an empty list to store batch indices
        batch_indices = []

        # Identify where tick resets and mark the indices
        for i in range(1, len(data)):
            if data.loc[i, 'tick'] == 0 and data.loc[i-1, 'tick'] != 0:
                batch_indices.append(i)

        # Add the final index to close the last batch
        batch_indices.append(len(data))

        # Now, iterate over the identified batches
        start_idx = 0
        for batch_num, end_idx in enumerate(batch_indices):
            batch_data = data.iloc[start_idx:end_idx]

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Generate a color map with different colors for each ue_id
            color_map = cm.get_cmap('tab20', len(batch_data['mock_ue_id'].unique()))

            # Plot each ue_id's movement over ticks in this batch
            for idx, ue_id in enumerate(batch_data['mock_ue_id'].unique()):
                ue_data = batch_data[batch_data['mock_ue_id'] == ue_id]
                color = color_map(idx)  # Get a unique color for each ue_id

                # Plot the path with arrows
                for i in range(len(ue_data) - 1):
                    x_start = ue_data.iloc[i]['longitude']
                    y_start = ue_data.iloc[i]['latitude']
                    x_end = ue_data.iloc[i+1]['longitude']
                    y_end = ue_data.iloc[i+1]['latitude']

                    # Calculate the direction vector
                    dx = x_end - x_start
                    dy = y_end - y_start

                    # Plot the line with an arrow with reduced width and unique color
                    plt.quiver(x_start, y_start, dx, dy, angles='xy', scale_units='xy', scale=1, color=color,
                              width=0.002, headwidth=3, headlength=5)

                # Plot starting points as circles with the same color
                plt.scatter(ue_data['longitude'].iloc[0], ue_data['latitude'].iloc[0], color=color, label=f'Start UE {ue_id}')

            # Set plot title and labels
            plt.title(f'UE Tracks with Direction for Batch {batch_num + 1}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()

            # Display the plot
            plt.show()

            # Update start_idx for the next batch
            start_idx = end_idx

    def save_data_to_csv(self, filename: str) -> None:
        """
        Save the UE tracks data to a CSV file.

        :param filename: The name of the file to save the CSV data to.
        """
        # Generate the data
        data = self.mobility_data_generation()

        # Save DataFrame to CSV
        data.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")



