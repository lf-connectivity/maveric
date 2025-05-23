# maveric/apps/traffic_load/config_gen.py

import os
import sys
import logging
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

# Attempt to import from RADP library for constants and GISTools.
# These might be used if the dummy training data generation becomes more sophisticated.
# For now, GISTools is primarily used for distance in dummy RSRP.
try:
    from radp.digital_twin.utils import constants as c
    from radp.digital_twin.utils.gis_tools import GISTools
except ImportError:
    print("Warning: Failed to import RADP constants or GISTools for config_gen.py.")
    print("Ensure RADP_ROOT is in PYTHONPATH. Using fallback definitions.")
    class c: # Define fallback constants ONLY IF import fails
        CELL_ID = "cell_id"; CELL_LAT = "cell_lat"; CELL_LON = "cell_lon"
        CELL_AZ_DEG = "cell_az_deg"
        ECGI = "ecgi"; SITE_ID = "site_id"; CELL_NAME = "cell_name"
        ENODEB_ID = "enodeb_id"; TAC = "tac"; CELL_CARRIER_FREQ_MHZ = "cell_carrier_freq_mhz"
        LAT = "lat"; LON = "lon"; CELL_EL_DEG = "cell_el_deg"
    class GISTools: # Minimal fallback for distance if needed
        @staticmethod
        def dist(coord1, coord2):
             R = 6371.0; lat1, lon1 = map(np.radians, coord1); lat2, lon2 = map(np.radians, coord2)
             dlon = lon2 - lon1; dlat = lat2 - lat1
             a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
             return R * 2 * np.arcsin(np.sqrt(a))

logger = logging.getLogger(__name__)

class ScenarioConfigurationGenerator:
    """
    Generates configuration files for traffic simulation scenarios, including
    network topology, initial cell parameters, and dummy training data.
    """

    def __init__(self):
        """Initializes the ScenarioConfigurationGenerator."""
        pass

    def _generate_dummy_topology_df(self,
                                   num_sites: int,
                                   cells_per_site: int = 3,
                                   lat_range: Tuple[float, float] = (40.7, 40.8),
                                   lon_range: Tuple[float, float] = (-74.05, -73.95),
                                   start_ecgi: int = 1001,
                                   start_enodeb_id: int = 1,
                                   default_tac: int = 1,
                                   default_freq: int = 2100,
                                   azimuth_step: int = 120) -> pd.DataFrame:
        """
        Generates a DataFrame with dummy topology data.
        """
        logger.info(f"Generating dummy topology for {num_sites} sites with {cells_per_site} cells each.")
        topology_data = []
        current_ecgi = start_ecgi
        current_enodeb_id = start_enodeb_id

        COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
        COL_CELL_AZ_DEG = getattr(c, 'CELL_AZ_DEG', 'cell_az_deg')
        COL_ECGI = getattr(c, 'ECGI', 'ecgi')
        COL_SITE_ID = getattr(c, 'SITE_ID', 'site_id')
        COL_CELL_NAME = getattr(c, 'CELL_NAME', 'cell_name')
        COL_ENODEB_ID = getattr(c, 'ENODEB_ID', 'enodeb_id')
        COL_TAC = getattr(c, 'TAC', 'tac')
        COL_CELL_CARRIER_FREQ_MHZ = getattr(c, 'CELL_CARRIER_FREQ_MHZ', 'cell_carrier_freq_mhz')

        for i in range(num_sites):
            site_lat = np.random.uniform(lat_range[0], lat_range[1])
            site_lon = np.random.uniform(lon_range[0], lon_range[1])
            site_id_str = f"Site{i+1}"

            for j in range(cells_per_site):
                cell_az = (j * azimuth_step) % 360
                # Consistent cell_id naming convention
                cell_id_str = f"cell_{current_enodeb_id}_{j}" # Use sector index j
                cell_name_str = f"{site_id_str}_Cell{j+1}"

                row = {
                    COL_ECGI: current_ecgi,
                    COL_SITE_ID: site_id_str,
                    COL_CELL_NAME: cell_name_str,
                    COL_ENODEB_ID: current_enodeb_id,
                    COL_CELL_AZ_DEG: cell_az,
                    COL_TAC: default_tac,
                    COL_CELL_LAT: site_lat, # All cells at a site share same lat/lon
                    COL_CELL_LON: site_lon,
                    COL_CELL_ID: cell_id_str,
                    COL_CELL_CARRIER_FREQ_MHZ: default_freq,
                }
                topology_data.append(row)
            current_ecgi += cells_per_site # Increment ECGI per site or per cell as needed
            current_enodeb_id += 1

        df = pd.DataFrame(topology_data)
        column_order = [ 
            COL_ECGI, COL_SITE_ID, COL_CELL_NAME, COL_ENODEB_ID, COL_CELL_AZ_DEG, COL_TAC,
            COL_CELL_LAT, COL_CELL_LON, COL_CELL_ID, COL_CELL_CARRIER_FREQ_MHZ
        ]
        df = df.reindex(columns=column_order)
        logger.info(f"Generated dummy topology DataFrame with {len(df)} cells.")
        return df

    def _generate_initial_config_df(self,
                                   topology_df: pd.DataFrame,
                                   default_config_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generates an initial configuration DataFrame for cells in the topology.
        """
        COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        if topology_df is None or topology_df.empty:
            raise ValueError("Topology DataFrame must be provided to generate config.")
        if COL_CELL_ID not in topology_df.columns:
            raise ValueError(f"Topology DataFrame must contain the column '{COL_CELL_ID}'.")

        logger.info(f"Generating initial config with params: {default_config_params}.")
        config_df = topology_df[[COL_CELL_ID]].drop_duplicates().reset_index(drop=True)

        for param_name, default_value in default_config_params.items():
            config_df[param_name] = default_value
        
        logger.info(f"Generated initial config DataFrame with {len(config_df)} rows.")
        return config_df

    def generate_topology_and_config_files(self,
                                           num_sites: int,
                                           cells_per_site: int,
                                           lat_range: Tuple[float, float],
                                           lon_range: Tuple[float, float],
                                           azimuth_step: int = 120,
                                           default_config_params: Dict[str, Any] = None,
                                           output_topology_path: str = "topology.csv",
                                           output_config_path: str = "config.csv"
                                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates and saves topology and initial configuration files.

        Args:
            num_sites: Number of cell sites.
            cells_per_site: Number of cells per site.
            lat_range: Latitude range for site generation.
            lon_range: Longitude range for site generation.
            default_power_dbm: Default cell transmission power.
            azimuth_step: Azimuth separation between cells at a site.
            default_config_params: Dict of parameters and their default values for config.csv
                                   (e.g., {'cell_el_deg': 12.0}).
            output_topology_path: Path to save topology.csv.
            output_config_path: Path to save config.csv.

        Returns:
            A tuple of (topology_df, initial_config_df).
        """
        if default_config_params is None:
            default_config_params = {getattr(c, 'CELL_EL_DEG', 'cell_el_deg'): 12.0}

        topology_df = self._generate_dummy_topology_df(
            num_sites=num_sites, cells_per_site=cells_per_site,
            lat_range=lat_range, lon_range=lon_range,
            azimuth_step=azimuth_step
        )
        initial_config_df = self._generate_initial_config_df(topology_df, default_config_params)

        try:
            os.makedirs(os.path.dirname(output_topology_path) or '.', exist_ok=True)
            topology_df.to_csv(output_topology_path, index=False)
            logger.info(f"Saved generated topology to {output_topology_path}")
        except Exception as e:
            logger.error(f"Could not save generated topology to {output_topology_path}: {e}")

        try:
            os.makedirs(os.path.dirname(output_config_path) or '.', exist_ok=True)
            initial_config_df.to_csv(output_config_path, index=False)
            logger.info(f"Saved generated initial config to {output_config_path}")
        except Exception as e:
            logger.error(f"Could not save generated config to {output_config_path}: {e}")

        return topology_df, initial_config_df

    def generate_dummy_training_data(self,
                                     topology_df: pd.DataFrame,
                                     ue_data_all_ticks: pd.DataFrame, # Combined DataFrame of UEs from all ticks
                                     num_training_samples: int = 12000,
                                     possible_tilts: List[float] = None,
                                     assumed_optimal_tilt: float = 8.0,
                                     tilt_penalty_factor: float = 0.5, # dB penalty per degree of deviation
                                     output_training_data_path: str = "dummy_ue_training_data.csv"
                                     ) -> pd.DataFrame:
        """
        Generates a DUMMY training dataset for pipeline testing.
        The RSRP values are highly simplified and NOT physically accurate regarding tilt.

        Args:
            topology_df: DataFrame containing cell information (must include c.CELL_ID,
                         c.CELL_LAT, c.CELL_LON, c.CELL_TXPWR_DBM).
            ue_data_all_ticks: A single DataFrame containing UE data from all ticks
                               (must include c.LAT, c.LON, 'ue_id', 'tick').
            num_training_samples: Approximate target number of rows in the output.
            possible_tilts: List of tilt values to sample from. Defaults to [0..20] deg.
            assumed_optimal_tilt: Reference tilt for the crude RSRP penalty.
            tilt_penalty_factor: Penalty (in dB) applied to RSRP per degree of deviation
                                 from assumed_optimal_tilt.
            output_training_data_path: Path to save the dummy training data CSV.

        Returns:
            A Pandas DataFrame with dummy training data.
        """
        logger.warning("Generating DUMMY training data. RSRP values DO NOT accurately reflect tilt effects.")
        logger.warning("An internal assumed TxPower of 25 dBm is used for dummy RSRP calculation.")
        if possible_tilts is None:
            possible_tilts = list(np.arange(0.0, 21.0, 1.0)) 

        COL_CELL_ID = getattr(c, 'CELL_ID', 'cell_id')
        COL_CELL_EL_DEG = getattr(c, 'CELL_EL_DEG', 'cell_el_deg')
        COL_LAT = getattr(c, 'LAT', 'lat')
        COL_LON = getattr(c, 'LON', 'lon')
        COL_CELL_LAT = getattr(c, 'CELL_LAT', 'cell_lat')
        COL_CELL_LON = getattr(c, 'CELL_LON', 'cell_lon')
        
        required_topo_cols = [COL_CELL_ID, COL_CELL_LAT, COL_CELL_LON]
        if not all(col in topology_df.columns for col in required_topo_cols):
            missing = [col for col in required_topo_cols if col not in topology_df.columns]
            raise ValueError(f"Topology DF missing required columns for dummy training data: {missing}")

        required_ue_cols = [COL_LAT, COL_LON, 'ue_id', 'tick']
        if not all(col in ue_data_all_ticks.columns for col in required_ue_cols):
            missing = [col for col in required_ue_cols if col not in ue_data_all_ticks.columns]
            raise ValueError(f"UE data DF missing required columns for dummy training data: {missing}")

        dummy_training_data_list = []
        # Assume a fixed Tx Power for this dummy RSRP calculation, as it's no longer in topology.csv
        assumed_cell_txpwr_val = 25.0 # dBm 
        ref_rx_power_at_1m_eff = assumed_cell_txpwr_val - 32.45 # Effective Pr at 1m assuming free space (FSPL at 1m for ~2GHz is ~32.45dB, so Pr = Pt-FSPL)
                                                        # Or, more simply, use an arbitrary reference for Pr at 1m
        ref_rx_power = -50 # Reference Rx power at a reference distance (e.g. 1m) if Pt was part of it.
                           # For a simpler model: RSRP ~ C - 10*n*log10(d). C includes TxPower & antenna gains.
                           # Let's use the previous simple model and just use the assumed_cell_txpwr_val where needed.

        path_loss_exponent = 3.5

        try:
            if ue_data_all_ticks.empty:
                logger.warning("Cannot generate dummy training data: Input UE data is empty.")
                return pd.DataFrame()

            num_cells = len(topology_df[COL_CELL_ID].unique())
            if num_cells == 0:
                raise ValueError("No cells found in topology_df for dummy training data generation.")

            # Sampling strategy: sample UE-tick pairs, then cross with all cells
            num_unique_ue_tick_pairs_to_sample = max(1, int(round(num_training_samples / num_cells)))
            
            num_available_ue_tick_pairs = len(ue_data_all_ticks)
            if num_unique_ue_tick_pairs_to_sample > num_available_ue_tick_pairs:
                logger.warning(f"Requested {num_unique_ue_tick_pairs_to_sample} UE-tick samples, but only {num_available_ue_tick_pairs} available. Using all.")
                num_unique_ue_tick_pairs_to_sample = num_available_ue_tick_pairs
            
            if num_unique_ue_tick_pairs_to_sample == 0:
                logger.warning("No UE-tick pairs to sample. Dummy training data will be empty.")
                return pd.DataFrame()

            sampled_ue_data = ue_data_all_ticks.sample(n=num_unique_ue_tick_pairs_to_sample, replace=False, random_state=42)
            logger.info(f"Generating dummy training data from {len(sampled_ue_data)} sampled UE-tick pairs, crossed with {num_cells} cells.")

            for _, ue_row in sampled_ue_data.iterrows():
                ue_lat_val = ue_row[COL_LAT]
                ue_lon_val = ue_row[COL_LON]

                for _, cell_row in topology_df.iterrows():
                    cell_id_val = cell_row[COL_CELL_ID]
                    cell_lat_val = cell_row[COL_CELL_LAT]
                    cell_lon_val = cell_row[COL_CELL_LON]

                    try:
                        dist_km = GISTools.dist((ue_lat_val, ue_lon_val), (cell_lat_val, cell_lon_val))
                        dist_m = dist_km * 1000.0
                        # Simplified RSRP: Use an arbitrary reference point that implies TxPower.
                        # Let's use 'ref_rx_power = -50' as power at some reference distance (e.g. 10m)
                        # and scale from there, or more directly: RSRP = TxPower_assumed - PathLoss
                        # PathLoss = K + 10*n*log10(d_meters)
                        # For this dummy data, we'll stick to the simple previous model structure, using the assumed TxPower.
                        # Arbitrary reference point for calculation start (e.g. Effective power after some initial loss)
                        effective_start_power = assumed_cell_txpwr_val - 40 # Example: TxPower minus some antenna gain and fixed losses
                        simple_rsrp = effective_start_power - 10 * path_loss_exponent * np.log10(dist_m) if dist_m > 1.0 else assumed_cell_txpwr_val - 40 # Avoid log(<=0) or very high RSRP at <1m
                    except Exception: 
                        simple_rsrp = assumed_cell_txpwr_val - 140 # Heavily penalize if distance calc fails

                    random_tilt = np.random.choice(possible_tilts)
                    tilt_deviation = abs(random_tilt - assumed_optimal_tilt)
                    rsrp_penalty = tilt_deviation * tilt_penalty_factor
                    adjusted_rsrp = simple_rsrp - rsrp_penalty

                    dummy_training_data_list.append({
                        COL_CELL_ID: cell_id_val,
                        "avg_rsrp": adjusted_rsrp, 
                        COL_LON: ue_lon_val,
                        COL_LAT: ue_lat_val,
                        COL_CELL_EL_DEG: random_tilt
                    })
            
            final_dummy_training_df = pd.DataFrame()
            if dummy_training_data_list:
                final_dummy_training_df = pd.DataFrame(dummy_training_data_list)
                output_cols = [COL_CELL_ID, "avg_rsrp", COL_LON, COL_LAT, COL_CELL_EL_DEG]
                final_dummy_training_df = final_dummy_training_df[output_cols]
            
            logger.info(f"Generated dummy training data with {len(final_dummy_training_df)} rows.")

            if not final_dummy_training_df.empty:
                try:
                    os.makedirs(os.path.dirname(output_training_data_path) or '.', exist_ok=True)
                    final_dummy_training_df.to_csv(output_training_data_path, index=False)
                    logger.info(f"Saved dummy training data to {output_training_data_path}")
                except Exception as e:
                    logger.error(f"Could not save dummy training data to {output_training_data_path}: {e}")
            
            return final_dummy_training_df

        except Exception as e:
            logger.error(f"Failed to generate dummy training data: {e}", exc_info=True)
            return pd.DataFrame()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running basic ScenarioConfigurationGenerator example...")

    generator = ScenarioConfigurationGenerator()

    # Example: Generate topology and config
    example_topo_df, example_config_df = generator.generate_topology_and_config_files(
        num_sites=2,
        cells_per_site=2,
        lat_range=(40.0, 40.1),
        lon_range=(-74.0, -73.9),
        output_topology_path="./example_topology.csv",
        output_config_path="./example_config.csv"
    )
    print("\nGenerated Topology (first 5 rows):")
    print(example_topo_df.head())
    print("\nGenerated Config (first 5 rows):")
    print(example_config_df.head())

    # Example: Generate dummy training data (requires UE data)
    # Create some dummy UE data for the example
    example_ue_data = {
        getattr(c, 'LAT', 'lat'): np.random.uniform(40.0, 40.1, 20),
        getattr(c, 'LON', 'lon'): np.random.uniform(-74.0, -73.9, 20),
        'ue_id': np.arange(20),
        'tick': np.random.randint(0, 2, 20)
    }
    example_ue_df_all_ticks = pd.DataFrame(example_ue_data)

    if not example_topo_df.empty and not example_ue_df_all_ticks.empty:
        dummy_train_df = generator.generate_dummy_training_data(
            topology_df=example_topo_df,
            ue_data_all_ticks=example_ue_df_all_ticks,
            num_training_samples=50, # Small sample for example
            output_training_data_path="./example_dummy_training_data.csv"
        )
        print("\nGenerated Dummy Training Data (first 5 rows):")
        print(dummy_train_df.head())
    else:
        print("\nSkipping dummy training data generation due to missing topology or UE data.")
