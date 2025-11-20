# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Default logic for cell selection, camping and attachment.
"""

from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from radp.digital_twin.utils import constants


def perform_attachment(
    ue_prediction_data: pd.DataFrame,
    topology: pd.DataFrame,
) -> pd.DataFrame:
    """This method looks at Rx power (predictions) from cells to pixels,
    for a given snapshot in time (e.g. simulation tick),
    and performs cell selection (camping or attachment).

    `ue_prediction_data` is a dataframe which contains at least the
    following columns:
        lon,            # pixel longitude or similar
        lat,            # pixel latitude or similar
        cell_id,        # cell ID for a cell
        rxpower_dbm,    # (predicted) Rx power from given cell to given location

    `topology` is a dataframe which contains at least the following columns:
        cell_id:                # cell ID for a cell
        cell_carrier_freq_mhz:  # carrier frequency (in MHz) for given cell

    This method returns a single dataframe back that represents
    the result of cell selection. The following columns are present in the result :
        loc_x,      # pixel longitude or similar
        loc_y,      # pixel latitude or similar
        cell_id,    # cell ID of selected cell
        rsrp_dbm,   # (predicted) Rx power of selected cell
        sinr_db,    # (predicted) SINR, that accounts for
                    # interference from other cells on the same layer (carrier frequency)
    """

    # initiate a dictionary to store power-by-layer dictionaries on a per-pixel basis
    rx_powers_by_layer_by_loc: Dict[Tuple[float, float], Dict[float, List[Tuple[Any, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # pull per-cell frequencies for faster lookup
    cell_id_to_freq = {row.cell_id: row.cell_carrier_freq_mhz for _, row in topology.iterrows()}

    # iterate over ue_prediction_data, to
    # build rx_powers_by_layer_by_loc map
    for _, row in ue_prediction_data.iterrows():
        # pull cell carrier frequency
        cell_carrier_freq_mhz = cell_id_to_freq[row.cell_id]

        # fetch the pixel longitude and latitude
        loc_x = row.get(constants.LOC_X, row.get(constants.LON))
        loc_y = row.get(constants.LOC_Y, row.get(constants.LAT))
        if loc_x is None or loc_y is None:
            raise Exception("loc_x or loc_y cannot be found in the dataset")

        # add (cell_id, rxpower) tuple on a per-row, per-freq basis
        rx_powers_by_layer_by_loc[(loc_x, loc_y)][cell_carrier_freq_mhz].append((row.cell_id, row.rxpower_dbm))

    # perform cell selection per location
    rf_dataframe_dict = defaultdict(list)

    for loc, rx_powers_by_layer in rx_powers_by_layer_by_loc.items():
        # compute strongest server, interference and SINR
        rsrp_dbm_by_layer, sinr_db_by_layer = get_rsrp_dbm_sinr_db_by_layer(rx_powers_by_layer)

        # pull sinr_db, cell_id and rsrp_dbm based on highest SINR
        max_sinr_db_item = max(sinr_db_by_layer.items(), key=lambda k: k[1][1])
        max_sinr_db_cell_id, max_sinr_db = max_sinr_db_item[1]
        rsrp_dbm = next(v[1] for v in rsrp_dbm_by_layer.values() if v[0] == max_sinr_db_cell_id)

        # update rf_dataframe output
        rf_dataframe_dict[constants.LOC_X].append(loc[0])
        rf_dataframe_dict[constants.LOC_Y].append(loc[1])
        rf_dataframe_dict[constants.CELL_ID].append(max_sinr_db_cell_id)
        rf_dataframe_dict[constants.SINR_DB].append(max_sinr_db)
        rf_dataframe_dict[constants.RSRP_DBM].append(rsrp_dbm)

    # return as dataframe
    return pd.DataFrame(rf_dataframe_dict)


def get_rsrp_dbm_sinr_db_by_layer(
    rx_powers_by_layer: Dict[float, List[Tuple[str, float]]],
) -> Tuple[Dict[float, Tuple[str, float]], Dict[float, Tuple[str, float]]]:
    """Given (predicted) Rx powers per layer, determine
    RSRP (served by max power) and SINR, within layer.
    """

    rsrp_dbm_by_layer: Dict[float, Tuple[str, float]] = {}
    sinr_db_by_layer: Dict[float, Tuple[str, float]] = {}

    # for each layer, compute strongest server, interference and SINR
    for cell_carrier_freq_mhz, rx_powers in rx_powers_by_layer.items():
        # get the max rsrp from the layer
        rsrp_dbm_by_layer[cell_carrier_freq_mhz] = max(rx_powers, key=itemgetter(1))

        # calculate background noise
        pred_noise = 10 ** (constants.LATENT_BACKGROUND_NOISE_DB / 10)

        # calculate prediction interference by subtracting
        # the strongest rx power from total sum of rx powers
        pred_interference = sum(10 ** (p / 10) for i, p in rx_powers) - (
            10 ** (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] / 10)
        )
        pred_interference_noise_dBm = 10 * np.log10(pred_interference + pred_noise)
        sinr_db_by_layer[cell_carrier_freq_mhz] = (
            rsrp_dbm_by_layer[cell_carrier_freq_mhz][0],
            (rsrp_dbm_by_layer[cell_carrier_freq_mhz][1] - pred_interference_noise_dBm),
        )

    return rsrp_dbm_by_layer, sinr_db_by_layer


def perform_attachment_hyst_ttt(ue_data: pd.DataFrame, hyst: float, ttt: int, rlf_threshold: float) -> pd.DataFrame:
    """
    Performs UE-to-cell attachment across all simulation ticks using hysteresis,
    time-to-trigger, and radio link failure threshold mechanisms to simulate
    realistic cellular handover behavior. Uses a NumPy-accelerated path for
    performance, while preserving the original behavior and output schema of
    the prior pandas implementation.

    This function implements a multi-stage cell selection algorithm:
    1. Hysteresis: Prevents frequent handovers by requiring new cell signal to exceed
       current cell signal by a threshold (hyst) in dB
    2. Time-to-Trigger (TTT): Requires consistent measurements over a time window
       before executing a handover, reducing ping-pong effects
    3. Radio Link Failure (RLF): Marks UEs as "RLF" when SINR falls below threshold
       and no suitable alternative cell is available

    Parameters:
        ue_data (pd.DataFrame): UE measurement data containing:
            - tick (int): Simulation time step
            - ue_id: User equipment identifier
            - cell_id: Cell tower identifier
            - cell_rxpower_dbm (float): Received power from cell in dBm
            - sinr_db (float): Signal-to-interference-plus-noise ratio in dB

        hyst (float): Hysteresis threshold in dB. A candidate cell must exceed
            the current serving cell's power by at least this amount to trigger
            consideration for handover.

        ttt (int): Time-to-trigger window size in ticks. The number of consecutive
            measurements required to confirm a handover decision.

        rlf_threshold (float): Minimum SINR in dB required to maintain a connection.
            UEs with SINR below this threshold will either handover to a better cell
            or be marked as "RLF" (Radio Link Failure) if no suitable cell exists.

    Returns:
        pd.DataFrame: Filtered UE-cell attachment states for all ticks, containing
            the same columns as the input, but only rows representing the actual
            UE-to-cell attachments after applying hysteresis, TTT, and RLF logic.
            Rows marked with cell_id="RLF" indicate radio link failures where
            cell_rxpower_dbm and sinr_db are set to -inf.
    """
    # Fast path: use NumPy helpers and convert back preserving columns/dtypes
    matrices, mappings = _np_preprocess_to_matrices(ue_data)
    final_attachments, pre_rlf_attachments = _np_process_all_ticks(matrices, hyst, ttt, rlf_threshold)
    result = _np_convert_results_to_dataframe(final_attachments, pre_rlf_attachments, mappings, ue_data)
    return result


def find_hyst_diff(df2: pd.DataFrame) -> float:
    """
    Finds the highest difference in the 'cell_rxpower_dbm' column
    by calculating the absolute difference between consecutive rows.

    Parameters:
        df2 (pd.DataFrame): Input DataFrame containing the 'cell_rxpower_dbm' column.

    Returns:
        float: The maximum absolute difference between consecutive values
    """
    # Make a copy of the dataframe to avoid modifying the original one
    df = df2.copy()

    # Replace infinite values with NaN in 'cell_rxpower_dbm'
    df["cell_rxpower_dbm"] = df["cell_rxpower_dbm"].replace([np.inf, -np.inf], np.nan)

    # Drop rows where 'cell_rxpower_dbm' is NaN
    df_clean = df.dropna(subset=["cell_rxpower_dbm"]).copy()

    # Calculate the difference between the maximum and minimum values in the cleaned data
    max_val = df_clean["cell_rxpower_dbm"].max()
    min_val = df_clean["cell_rxpower_dbm"].min()

    # Return the difference between the max and min values
    return max_val - min_val


def _np_preprocess_to_matrices(
    ue_data: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Convert pandas DataFrame to compact NumPy matrices.

    Matrices:
      - power[t,u,c]: float32 of cell_rxpower_dbm (or -inf)
      - sinr[t,u,c]: float32 of sinr_db (or -inf)
      - strongest_cells[t,u]: int32 of argmax over power

    Mappings include idx<->id maps and sizes.
    """
    # Create ID mappings for fast lookups (sorted for determinism)
    unique_ticks = sorted(ue_data["tick"].unique().tolist())
    unique_ues = sorted(ue_data["ue_id"].unique().tolist())
    unique_cells = sorted(ue_data["cell_id"].unique().tolist())

    n_ticks = len(unique_ticks)
    n_ues = len(unique_ues)
    n_cells = len(unique_cells)

    tick_to_idx = {tick: i for i, tick in enumerate(unique_ticks)}
    ue_to_idx = {ue: i for i, ue in enumerate(unique_ues)}
    cell_to_idx = {cell: i for i, cell in enumerate(unique_cells)}

    power = np.full((n_ticks, n_ues, n_cells), -np.inf, dtype=np.float32)
    sinr = np.full((n_ticks, n_ues, n_cells), -np.inf, dtype=np.float32)

    # Fill matrices
    for _, row in ue_data.iterrows():
        t = tick_to_idx[row["tick"]]
        u = ue_to_idx[row["ue_id"]]
        c = cell_to_idx[row["cell_id"]]
        power[t, u, c] = row["cell_rxpower_dbm"]
        # Allow datasets without sinr_db column by defaulting to -inf
        if "sinr_db" in ue_data.columns:
            sinr[t, u, c] = row["sinr_db"]

    strongest_cells = np.argmax(power, axis=2).astype(np.int32)

    matrices: Dict[str, np.ndarray] = {
        "power": power,
        "sinr": sinr,
        "strongest_cells": strongest_cells,
    }

    mappings: Dict[str, Any] = {
        "tick_to_idx": tick_to_idx,
        "ue_to_idx": ue_to_idx,
        "cell_to_idx": cell_to_idx,
        "idx_to_tick": {i: t for t, i in tick_to_idx.items()},
        "idx_to_ue": {i: u for u, i in ue_to_idx.items()},
        "idx_to_cell": {i: c for c, i in cell_to_idx.items()},
        "n_ticks": n_ticks,
        "n_ues": n_ues,
        "n_cells": n_cells,
    }

    return matrices, mappings


def _np_process_all_ticks(
    matrices: Dict[str, np.ndarray],
    hyst: float,
    ttt: int,
    rlf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process all ticks using NumPy, mirroring pandas behavior.

    Returns (final_attachments, pre_rlf_attachments), each shaped (n_ticks, n_ues)
    with per-UE cell indices (>=0). final_attachments may contain -2 for RLF.
    """
    power = matrices["power"]
    strongest = matrices["strongest_cells"]

    n_ticks, n_ues, _ = power.shape

    attachments = np.full((n_ticks, n_ues), -1, dtype=np.int32)
    pre_rlf_attachments = np.full((n_ticks, n_ues), -1, dtype=np.int32)

    # History holds previous ttt-1 strongest entries; on decision tick, we evaluate
    # TTT consistency across [history (ttt-1), current-hyst].
    hist_len = max(0, ttt - 1)
    history_buffer = np.full((hist_len, n_ues), -1, dtype=np.int32)
    history_size = 0  # number of valid rows in history_buffer (<= hist_len)

    for tick in range(n_ticks):
        # Determine current strongest per UE
        current_strongest = strongest[tick]

        if tick < hist_len:
            # Warm-up phase: attach to strongest (then RLF check)
            selected = current_strongest.copy()
            # Store strongest in history for future TTT checks
            history_buffer[tick] = current_strongest
            history_size += 1

            # No hysteresis/TTT yet; preserve selected before RLF
            pre_rlf_attachments[tick] = selected
            final_selected = _np_apply_rlf_check(selected, tick, matrices, rlf_threshold)
            attachments[tick] = final_selected
            continue

        # Normal phase: apply hyst -> include in history -> TTT -> final hyst -> RLF
        past_attachment = attachments[tick - 1]

        # Step 1: hysteresis vs past cell using current tick powers
        hyst_result = _np_apply_hysteresis(tick, matrices, past_attachment, current_strongest, hyst)

        # Step 2: TTT consistency across last (ttt-1) strongest and current hyst result
        ttt_result = _np_apply_ttt_check(history_buffer[:history_size], hyst_result, past_attachment, tick, matrices)

        # Step 3: final hysteresis to avoid unnecessary handovers
        final_before_rlf = _np_apply_final_hysteresis(tick, matrices, ttt_result, past_attachment, hyst)

        # Save pre-RLF selection (for parity when marking RLF)
        pre_rlf_attachments[tick] = final_before_rlf

        # Step 4: RLF threshold handling (select best-SINR cell if needed; else mark RLF)
        final_selected = _np_apply_rlf_check(final_before_rlf, tick, matrices, rlf_threshold)
        attachments[tick] = final_selected

        # Update circular history: drop oldest, add current hyst entry
        if hist_len > 0:
            # Move up rows if buffer full; keep last hist_len-1 previous + current hyst
            if history_size < hist_len:
                # strictly this shouldn't happen here, but keep safe
                history_buffer[history_size] = hyst_result
                history_size += 1
            else:
                history_buffer[:-1] = history_buffer[1:]
                history_buffer[-1] = hyst_result

    return attachments, pre_rlf_attachments


def _np_apply_hysteresis(
    tick: int,
    matrices: Dict[str, np.ndarray],
    past_attachment: np.ndarray,
    current_strongest: np.ndarray,
    hyst: float,
) -> np.ndarray:
    """Vectorized hysteresis check vs past cell in current tick."""
    n_ues = current_strongest.shape[0]
    result = current_strongest.copy()

    ue_idx = np.arange(n_ues, dtype=np.int32)
    curr_powers = matrices["power"][tick, ue_idx, current_strongest]

    # past cell powers in current tick (where valid)
    past_valid = past_attachment >= 0
    past_powers = np.full(n_ues, -999.0, dtype=np.float32)
    if np.any(past_valid):
        past_powers[past_valid] = matrices["power"][tick, ue_idx[past_valid], past_attachment[past_valid]]

    # Apply hysteresis and fallback to past where it fails
    need_revert = (curr_powers < (past_powers + hyst)) & past_valid
    result[need_revert] = past_attachment[need_revert]
    return result


def _np_apply_ttt_check(
    history_buffer: np.ndarray,  # shape: (ttt-1, n_ues)
    current_hyst: np.ndarray,  # shape: (n_ues,)
    past_attachment: np.ndarray,  # shape: (n_ues,)
    tick: int,
    matrices: Dict[str, np.ndarray],
) -> np.ndarray:
    """Apply TTT using only the last (ttt-1) strongest cells.

    If the same cell persisted across the (ttt-1) history ticks, attach to
    that cell for the current tick; otherwise, keep the past attachment.
    Special case: if past is RLF (encoded as -2), use the strongest in current tick.
    """
    n_ues = current_hyst.shape[0]
    result = current_hyst.copy()

    if history_buffer.size == 0:
        # No history; keep current selection
        return result

    # Evaluate per-UE to keep logic identical to the original pandas helper
    for ue in range(n_ues):
        # Consider only the (ttt-1) historical strongest cells
        hist_cells = history_buffer[:, ue]
        hist_cells = hist_cells[hist_cells >= 0]

        if hist_cells.size == 0:
            # Fallback to past if present
            if past_attachment[ue] >= 0:
                result[ue] = past_attachment[ue]
            continue

        unique_cells = np.unique(hist_cells)
        if unique_cells.size == 1:
            consistent = unique_cells[0]
            # Ensure this cell actually exists in current tick
            if matrices["power"][tick, ue, consistent] > -np.inf:
                result[ue] = consistent
        else:
            past_cell = past_attachment[ue]
            if past_cell == -2:  # RLF
                # Use strongest cell in current tick
                result[ue] = matrices["strongest_cells"][tick, ue]
            elif past_cell >= 0:
                # Keep past if it exists in current tick
                if matrices["power"][tick, ue, past_cell] > -np.inf:
                    result[ue] = past_cell

    return result


def _np_apply_final_hysteresis(
    tick: int,
    matrices: Dict[str, np.ndarray],
    current_attachment: np.ndarray,
    past_attachment: np.ndarray,
    hyst: float,
) -> np.ndarray:
    """Final hysteresis check mirroring _check_hyst_in_current_tick."""
    result = current_attachment.copy()
    # Identify UEs that switched cells and have a valid past
    switched = (current_attachment != past_attachment) & (past_attachment >= 0)
    if not np.any(switched):
        return result

    ue_switched = np.where(switched)[0]
    for ue in ue_switched:
        curr_cell = current_attachment[ue]
        past_cell = past_attachment[ue]
        curr_power = matrices["power"][tick, ue, curr_cell]
        past_power = matrices["power"][tick, ue, past_cell]
        if curr_power < (past_power + hyst):
            result[ue] = past_cell
    return result


def _np_apply_rlf_check(
    attachment: np.ndarray,  # shape: (n_ues,)
    tick: int,
    matrices: Dict[str, np.ndarray],
    rlf_threshold: float,
) -> np.ndarray:
    """Apply RLF threshold, matching _check_rlf_threshold behavior.

    If UE's attached SINR < threshold, replace with the row from current tick
    with max SINR if that SINR >= threshold; else set to RLF (-2).
    """
    n_ues = attachment.shape[0]
    result = attachment.copy()

    ue_idx = np.arange(n_ues, dtype=np.int32)

    valid = attachment >= 0
    if np.any(valid):
        att_cells = attachment[valid]
        att_sinr = matrices["sinr"][tick, ue_idx[valid], att_cells]

        need_update = att_sinr < rlf_threshold
        if np.any(need_update):
            upd_ues = ue_idx[valid][need_update]
            # For each UE, pick the cell with maximum SINR in current tick
            for ue in upd_ues:
                ue_sinr_all = matrices["sinr"][tick, ue, :]
                # Find argmax SINR (ignoring -inf naturally)
                best_cell = int(np.argmax(ue_sinr_all))
                best_sinr = ue_sinr_all[best_cell]
                if best_sinr >= rlf_threshold:
                    result[ue] = best_cell
                else:
                    result[ue] = -2  # RLF

    return result


def _np_convert_results_to_dataframe(
    final_attachments: np.ndarray,
    pre_rlf_attachments: np.ndarray,
    mappings: Dict[str, Any],
    original_data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert attachment indices to a DataFrame preserving original columns/dtypes.

    For RLF entries, preserve the pre-RLF row and only override
    'cell_id', 'cell_rxpower_dbm', and 'sinr_db' to match pandas parity.
    """
    idx_to_tick = mappings["idx_to_tick"]
    idx_to_ue = mappings["idx_to_ue"]
    idx_to_cell = mappings["idx_to_cell"]

    n_ticks, n_ues = final_attachments.shape
    rows: List[Dict[str, Any]] = []

    # Build a quick lookup for original rows by (tick, ue, cell)
    # to avoid repeated filtering cost on large frames.
    # Group by tick for faster slicing
    grouped_by_tick = {t: df for t, df in original_data.groupby("tick", sort=False)}

    for t_idx in range(n_ticks):
        tick_val = idx_to_tick[t_idx]
        tick_df = grouped_by_tick.get(tick_val, pd.DataFrame())
        if tick_df.empty:
            continue

        for u_idx in range(n_ues):
            ue_val = idx_to_ue[u_idx]
            cell_idx = int(final_attachments[t_idx, u_idx])
            pre_cell_idx = int(pre_rlf_attachments[t_idx, u_idx])

            if cell_idx >= 0:
                # Use the exact row from original data (tick, ue, cell)
                cell_val = idx_to_cell[cell_idx]
                match = tick_df[(tick_df["ue_id"] == ue_val) & (tick_df["cell_id"] == cell_val)]
                if not match.empty:
                    rows.append(match.iloc[0].to_dict())
            elif cell_idx == -2:
                # RLF: take the pre-RLF row as base and override key fields
                if pre_cell_idx >= 0:
                    pre_cell_val = idx_to_cell[pre_cell_idx]
                    match = tick_df[(tick_df["ue_id"] == ue_val) & (tick_df["cell_id"] == pre_cell_val)]
                else:
                    # Fallback: any row for this (tick, ue)
                    match = tick_df[(tick_df["tick"] == tick_val) & (tick_df["ue_id"] == ue_val)]

                if not match.empty:
                    base = match.iloc[0].to_dict()
                    base["cell_id"] = "RLF"
                    # Update only if columns exist
                    if "cell_rxpower_dbm" in base:
                        base["cell_rxpower_dbm"] = -np.inf
                    if "sinr_db" in base:
                        base["sinr_db"] = -np.inf
                    rows.append(base)

    if not rows:
        return pd.DataFrame(columns=original_data.columns)

    result = pd.DataFrame(rows)
    return result
