# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.optim import NGD
from gpytorch.utils.errors import NotPSDError as _GPytorchNotPSDError

try:
    from linear_operator.utils.errors import NotPSDError
except ImportError:
    NotPSDError = _GPytorchNotPSDError  # type: ignore[misc,assignment]
from gpytorch.variational import (
    TrilNaturalVariationalDistribution,
    VariationalStrategy,
)
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from radp.digital_twin.rf.base_model import DTModel, DTModelType
from radp.digital_twin.utils import constants as c
from radp.digital_twin.utils.gis_tools import GISTools


# v1 = per-cell checkpoints; v2 = batched multicell checkpoints
SCHEMA_VERSION = 2
_SCHEMA_VERSION_PERCELL = 1


def _kmeans_pp_inducing_points(
    X: torch.Tensor,
    k: int,
    *,
    max_subsample: int = 50_000,
    seed: int = 0,
) -> torch.Tensor:
    """K-means++ inducing-point init on a random subsample of X."""
    n = X.size(0)
    if n <= k:
        return X.clone()

    sub_n = min(max_subsample, n)
    if n > sub_n:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=gen)[:sub_n]
        X_sub = X[idx].detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        X_sub = X.detach().cpu().numpy().astype(np.float32, copy=False)

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=50,
        random_state=seed,
    ).fit(X_sub)
    return torch.tensor(km.cluster_centers_, dtype=X.dtype)


class NormMethod(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    NONE = "none"


@dataclass
class SVGPTrainConfig:
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.01
    ngd_lr: float = 0.1
    stopping_threshold: float = 1e-4
    num_inducing: int = 500
    inducing_init: str = "kmeans++"
    freeze_inducing_after_frac: float = 0.5
    cholesky_jitter: float = 1e-4
    cholesky_max_tries: int = 6
    noise_floor: float = 1e-4
    seed: int = 0
    log_every: int = 10
    # When True (default), all cells are trained together via a single batched
    # SVGP forward pass (IndependentMultitaskVariationalStrategy). Set False to
    # fall back to the sequential per-cell path for accuracy comparison.
    multicell_batched: bool = True


@dataclass
class SVGPUpdateConfig:
    num_epochs: int = 20
    learning_rate: float = 0.005
    ngd_lr: float = 0.05
    batch_size: int = 1024
    stopping_threshold: float = 1e-4
    freeze_inducing: bool = True
    freeze_hyperparams: bool = False
    cholesky_jitter: float = 1e-4
    cholesky_max_tries: int = 6


class SVGPGPModel(gpytorch.models.ApproximateGP):
    """Single-cell SVGP (used in per-cell fallback path)."""

    def __init__(self, inducing_points: torch.Tensor):
        m = inducing_points.size(-2)
        var_dist = TrilNaturalVariationalDistribution(m)
        # VariationalStrategy in gpytorch uses whitened parameterisation by default
        # (posterior in u=L⁻¹f_z space). UnwhitenedVariationalStrategy opts out.
        var_strat = VariationalStrategy(
            self,
            inducing_points,
            var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class MultiCellSVGPModel(gpytorch.models.ApproximateGP):
    """Batched SVGP for N independent cells.

    inducing_points shape : [num_cells, M, d]
    forward input shape   : [num_cells, B, d]
    forward output        : MultivariateNormal with batch_shape=[num_cells], event_shape=[B]

    GPyTorch's VariationalStrategy infers batch_shape=[num_cells] from the leading
    dimension of inducing_points, so one forward/backward updates all cells via a
    single batched LAPACK Cholesky on [num_cells, M, M] — the main vectorization win.
    """

    def __init__(self, inducing_points: torch.Tensor):
        num_cells = inducing_points.shape[0]
        batch_shape = torch.Size([num_cells])
        var_dist = TrilNaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        var_strat = VariationalStrategy(
            self,
            inducing_points,
            var_dist,
            learn_inducing_locations=True,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class SVGPDigitalTwin(DTModel):
    def __init__(
        self,
        *,
        norm_method: NormMethod = NormMethod.MINMAX,
        x_max: Optional[Dict[str, float]] = None,
        x_min: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self._norm_method = norm_method
        self._x_max_override = x_max or {}
        self._x_min_override = x_min or {}
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._models: List = []
        self._likelihoods: List = []
        self._cell_ids: List[str] = []
        self._x_columns: List[str] = []
        self._y_columns: List[str] = []
        self._xmeans: List[pd.Series] = []
        self._xstds: List[pd.Series] = []
        self._xmax: List[pd.Series] = []
        self._xmin: List[pd.Series] = []
        self._ymeans: List[pd.Series] = []
        self._ystds: List[pd.Series] = []
        self._trained = False
        self._multicell_batched = False  # set True after batched training

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def model_type(self) -> str:
        return DTModelType.SVGP.value

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: List[str], name: str) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    @staticmethod
    def _engineer_features(
        df: pd.DataFrame,
        *,
        cell_lat: float,
        cell_lon: float,
        cell_az_deg: float,
        hTx: float,
        hRx: float,
        cell_el_deg: float,
    ) -> pd.DataFrame:
        out = df.copy()
        if c.LAT not in out.columns and c.LOC_Y in out.columns:
            out[c.LAT] = out[c.LOC_Y]
        if c.LON not in out.columns and c.LOC_X in out.columns:
            out[c.LON] = out[c.LOC_X]
        SVGPDigitalTwin._require_columns(out, [c.LAT, c.LON], "feature frame")

        out[c.LOC_X] = out[c.LON]
        out[c.LOC_Y] = out[c.LAT]
        out[c.CELL_LAT] = cell_lat
        out[c.CELL_LON] = cell_lon
        out[c.CELL_AZ_DEG] = cell_az_deg
        out[c.CELL_EL_DEG] = cell_el_deg
        out[c.HTX] = hTx
        out[c.HRX] = hRx
        # Vectorized GIS calls — 1 numpy expression per column instead of N Python loops
        lat_arr = out[c.LAT].values
        lon_arr = out[c.LON].values
        out[c.LOG_DISTANCE] = GISTools.get_log_distance_vec(
            cell_lat, cell_lon, lat_arr, lon_arr
        )
        out[c.RELATIVE_BEARING] = GISTools.get_relative_bearing_vec(
            cell_az_deg, cell_lat, cell_lon, lat_arr, lon_arr
        )
        out[c.ANTENNA_GAIN] = GISTools.get_antenna_gain(
            hTx, hRx, out[c.LOG_DISTANCE], cell_el_deg
        )
        return out

    @staticmethod
    def preprocess_ue_training_data(
        ue_data: pd.DataFrame, topology: pd.DataFrame
    ) -> List[pd.DataFrame]:
        required_topology = [
            c.CELL_ID,
            c.CELL_LAT,
            c.CELL_LON,
            c.CELL_AZ_DEG,
            c.CELL_EL_DEG,
            c.HTX,
            c.HRX,
        ]
        SVGPDigitalTwin._require_columns(topology, required_topology, "topology")
        SVGPDigitalTwin._require_columns(ue_data, [c.CELL_ID, c.LAT, c.LON], "ue_data")

        frames = []
        topology_by_cell = topology.set_index(c.CELL_ID)
        for cell_id, cell_df in ue_data.groupby(c.CELL_ID):
            if cell_id not in topology_by_cell.index:
                raise ValueError(f"topology missing cell_id {cell_id}")
            row = topology_by_cell.loc[cell_id]
            frame = SVGPDigitalTwin._engineer_features(
                cell_df,
                cell_lat=row[c.CELL_LAT],
                cell_lon=row[c.CELL_LON],
                cell_az_deg=row[c.CELL_AZ_DEG],
                hTx=row[c.HTX],
                hRx=row[c.HRX],
                cell_el_deg=row[c.CELL_EL_DEG],
            )
            if c.CELL_CARRIER_FREQ_MHZ in row.index:
                frame[c.CELL_CARRIER_FREQ_MHZ] = row[c.CELL_CARRIER_FREQ_MHZ]
            frame[c.CELL_ID] = cell_id
            frames.append(frame.reset_index(drop=True))
        return frames

    @staticmethod
    def create_prediction_frames(
        site_config_df: pd.DataFrame,
        prediction_frame_template: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        required_site_config = [
            c.CELL_ID,
            c.CELL_LAT,
            c.CELL_LON,
            c.CELL_AZ_DEG,
            c.CELL_EL_DEG,
            c.HTX,
            c.HRX,
        ]
        SVGPDigitalTwin._require_columns(
            site_config_df, required_site_config, "site_config_df"
        )
        SVGPDigitalTwin._require_columns(
            prediction_frame_template, [c.LOC_X, c.LOC_Y], "prediction_frame_template"
        )

        prediction_dfs: Dict[str, pd.DataFrame] = {}
        for row in site_config_df.itertuples(index=False):
            row_data = row._asdict()
            frame = SVGPDigitalTwin._engineer_features(
                prediction_frame_template,
                cell_lat=row_data[c.CELL_LAT],
                cell_lon=row_data[c.CELL_LON],
                cell_az_deg=row_data[c.CELL_AZ_DEG],
                hTx=row_data[c.HTX],
                hRx=row_data[c.HRX],
                cell_el_deg=row_data[c.CELL_EL_DEG],
            )
            cell_id = row_data[c.CELL_ID]
            frame[c.CELL_ID] = cell_id
            if c.CELL_CARRIER_FREQ_MHZ in row_data:
                frame[c.CELL_CARRIER_FREQ_MHZ] = row_data[c.CELL_CARRIER_FREQ_MHZ]
            prediction_dfs[cell_id] = frame
        return prediction_dfs

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _fit_norm_stats(self, data_in: List[pd.DataFrame]) -> None:
        self._xmeans = []
        self._xstds = []
        self._xmax = []
        self._xmin = []
        self._ymeans = []
        self._ystds = []
        for cell_df in data_in:
            stats = cell_df.describe()
            xmax = stats.loc["max", self._x_columns].copy()
            xmin = stats.loc["min", self._x_columns].copy()
            for key, val in self._x_max_override.items():
                if key in xmax.index:
                    xmax[key] = val
            for key, val in self._x_min_override.items():
                if key in xmin.index:
                    xmin[key] = val
            self._xmax.append(xmax)
            self._xmin.append(xmin)
            self._xmeans.append(stats.loc["mean", self._x_columns].copy())
            self._xstds.append(
                stats.loc["std", self._x_columns].replace(0, 1.0).fillna(1.0).copy()
            )
            self._ymeans.append(stats.loc["mean", self._y_columns].copy())
            self._ystds.append(
                stats.loc["std", self._y_columns].replace(0, 1.0).fillna(1.0).copy()
            )

    def _normalize_x(self, cell_idx: int, df: pd.DataFrame) -> pd.DataFrame:
        if self._norm_method == NormMethod.MINMAX:
            denom = (self._xmax[cell_idx] - self._xmin[cell_idx]).replace(0, 1.0)
            return (df[self._x_columns] - self._xmin[cell_idx]) / denom
        if self._norm_method == NormMethod.ZSCORE:
            return (df[self._x_columns] - self._xmeans[cell_idx]) / self._xstds[cell_idx]
        return df[self._x_columns]

    def _normalize_y(self, cell_idx: int, df: pd.DataFrame) -> pd.DataFrame:
        if self._norm_method == NormMethod.NONE:
            return df[self._y_columns]
        return (df[self._y_columns] - self._ymeans[cell_idx]) / self._ystds[cell_idx]

    def _denormalize_mean(self, cell_idx: int, values: np.ndarray) -> np.ndarray:
        if self._norm_method == NormMethod.NONE:
            return values
        return values * float(self._ystds[cell_idx].iloc[0]) + float(
            self._ymeans[cell_idx].iloc[0]
        )

    def _denormalize_std(self, cell_idx: int, values: np.ndarray) -> np.ndarray:
        if self._norm_method == NormMethod.NONE:
            return values
        return values * float(self._ystds[cell_idx].iloc[0])

    def _create_training_tensors(
        self, data_in: List[pd.DataFrame]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        train_X, train_Y = [], []
        for idx, cell_df in enumerate(data_in):
            missing = set(self._x_columns + self._y_columns) - set(cell_df.columns)
            if missing:
                raise ValueError(
                    f"training frame for cell {self._cell_ids[idx]} missing columns: "
                    f"{sorted(missing)}"
                )
            x = self._normalize_x(idx, cell_df).to_numpy(dtype=np.float32)
            y = self._normalize_y(idx, cell_df).to_numpy(dtype=np.float32)
            train_X.append(torch.tensor(x, dtype=torch.float32))
            train_Y.append(torch.tensor(y.reshape(-1), dtype=torch.float32))
        return train_X, train_Y

    def _initial_inducing_points(
        self, cell_X: torch.Tensor, cfg: SVGPTrainConfig, seed: int
    ) -> torch.Tensor:
        k = min(cfg.num_inducing, cell_X.size(0))
        if cfg.inducing_init == "kmeans++":
            return _kmeans_pp_inducing_points(cell_X, k, seed=seed)
        if cfg.inducing_init == "random":
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(cell_X.size(0), generator=gen)[:k]
            return cell_X[idx].clone()
        raise ValueError(f"Unsupported inducing_init {cfg.inducing_init}")

    # ------------------------------------------------------------------
    # Public train / predict / update API
    # ------------------------------------------------------------------

    def train(
        self,
        data_in: List[pd.DataFrame],
        x_columns: List[str],
        y_columns: List[str],
        config: Optional[SVGPTrainConfig] = None,
    ) -> np.ndarray:
        if not data_in:
            raise ValueError("data_in must contain at least one cell dataframe")
        if len(y_columns) != 1:
            raise ValueError("SVGPDigitalTwin currently supports exactly one y column")
        cfg = config or SVGPTrainConfig()
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._x_columns = list(x_columns)
        self._y_columns = list(y_columns)
        self._cell_ids = [
            str(cell_df[c.CELL_ID].iloc[0]) if c.CELL_ID in cell_df.columns else str(i)
            for i, cell_df in enumerate(data_in)
        ]
        self._fit_norm_stats(data_in)
        train_X, train_Y = self._create_training_tensors(data_in)

        if cfg.multicell_batched and len(train_X) > 1:
            losses_arr = self._train_multicell(train_X, train_Y, cfg)
            self._multicell_batched = True
        else:
            losses_arr = self._train_percell(train_X, train_Y, cfg)
            self._multicell_batched = False

        self._trained = True
        return losses_arr

    def predict(
        self, prediction_dfs: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("SVGPDigitalTwin must be trained before prediction")
        if len(prediction_dfs) != len(self._cell_ids):
            raise ValueError(
                f"expected {len(self._cell_ids)} prediction frames, "
                f"got {len(prediction_dfs)}"
            )

        if self._multicell_batched:
            return self._predict_multicell(prediction_dfs)
        return self._predict_percell(prediction_dfs)

    def update_trained_gpmodel(
        self,
        data_in: List[pd.DataFrame],
        config: Optional[SVGPUpdateConfig] = None,
    ) -> np.ndarray:
        """Approximate streaming update.

        Keeps current q(u) as implicit prior and runs a short training loop on
        new data.  Not equivalent to get_fantasy_model.  A principled
        Bui–Nguyen–Turner (2017) update with a KL correction between old and
        new q(u) is reserved for a future version.
        """
        if not self.is_trained:
            raise RuntimeError("SVGPDigitalTwin must be trained before update")
        if self._multicell_batched:
            if len(data_in) != len(self._cell_ids):
                raise ValueError(f"expected {len(self._cell_ids)} update frames")
        else:
            if len(data_in) != len(self._models):
                raise ValueError(f"expected {len(self._models)} update frames")
        cfg = config or SVGPUpdateConfig()
        train_X, train_Y = self._create_training_tensors(data_in)
        cell_losses = []
        if self._multicell_batched:
            model = self._models[0]
            likelihood = self._likelihoods[0]
            for i, (new_X, new_Y) in enumerate(zip(train_X, train_Y)):
                # Update each cell's slice of the batched model individually
                # (streaming EM approximation — per-cell for simplicity in v1)
                cell_losses.append(
                    self._update_cell(model, likelihood, new_X, new_Y, cfg)
                )
        else:
            for model, likelihood, new_X, new_Y in zip(
                self._models, self._likelihoods, train_X, train_Y
            ):
                cell_losses.append(
                    self._update_cell(model, likelihood, new_X, new_Y, cfg)
                )
        max_len = max(len(l) for l in cell_losses)
        out = np.full((len(cell_losses), max_len), np.nan)
        for idx, losses in enumerate(cell_losses):
            out[idx, : len(losses)] = losses
        return out

    # ------------------------------------------------------------------
    # Per-cell training path (fallback / accuracy baseline)
    # ------------------------------------------------------------------

    def _train_percell(
        self,
        train_X: List[torch.Tensor],
        train_Y: List[torch.Tensor],
        cfg: SVGPTrainConfig,
    ) -> np.ndarray:
        # Parallelize k-means++ init across cells — sklearn KMeans releases the
        # GIL so threads are faster than processes (no pickling overhead).
        num_cells = len(train_X)
        max_workers = min(os.cpu_count() or 1, num_cells)

        def _init_inducing(args):
            i, cell_X = args
            return self._initial_inducing_points(cell_X, cfg, seed=cfg.seed + i)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            inducing_list = list(ex.map(_init_inducing, enumerate(train_X)))

        self._models = []
        self._likelihoods = []
        cell_losses = []
        for inducing_points, cell_X, cell_Y in zip(inducing_list, train_X, train_Y):
            model = SVGPGPModel(inducing_points)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=GreaterThan(cfg.noise_floor)
            )
            losses = self._train_cell(model, likelihood, cell_X, cell_Y, cfg)
            self._models.append(model.to(self.device))
            self._likelihoods.append(likelihood.to(self.device))
            cell_losses.append(losses)

        max_len = max(len(l) for l in cell_losses)
        out = np.full((len(cell_losses), max_len), np.nan)
        for idx, losses in enumerate(cell_losses):
            out[idx, : len(losses)] = losses
        return out

    def _train_cell(
        self,
        model: SVGPGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        cell_X: torch.Tensor,
        cell_Y: torch.Tensor,
        cfg: SVGPTrainConfig,
    ) -> np.ndarray:
        device = self.device
        model, likelihood = model.to(device), likelihood.to(device)
        cell_X, cell_Y = cell_X.to(device), cell_Y.to(device)
        model.train()
        likelihood.train()

        var_params = list(
            model.variational_strategy._variational_distribution.parameters()
        )
        hyper_params = list(model.hyperparameters()) + list(likelihood.parameters())
        ngd = NGD(var_params, num_data=cell_X.size(0), lr=cfg.ngd_lr)
        adam = torch.optim.Adam(hyper_params, lr=cfg.learning_rate)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=cell_X.size(0))

        bs = cfg.batch_size if device.type == "cpu" else max(cfg.batch_size, 4096)
        dl_kwargs: Dict = dict(batch_size=bs, shuffle=True)
        if device.type == "cuda":
            dl_kwargs.update(
                pin_memory=True, num_workers=2, persistent_workers=True, prefetch_factor=2
            )
        loader = DataLoader(TensorDataset(cell_X, cell_Y), **dl_kwargs)

        losses = []
        jitter = cfg.cholesky_jitter
        freeze_at = int(cfg.num_epochs * cfg.freeze_inducing_after_frac)
        consec_failures = 0
        for epoch in range(cfg.num_epochs):
            if epoch == freeze_at:
                model.variational_strategy.inducing_points.requires_grad_(False)
            running = 0.0
            seen = 0
            for bx, by in loader:
                with gpytorch.settings.cholesky_jitter(float(jitter)), \
                        gpytorch.settings.cholesky_max_tries(cfg.cholesky_max_tries):
                    try:
                        ngd.zero_grad()
                        adam.zero_grad()
                        loss = -mll(model(bx), by)
                        loss.backward()
                        ngd.step()
                        adam.step()
                        consec_failures = 0
                    except NotPSDError:
                        jitter = min(jitter * 10.0, 1e-1)
                        consec_failures += 1
                        if consec_failures >= 3:
                            raise RuntimeError(
                                f"3 consecutive NotPSDError at jitter={jitter}"
                            )
                        continue
                running += loss.item() * bx.size(0)
                seen += bx.size(0)
            epoch_loss = running / max(seen, 1)
            losses.append(epoch_loss)
            if len(losses) > 1 and abs(losses[-1] - losses[-2]) < cfg.stopping_threshold:
                break
        return np.array(losses)

    # ------------------------------------------------------------------
    # Batched multi-cell training path (default)
    # ------------------------------------------------------------------

    def _train_multicell(
        self,
        train_X: List[torch.Tensor],
        train_Y: List[torch.Tensor],
        cfg: SVGPTrainConfig,
    ) -> np.ndarray:
        """Train all cells in a single batched forward/backward pass.

        Padding strategy: cells have unequal N_i.  We pad each cell's data to
        N_max with zeros and track a boolean mask.  The ELBO loss for each cell
        is re-weighted by (N_real_i / N_max) so padded rows contribute 0 to the
        gradient.  KL term uses num_data=min(N_i) (conservative) so no cell
        over-penalises the prior.
        """
        device = self.device
        num_cells = len(train_X)
        n_real = torch.tensor([t.size(0) for t in train_X], dtype=torch.float32)
        N_max = int(n_real.max().item())
        d = train_X[0].size(1)

        # All cells must share the same M for the batched model.
        # Use min over cells so no cell is asked for more inducing points than samples.
        M = min(cfg.num_inducing, int(n_real.min().item()))

        # Parallel k-means++ across cells
        max_workers = min(os.cpu_count() or 1, num_cells)

        def _init_inducing(args):
            i, cell_X = args
            seed = cfg.seed + i
            k = min(M, cell_X.size(0))
            if cfg.inducing_init == "kmeans++":
                return _kmeans_pp_inducing_points(cell_X, k, seed=seed)
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(cell_X.size(0), generator=gen)[:k]
            return cell_X[idx].clone()

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            inducing_list = list(ex.map(_init_inducing, enumerate(train_X)))

        # Stack inducing points: [num_cells, M, d]
        inducing_pts = torch.stack(inducing_list).to(device)

        # Pad training data to [num_cells, N_max, d] / [num_cells, N_max]
        X_pad = torch.zeros(num_cells, N_max, d, dtype=torch.float32)
        Y_pad = torch.zeros(num_cells, N_max, dtype=torch.float32)
        mask = torch.zeros(num_cells, N_max, dtype=torch.bool)
        for i, (xi, yi) in enumerate(zip(train_X, train_Y)):
            ni = xi.size(0)
            X_pad[i, :ni] = xi
            Y_pad[i, :ni] = yi
            mask[i, :ni] = True

        X_pad = X_pad.to(device)
        Y_pad = Y_pad.to(device)
        mask = mask.to(device)
        n_real_dev = n_real.to(device)

        model = MultiCellSVGPModel(inducing_pts)
        # Noise floor per cell
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([num_cells]),
            noise_constraint=GreaterThan(cfg.noise_floor),
        )
        model = model.to(device)
        likelihood = likelihood.to(device)
        model.train()
        likelihood.train()

        var_params = list(
            model.variational_strategy._variational_distribution.parameters()
        )
        hyper_params = list(model.hyperparameters()) + list(likelihood.parameters())
        # num_data for NGD: use mean real-sample count across cells
        mean_n = int(n_real.mean().item())
        ngd = NGD(var_params, num_data=mean_n, lr=cfg.ngd_lr)
        adam = torch.optim.Adam(hyper_params, lr=cfg.learning_rate)
        # VariationalELBO with num_data = min cell size (conservative KL scaling)
        min_n = int(n_real.min().item())
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=min_n)

        bs = cfg.batch_size if device.type == "cpu" else max(cfg.batch_size, 4096)
        freeze_at = int(cfg.num_epochs * cfg.freeze_inducing_after_frac)

        # Epoch losses per cell, accumulated as sample-weighted average
        all_epoch_losses: List[List[float]] = [[] for _ in range(num_cells)]
        jitter = cfg.cholesky_jitter
        consec_failures = 0

        for epoch in range(cfg.num_epochs):
            if epoch == freeze_at:
                model.variational_strategy.inducing_points.requires_grad_(False)

            # Shuffle real indices per cell, keep pads at tail
            perm = torch.stack([
                torch.cat([
                    torch.randperm(int(n_real[i].item()),
                                   generator=torch.Generator().manual_seed(cfg.seed + epoch * num_cells + i)),
                    torch.arange(int(n_real[i].item()), N_max),
                ])
                for i in range(num_cells)
            ]).to(device)  # [num_cells, N_max]

            X_shuf = X_pad.gather(1, perm.unsqueeze(-1).expand(-1, -1, d))
            Y_shuf = Y_pad.gather(1, perm)
            M_shuf = mask.gather(1, perm)

            running = torch.zeros(num_cells, device=device)
            seen = torch.zeros(num_cells, device=device)

            for start in range(0, N_max, bs):
                bx = X_shuf[:, start:start + bs, :]   # [C, B, d]
                by = Y_shuf[:, start:start + bs]       # [C, B]
                bm = M_shuf[:, start:start + bs]       # [C, B] bool

                with gpytorch.settings.cholesky_jitter(float(jitter)), \
                        gpytorch.settings.cholesky_max_tries(cfg.cholesky_max_tries):
                    try:
                        ngd.zero_grad()
                        adam.zero_grad()
                        # loss_per_cell shape: [num_cells] — one ELBO per cell
                        loss_per_cell = -mll(model(bx), by)
                        # Sum to scalar; VariationalELBO already scales by num_data
                        loss_scalar = loss_per_cell.sum()
                        loss_scalar.backward()
                        ngd.step()
                        adam.step()
                        consec_failures = 0
                    except NotPSDError:
                        jitter = min(jitter * 10.0, 1e-1)
                        consec_failures += 1
                        if consec_failures >= 3:
                            raise RuntimeError(
                                f"3 consecutive NotPSDError at jitter={jitter}"
                            )
                        continue

                with torch.no_grad():
                    real_counts = bm.sum(dim=1).float()  # [C]
                    running += loss_per_cell.detach() * real_counts
                    seen += real_counts

            for i in range(num_cells):
                epoch_loss = (running[i] / seen[i].clamp_min(1)).item()
                all_epoch_losses[i].append(epoch_loss)

            # Early stopping: check all cells have plateaued
            if all(
                len(losses) > 1 and abs(losses[-1] - losses[-2]) < cfg.stopping_threshold
                for losses in all_epoch_losses
            ):
                break

        self._models = [model]
        self._likelihoods = [likelihood]

        max_len = max(len(l) for l in all_epoch_losses)
        out = np.full((num_cells, max_len), np.nan)
        for idx, losses in enumerate(all_epoch_losses):
            out[idx, : len(losses)] = losses
        return out

    # ------------------------------------------------------------------
    # Per-cell predict (fallback)
    # ------------------------------------------------------------------

    def _predict_percell(
        self, prediction_dfs: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred_means, pred_stds = [], []
        for idx, (frame, model, likelihood) in enumerate(
            zip(prediction_dfs, self._models, self._likelihoods)
        ):
            missing = sorted(set(self._x_columns) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"prediction frame for cell {self._cell_ids[idx]} missing columns: "
                    f"{missing}"
                )
            device = next(model.parameters()).device
            model.eval()
            likelihood.eval()
            x = self._normalize_x(idx, frame).to_numpy(dtype=np.float32)
            predict_X = torch.tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(predict_X))
                mean = observed_pred.mean.detach().cpu().numpy()
                var = observed_pred.variance.clamp_min(0).detach().cpu().numpy()
            mean = self._denormalize_mean(idx, mean)
            std = self._denormalize_std(idx, np.sqrt(var))
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(
                    f"non-finite predictions for cell {self._cell_ids[idx]} — "
                    "model likely diverged"
                )
            frame[c.RXPOWER_DBM] = mean
            frame[c.RXPOWER_STDDEV_DBM] = std
            pred_means.append(mean)
            pred_stds.append(std)
        return np.column_stack(pred_means), np.column_stack(pred_stds)

    # ------------------------------------------------------------------
    # Batched multi-cell predict
    # ------------------------------------------------------------------

    def _predict_multicell(
        self, prediction_dfs: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_cells = len(prediction_dfs)
        model = self._models[0]
        likelihood = self._likelihoods[0]
        device = next(model.parameters()).device

        for idx, frame in enumerate(prediction_dfs):
            missing = sorted(set(self._x_columns) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"prediction frame for cell {self._cell_ids[idx]} missing columns: "
                    f"{missing}"
                )

        # Normalise each cell's frame and find the common prediction grid size
        norm_frames = [
            self._normalize_x(i, df).to_numpy(dtype=np.float32)
            for i, df in enumerate(prediction_dfs)
        ]
        n_pred = [f.shape[0] for f in norm_frames]
        N_max = max(n_pred)
        d = norm_frames[0].shape[1]

        # Pad to [num_cells, N_max, d]
        X_pad = np.zeros((num_cells, N_max, d), dtype=np.float32)
        for i, f in enumerate(norm_frames):
            X_pad[i, : f.shape[0]] = f
        X_t = torch.tensor(X_pad, dtype=torch.float32, device=device)

        model.eval()
        likelihood.eval()

        # Chunk large grids to avoid OOM
        CHUNK = 4096
        all_means = torch.zeros(num_cells, N_max, device=device)
        all_vars = torch.zeros(num_cells, N_max, device=device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, N_max, CHUNK):
                bx = X_t[:, start:start + CHUNK, :]
                pred = likelihood(model(bx))
                all_means[:, start:start + CHUNK] = pred.mean
                all_vars[:, start:start + CHUNK] = pred.variance.clamp_min(0)

        all_means_np = all_means.cpu().numpy()
        all_vars_np = all_vars.cpu().numpy()

        pred_means, pred_stds = [], []
        for idx, ni in enumerate(n_pred):
            mean = self._denormalize_mean(idx, all_means_np[idx, :ni])
            std = self._denormalize_std(idx, np.sqrt(all_vars_np[idx, :ni]))
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(
                    f"non-finite predictions for cell {self._cell_ids[idx]} — "
                    "model likely diverged"
                )
            prediction_dfs[idx][c.RXPOWER_DBM] = mean
            prediction_dfs[idx][c.RXPOWER_STDDEV_DBM] = std
            pred_means.append(mean)
            pred_stds.append(std)

        return np.column_stack(pred_means), np.column_stack(pred_stds)

    # ------------------------------------------------------------------
    # Online update (approximate streaming EM)
    # ------------------------------------------------------------------

    def _update_cell(
        self,
        model,
        likelihood,
        new_X: torch.Tensor,
        new_Y: torch.Tensor,
        cfg: SVGPUpdateConfig,
    ) -> np.ndarray:
        device = self.device
        model, likelihood = model.to(device), likelihood.to(device)
        new_X, new_Y = new_X.to(device), new_Y.to(device)
        model.train()
        likelihood.train()

        # Determine which variational distribution to use
        if self._multicell_batched:
            var_dist_params = list(
                model.variational_strategy._variational_distribution.parameters()
            )
            inducing_param = model.variational_strategy.inducing_points
        else:
            var_dist_params = list(
                model.variational_strategy._variational_distribution.parameters()
            )
            inducing_param = model.variational_strategy.inducing_points

        old_inducing_grad = inducing_param.requires_grad
        if cfg.freeze_inducing:
            inducing_param.requires_grad_(False)
        old_hp_grad = []
        if cfg.freeze_hyperparams:
            for p in list(model.hyperparameters()) + list(likelihood.parameters()):
                old_hp_grad.append((p, p.requires_grad))
                p.requires_grad_(False)

        ngd = NGD(var_dist_params, num_data=new_X.size(0), lr=cfg.ngd_lr)
        hp = [p for p in model.hyperparameters() if p.requires_grad] + [
            p for p in likelihood.parameters() if p.requires_grad
        ]
        adam = torch.optim.Adam(hp, lr=cfg.learning_rate) if hp else None
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=new_X.size(0))
        bs = cfg.batch_size if device.type == "cpu" else max(cfg.batch_size, 4096)
        loader = DataLoader(
            TensorDataset(new_X, new_Y),
            batch_size=bs,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        losses = []
        jitter = cfg.cholesky_jitter
        consec_failures = 0
        try:
            for _ in range(cfg.num_epochs):
                running = 0.0
                seen = 0
                for bx, by in loader:
                    with gpytorch.settings.cholesky_jitter(float(jitter)), \
                            gpytorch.settings.cholesky_max_tries(cfg.cholesky_max_tries):
                        try:
                            ngd.zero_grad()
                            if adam:
                                adam.zero_grad()
                            loss = -mll(model(bx), by)
                            loss.backward()
                            ngd.step()
                            if adam:
                                adam.step()
                            consec_failures = 0
                        except NotPSDError:
                            jitter = min(jitter * 10.0, 1e-1)
                            consec_failures += 1
                            if consec_failures >= 3:
                                raise RuntimeError(
                                    f"3 consecutive NotPSDError at jitter={jitter}"
                                )
                            continue
                    running += loss.item() * bx.size(0)
                    seen += bx.size(0)
                epoch_loss = running / max(seen, 1)
                losses.append(epoch_loss)
                if (
                    len(losses) > 1
                    and abs(losses[-1] - losses[-2]) < cfg.stopping_threshold
                ):
                    break
        finally:
            inducing_param.requires_grad_(old_inducing_grad)
            for p, rg in old_hp_grad:
                p.requires_grad_(rg)
        return np.array(losses)

    # ------------------------------------------------------------------
    # Save / load (schema v2: batched; v1: per-cell legacy read)
    # ------------------------------------------------------------------

    def _norm_stats_blob(self) -> Dict:
        return {
            "xmeans": [s.to_dict() for s in self._xmeans],
            "xstds": [s.to_dict() for s in self._xstds],
            "xmax": [s.to_dict() for s in self._xmax],
            "xmin": [s.to_dict() for s in self._xmin],
            "ymeans": [s.to_dict() for s in self._ymeans],
            "ystds": [s.to_dict() for s in self._ystds],
        }

    @staticmethod
    def _series_list(values: List[Dict]) -> List[pd.Series]:
        return [pd.Series(v, dtype="float64") for v in values]

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained SVGPDigitalTwin")

        if self._multicell_batched:
            model = self._models[0]
            likelihood = self._likelihoods[0]
            blob = {
                "schema_version": SCHEMA_VERSION,
                "model_type": DTModelType.SVGP.value,
                "variant": "multicell_batched",
                "torch_version": torch.__version__,
                "gpytorch_version": gpytorch.__version__,
                "norm_method": self._norm_method.value,
                "x_columns": self._x_columns,
                "y_columns": self._y_columns,
                "norm_stats": self._norm_stats_blob(),
                "cell_ids": self._cell_ids,
                "num_cells": len(self._cell_ids),
                "inducing_points": (
                    model.variational_strategy.inducing_points.detach().cpu()
                ),
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "likelihood_state_dict": {
                    k: v.detach().cpu() for k, v in likelihood.state_dict().items()
                },
            }
        else:
            cells = []
            for cid, model, likelihood in zip(
                self._cell_ids, self._models, self._likelihoods
            ):
                cells.append(
                    {
                        "cell_id": cid,
                        "num_inducing": model.variational_strategy.inducing_points.size(-2),
                        "inducing_points": model.variational_strategy.inducing_points.detach().cpu(),
                        "model_state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "likelihood_state_dict": {
                            k: v.detach().cpu()
                            for k, v in likelihood.state_dict().items()
                        },
                        "variational_distribution_class": "TrilNaturalVariationalDistribution",
                        "likelihood_class": "GaussianLikelihood",
                    }
                )
            blob = {
                "schema_version": _SCHEMA_VERSION_PERCELL,
                "model_type": DTModelType.SVGP.value,
                "variant": "percell",
                "torch_version": torch.__version__,
                "gpytorch_version": gpytorch.__version__,
                "norm_method": self._norm_method.value,
                "x_columns": self._x_columns,
                "y_columns": self._y_columns,
                "norm_stats": self._norm_stats_blob(),
                "cells": cells,
            }
        torch.save(blob, path)

    @classmethod
    def load(
        cls, path: str, *, map_location: Optional[str] = None
    ) -> "SVGPDigitalTwin":
        blob = torch.load(path, map_location=map_location or "cpu")
        version = blob.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unsupported schema_version {version}")
        if blob["model_type"] != DTModelType.SVGP.value:
            raise ValueError(f"Expected svgp model, got {blob['model_type']}")

        engine = cls(
            norm_method=NormMethod(blob["norm_method"]),
            device=map_location,
        )
        engine._x_columns = list(blob["x_columns"])
        engine._y_columns = list(blob["y_columns"])
        stats = blob["norm_stats"]
        engine._xmeans = cls._series_list(stats["xmeans"])
        engine._xstds = cls._series_list(stats["xstds"])
        engine._xmax = cls._series_list(stats["xmax"])
        engine._xmin = cls._series_list(stats["xmin"])
        engine._ymeans = cls._series_list(stats["ymeans"])
        engine._ystds = cls._series_list(stats["ystds"])

        variant = blob.get("variant", "percell")

        if variant == "multicell_batched":
            cell_ids = list(blob["cell_ids"])
            num_cells = blob["num_cells"]
            inducing_points = blob["inducing_points"].to(engine.device)
            model = MultiCellSVGPModel(inducing_points)
            model.load_state_dict(blob["model_state_dict"])
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([num_cells])
            )
            likelihood.load_state_dict(blob["likelihood_state_dict"])
            engine._cell_ids = cell_ids
            engine._models = [model.to(engine.device)]
            engine._likelihoods = [likelihood.to(engine.device)]
            engine._multicell_batched = True
        else:
            engine._cell_ids = []
            engine._models = []
            engine._likelihoods = []
            for cell in blob["cells"]:
                inducing_points = cell["inducing_points"].to(engine.device)
                expected = int(cell["num_inducing"])
                actual = inducing_points.size(-2)
                if actual != expected:
                    raise ValueError(
                        f"Inducing-point count mismatch on cell {cell['cell_id']}: "
                        f"ckpt={expected}, state_dict={actual}"
                    )
                model = SVGPGPModel(inducing_points)
                state_inducing = cell["model_state_dict"][
                    "variational_strategy.inducing_points"
                ]
                if state_inducing.size(-2) != expected:
                    raise ValueError(
                        f"Inducing-point count mismatch on cell {cell['cell_id']}: "
                        f"ckpt={expected}, state_dict={state_inducing.size(-2)}"
                    )
                model.load_state_dict(cell["model_state_dict"])
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.load_state_dict(cell["likelihood_state_dict"])
                engine._cell_ids.append(cell["cell_id"])
                engine._models.append(model.to(engine.device))
                engine._likelihoods.append(likelihood.to(engine.device))
            engine._multicell_batched = False

        engine._trained = True
        return engine
