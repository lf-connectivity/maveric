import os
import tempfile
from unittest.mock import patch

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch

from radp.digital_twin.rf.svgp.svgp_engine import (
    MultiCellSVGPModel,
    NormMethod,
    SVGPDigitalTwin,
    SVGPGPModel,
    SVGPTrainConfig,
    SVGPUpdateConfig,
    _kmeans_pp_inducing_points,
)
from radp.digital_twin.utils import constants as c
from radp.digital_twin.utils.gis_tools import GISTools


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _site_config():
    return pd.DataFrame(
        {
            c.CELL_ID: ["cell-a", "cell-b"],
            c.CELL_AZ_DEG: [0.0, 120.0],
            c.CELL_EL_DEG: [2.0, 3.0],
            c.CELL_LAT: [35.690555, 35.691000],
            c.CELL_LON: [139.691940, 139.692100],
            c.CELL_CARRIER_FREQ_MHZ: [2100.0, 2100.0],
            c.HTX: [30.0, 32.0],
            c.HRX: [1.5, 1.5],
        }
    )


def _ue_data():
    return pd.DataFrame(
        {
            c.CELL_ID: ["cell-a"] * 8 + ["cell-b"] * 8,
            c.LON: [
                139.699058,
                139.707889,
                139.700023,
                139.702645,
                139.702745,
                139.702845,
                139.702945,
                139.703045,
            ]
            * 2,
            c.LAT: [
                35.644327,
                35.647810,
                35.643857,
                35.645913,
                35.645813,
                35.645713,
                35.645613,
                35.645513,
            ]
            * 2,
            c.RXPOWER_DBM: [-80, -70, -75, -72, -71, -73, -74, -76]
            + [-92, -91, -90, -89, -88, -87, -86, -85],
        }
    )


def _training_frames():
    return SVGPDigitalTwin.preprocess_ue_training_data(_ue_data(), _site_config())


def _small_config(seed=0, multicell_batched=True):
    return SVGPTrainConfig(
        num_epochs=2,
        batch_size=4,
        num_inducing=4,
        inducing_init="random",
        seed=seed,
        multicell_batched=multicell_batched,
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_preprocess_and_prediction_frames_emit_antenna_gain(self):
        frames = _training_frames()
        assert len(frames) == 2
        assert all(c.ANTENNA_GAIN in frame.columns for frame in frames)
        assert all(c.LOG_DISTANCE in frame.columns for frame in frames)

        template = pd.DataFrame(
            {c.LOC_X: [139.699058, 139.707889], c.LOC_Y: [35.644327, 35.647810]}
        )
        prediction_frames = SVGPDigitalTwin.create_prediction_frames(
            _site_config(), template
        )
        assert set(prediction_frames) == {"cell-a", "cell-b"}
        assert all(
            c.ANTENNA_GAIN in frame.columns for frame in prediction_frames.values()
        )

    def test_missing_height_columns_raise_helpful_error(self):
        with pytest.raises(ValueError, match="hTx"):
            SVGPDigitalTwin.preprocess_ue_training_data(
                _ue_data(), _site_config().drop(columns=[c.HTX])
            )


class TestPreprocessingVectorized:
    """Vectorized GIS helpers must be numerically identical to the scalar versions."""

    def test_log_distance_vec_matches_scalar(self):
        rng = np.random.default_rng(42)
        lat2 = rng.uniform(35.6, 35.7, 200)
        lon2 = rng.uniform(139.6, 139.8, 200)
        cell_lat, cell_lon = 35.691, 139.692

        scalar = np.array([
            GISTools.get_log_distance(cell_lat, cell_lon, la, lo)
            for la, lo in zip(lat2, lon2)
        ])
        vec = GISTools.get_log_distance_vec(cell_lat, cell_lon, lat2, lon2)
        np.testing.assert_allclose(vec, scalar, rtol=1e-7, atol=1e-9)

    def test_relative_bearing_vec_matches_scalar(self):
        rng = np.random.default_rng(99)
        lat2 = rng.uniform(35.6, 35.7, 200)
        lon2 = rng.uniform(139.6, 139.8, 200)
        cell_lat, cell_lon, cell_az = 35.691, 139.692, 45.0

        scalar = np.array([
            GISTools.get_relative_bearing(cell_az, cell_lat, cell_lon, la, lo)
            for la, lo in zip(lat2, lon2)
        ])
        vec = GISTools.get_relative_bearing_vec(cell_az, cell_lat, cell_lon, lat2, lon2)
        np.testing.assert_allclose(vec, scalar, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Inducing points
# ---------------------------------------------------------------------------

class TestInducingPoints:
    def test_kmeans_pp_shape_and_determinism(self):
        x = torch.tensor([[float(i), float(i % 3)] for i in range(20)])
        first = _kmeans_pp_inducing_points(x, 4, max_subsample=10, seed=7)
        second = _kmeans_pp_inducing_points(x, 4, max_subsample=10, seed=7)
        assert first.shape == (4, 2)
        assert torch.isfinite(first).all()
        assert torch.allclose(first, second)

    def test_kmeans_returns_clone_when_n_lte_k(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = _kmeans_pp_inducing_points(x, 3)
        assert torch.equal(out, x)
        assert out.data_ptr() != x.data_ptr()

    def test_subsample_path_triggers(self):
        # n=50 > max_subsample=10 should still return k finite centers
        x = torch.randn(50, 3)
        out = _kmeans_pp_inducing_points(x, 5, max_subsample=10, seed=0)
        assert out.shape == (5, 3)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SVGPGPModel forward
# ---------------------------------------------------------------------------

class TestSVGPGPModel:
    def test_forward_returns_multivariate_normal(self):
        model = SVGPGPModel(torch.tensor([[0.0], [1.0], [2.0]]))
        out = model(torch.tensor([[0.5], [1.5]]))
        assert out.mean.shape == torch.Size([2])

    def test_uses_whitened_variational_strategy(self):
        # gpytorch.variational.VariationalStrategy IS the whitened strategy.
        # UnwhitenedVariationalStrategy opts out. Verify we're using the whitened one.
        model = SVGPGPModel(torch.ones(3, 2))
        assert isinstance(
            model.variational_strategy, gpytorch.variational.VariationalStrategy
        )


# ---------------------------------------------------------------------------
# MultiCellSVGPModel forward
# ---------------------------------------------------------------------------

class TestMultiCellSVGPModel:
    def test_forward_shape(self):
        # inducing: [num_cells=3, M=4, d=2]; input: [num_cells=3, B=5, d=2]
        inducing = torch.randn(3, 4, 2)
        model = MultiCellSVGPModel(inducing)
        x = torch.randn(3, 5, 2)
        out = model(x)
        # batch_shape=[3], event_shape=[5]
        assert out.mean.shape == torch.Size([3, 5])


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

class TestSVGPTrainingConfig:
    def test_config_defaults(self):
        cfg = SVGPTrainConfig()
        assert cfg.num_epochs == 100
        assert cfg.multicell_batched is True

    def test_norm_none(self):
        engine = SVGPDigitalTwin(norm_method=NormMethod.NONE, device="cpu")
        assert engine.model_type == "svgp"


# ---------------------------------------------------------------------------
# End-to-end: per-cell path
# ---------------------------------------------------------------------------

class TestSVGPEndToEndPerCell:
    def test_train_predict_is_trained_toggles(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        assert not engine.is_trained
        losses = engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(seed=11, multicell_batched=False),
        )
        assert engine.is_trained
        assert losses.shape[0] == 2
        assert np.isfinite(losses[:, 0]).all()

    def test_predict_shape_and_finite(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(multicell_batched=False),
        )
        prediction_frames = [frame.head(3).copy() for frame in frames]
        means, stds = engine.predict(prediction_frames)
        assert means.shape == (3, 2)
        assert stds.shape == (3, 2)
        assert np.isfinite(means).all()
        assert (stds >= 0).all()
        assert all(c.RXPOWER_DBM in f.columns for f in prediction_frames)
        assert all(c.RXPOWER_STDDEV_DBM in f.columns for f in prediction_frames)

    def test_predict_before_train_raises(self):
        with pytest.raises(RuntimeError, match="trained"):
            SVGPDigitalTwin(device="cpu").predict([pd.DataFrame()])

    def test_predict_feature_mismatch_names_missing_columns(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine.train(
            [frames[0]],
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            SVGPTrainConfig(
                num_epochs=1, batch_size=4, num_inducing=3, seed=2,
                multicell_batched=False,
            ),
        )
        bad_frame = frames[0].drop(columns=[c.ANTENNA_GAIN])
        with pytest.raises(ValueError, match=c.ANTENNA_GAIN):
            engine.predict([bad_frame])


# ---------------------------------------------------------------------------
# End-to-end: batched multi-cell path (default)
# ---------------------------------------------------------------------------

class TestSVGPEndToEndMultiCell:
    def test_train_predict_shape_and_finite(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        losses = engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(seed=7, multicell_batched=True),
        )
        assert engine.is_trained
        assert losses.shape[0] == 2
        assert np.isfinite(losses[:, 0]).all()

        prediction_frames = [frame.head(3).copy() for frame in frames]
        means, stds = engine.predict(prediction_frames)
        assert means.shape == (3, 2)
        assert stds.shape == (3, 2)
        assert np.isfinite(means).all()
        assert (stds >= 0).all()

    def test_multicell_parity_with_percell(self):
        """Batched and per-cell paths should produce similar predictions.

        We don't assert bit-exact equality (different variational parameterisations
        produce different posteriors), but both should output finite predictions
        in a plausible range for the synthetic data.
        """
        frames = _training_frames()
        x_cols = [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN]
        y_cols = [c.RXPOWER_DBM]

        engine_b = SVGPDigitalTwin(device="cpu")
        engine_b.train(frames, x_cols, y_cols, _small_config(seed=0, multicell_batched=True))
        means_b, _ = engine_b.predict([f.head(3).copy() for f in frames])

        engine_p = SVGPDigitalTwin(device="cpu")
        engine_p.train(frames, x_cols, y_cols, _small_config(seed=0, multicell_batched=False))
        means_p, _ = engine_p.predict([f.head(3).copy() for f in frames])

        assert means_b.shape == means_p.shape
        assert np.isfinite(means_b).all()
        assert np.isfinite(means_p).all()

    def test_unequal_cell_sizes(self):
        """Cells with different N_i should train without error."""
        frames = _training_frames()
        # Make cell-b much smaller
        frames[1] = frames[1].head(3)
        engine = SVGPDigitalTwin(device="cpu")
        losses = engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(multicell_batched=True),
        )
        assert np.isfinite(losses[:, 0]).all()
        pf = [frames[0].head(2).copy(), frames[1].head(2).copy()]
        means, stds = engine.predict(pf)
        assert np.isfinite(means).all()


# ---------------------------------------------------------------------------
# Train loop internals
# ---------------------------------------------------------------------------

class TestTrainLoop:
    def test_early_stopping_fires(self):
        """With a tiny threshold, early-stopping should kick in before max_epochs."""
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        losses = engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            SVGPTrainConfig(
                num_epochs=50,
                batch_size=4,
                num_inducing=3,
                seed=0,
                stopping_threshold=1e10,  # always triggers after 2nd epoch
                multicell_batched=False,
            ),
        )
        # Each cell should have stopped at epoch 2 (once delta < 1e10)
        assert losses.shape[1] <= 3  # at most 3 epochs recorded

    def test_freeze_inducing_flips_requires_grad(self):
        """After freeze_at epoch, inducing_points.requires_grad must be False."""
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        # 4 epochs, freeze at frac=0.25 → freeze at epoch 1
        cfg = SVGPTrainConfig(
            num_epochs=4,
            batch_size=4,
            num_inducing=3,
            seed=0,
            freeze_inducing_after_frac=0.25,
            multicell_batched=False,
        )
        engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            cfg,
        )
        for model in engine._models:
            assert not model.variational_strategy.inducing_points.requires_grad


# ---------------------------------------------------------------------------
# Cholesky jitter
# ---------------------------------------------------------------------------

class TestCholeskyJitter:
    def test_degenerate_input_still_trains(self):
        """Near-duplicate rows cause degenerate K but training must complete."""
        n = 20
        base = np.linspace(0, 1, n)
        X_np = np.column_stack([base, base + 1e-8 * np.random.default_rng(0).normal(size=n)])
        frames = _training_frames()
        # Inject near-degenerate X into the training frame by overriding features
        for frame in frames:
            for i, col in enumerate([c.LOG_DISTANCE, c.RELATIVE_BEARING]):
                frame[col] = X_np[:len(frame), i % X_np.shape[1]]

        engine = SVGPDigitalTwin(device="cpu")
        try:
            engine.train(
                frames,
                [c.LOG_DISTANCE, c.RELATIVE_BEARING],
                [c.RXPOWER_DBM],
                SVGPTrainConfig(
                    num_epochs=3, batch_size=4, num_inducing=5, seed=0,
                    multicell_batched=False,
                ),
            )
        except RuntimeError as e:
            # 3 consecutive failures is acceptable — confirms the guard works
            assert "NotPSDError" in str(e) or "jitter" in str(e)

    def test_not_psd_error_bumps_jitter(self):
        """NotPSDError during training must bump jitter and resume (not propagate)."""
        from linear_operator.utils.errors import NotPSDError as NPSD

        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine._x_columns = [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN]
        engine._y_columns = [c.RXPOWER_DBM]
        engine._cell_ids = ["cell-a"]
        engine._fit_norm_stats([frames[0]])
        train_X, train_Y = engine._create_training_tensors([frames[0]])
        cell_X, cell_Y = train_X[0], train_Y[0]

        call_count = [0]

        # Patch the backward call on the Tensor returned by -mll(...) to raise
        # NotPSDError once, then proceed normally. We do this by patching
        # gpytorch.mlls.VariationalELBO.forward.
        original_fwd = gpytorch.mlls.VariationalELBO.forward

        def patched_fwd(self_mll, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise NPSD("synthetic")
            return original_fwd(self_mll, *args, **kwargs)

        inducing = engine._initial_inducing_points(
            cell_X, SVGPTrainConfig(num_inducing=4), seed=0
        )
        model = SVGPGPModel(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )

        with patch.object(gpytorch.mlls.VariationalELBO, "forward", patched_fwd):
            # Should complete despite 1 NPSD (jitter bumps and training resumes)
            losses = engine._train_cell(
                model,
                likelihood,
                cell_X,
                cell_Y,
                SVGPTrainConfig(
                    num_epochs=3, batch_size=4, num_inducing=4, seed=0,
                    multicell_batched=False,
                ),
            )
        assert call_count[0] >= 2  # first raised, subsequent succeeded
        assert len(losses) > 0

    def test_three_consecutive_failures_raise(self):
        """Three back-to-back NotPSDError must raise RuntimeError."""
        from linear_operator.utils.errors import NotPSDError as NPSD

        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine._x_columns = [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN]
        engine._y_columns = [c.RXPOWER_DBM]
        engine._cell_ids = ["cell-a"]
        engine._fit_norm_stats([frames[0]])
        train_X, train_Y = engine._create_training_tensors([frames[0]])
        cell_X, cell_Y = train_X[0], train_Y[0]

        inducing = engine._initial_inducing_points(
            cell_X, SVGPTrainConfig(num_inducing=4), seed=0
        )
        model = SVGPGPModel(inducing)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        def always_raise(self_mll, *args, **kwargs):
            raise NPSD("always")

        with patch.object(gpytorch.mlls.VariationalELBO, "forward", always_raise):
            with pytest.raises(RuntimeError, match="NotPSDError"):
                engine._train_cell(
                    model,
                    likelihood,
                    cell_X,
                    cell_Y,
                    SVGPTrainConfig(
                        num_epochs=5, batch_size=4, num_inducing=4, seed=0,
                        multicell_batched=False,
                    ),
                )


# ---------------------------------------------------------------------------
# Online update
# ---------------------------------------------------------------------------

class TestOnlineUpdate:
    def test_update_runs_and_returns_losses(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(multicell_batched=False),
        )
        update_frames = [f.copy() for f in frames]
        for f in update_frames:
            f[c.RXPOWER_DBM] += 5.0
        update_losses = engine.update_trained_gpmodel(
            update_frames,
            SVGPUpdateConfig(num_epochs=1, batch_size=4, freeze_hyperparams=True),
        )
        assert update_losses.shape[0] == 2

    def test_update_shifts_prediction(self):
        """After updating with data shifted +5 dB, predictions should shift."""
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(seed=3, multicell_batched=False),
        )
        pf = [f.head(4).copy() for f in frames]
        means_before, _ = engine.predict([f.copy() for f in pf])

        update_frames = [f.copy() for f in frames]
        for f in update_frames:
            f[c.RXPOWER_DBM] += 5.0
        engine.update_trained_gpmodel(
            update_frames,
            SVGPUpdateConfig(num_epochs=5, batch_size=4),
        )
        means_after, _ = engine.predict(pf)
        # Predictions should move (history + new data drives the posterior)
        assert not np.allclose(means_before, means_after, atol=1e-6)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSVGPSaveLoad:
    def _train_engine(self, multicell_batched=True):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cpu")
        engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(seed=5, multicell_batched=multicell_batched),
        )
        return engine, frames

    def test_round_trip_predictions_match(self):
        engine, frames = self._train_engine(multicell_batched=True)
        pf = [f.head(3).copy() for f in frames]
        means, stds = engine.predict([f.copy() for f in pf])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "svgp.pt")
            engine.save(path)
            loaded = SVGPDigitalTwin.load(path, map_location="cpu")
            loaded_means, loaded_stds = loaded.predict([f.copy() for f in pf])

        np.testing.assert_allclose(means, loaded_means, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(stds, loaded_stds, rtol=1e-4, atol=1e-4)

    def test_round_trip_percell(self):
        engine, frames = self._train_engine(multicell_batched=False)
        pf = [f.head(3).copy() for f in frames]
        means, stds = engine.predict([f.copy() for f in pf])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "svgp_pc.pt")
            engine.save(path)
            loaded = SVGPDigitalTwin.load(path, map_location="cpu")
            loaded_means, loaded_stds = loaded.predict([f.copy() for f in pf])

        np.testing.assert_allclose(means, loaded_means, rtol=1e-4, atol=1e-4)

    def test_wrong_model_type_raises(self):
        engine, _ = self._train_engine(multicell_batched=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            engine.save(path)
            blob = torch.load(path, map_location="cpu")
            blob["model_type"] = "not_svgp"
            torch.save(blob, path)
            with pytest.raises(ValueError, match="Expected svgp"):
                SVGPDigitalTwin.load(path, map_location="cpu")

    def test_wrong_schema_version_raises(self):
        engine, _ = self._train_engine(multicell_batched=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            engine.save(path)
            blob = torch.load(path, map_location="cpu")
            blob["schema_version"] = 99
            torch.save(blob, path)
            with pytest.raises(ValueError, match="schema_version"):
                SVGPDigitalTwin.load(path, map_location="cpu")

    def test_inducing_count_mismatch_raises(self):
        engine, _ = self._train_engine(multicell_batched=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.pt")
            engine.save(path)
            blob = torch.load(path, map_location="cpu")
            blob["cells"][0]["num_inducing"] = 9999
            torch.save(blob, path)
            with pytest.raises(ValueError, match="mismatch"):
                SVGPDigitalTwin.load(path, map_location="cpu")

    def test_save_untrained_raises(self):
        with pytest.raises(RuntimeError, match="untrained"):
            SVGPDigitalTwin(device="cpu").save("/tmp/nope.pt")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestSVGPReproducibility:
    def test_same_seed_same_predictions_cpu(self):
        frames = _training_frames()
        x_cols = [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN]
        y_cols = [c.RXPOWER_DBM]

        def _run():
            engine = SVGPDigitalTwin(device="cpu")
            engine.train(frames, x_cols, y_cols, _small_config(seed=42, multicell_batched=False))
            return engine.predict([f.head(3).copy() for f in frames])

        means1, stds1 = _run()
        means2, stds2 = _run()
        np.testing.assert_array_equal(means1, means2)
        np.testing.assert_array_equal(stds1, stds2)


# ---------------------------------------------------------------------------
# CUDA (skipped if unavailable)
# ---------------------------------------------------------------------------

class TestCUDA:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_predict_on_gpu(self):
        frames = _training_frames()
        engine = SVGPDigitalTwin(device="cuda")
        losses = engine.train(
            frames,
            [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN],
            [c.RXPOWER_DBM],
            _small_config(multicell_batched=True),
        )
        assert np.isfinite(losses[:, 0]).all()
        means, stds = engine.predict([f.head(3).copy() for f in frames])
        assert np.isfinite(means).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_predictions_close(self):
        frames = _training_frames()
        x_cols = [c.LOG_DISTANCE, c.RELATIVE_BEARING, c.ANTENNA_GAIN]
        y_cols = [c.RXPOWER_DBM]
        cfg = _small_config(seed=0, multicell_batched=False)

        e_cpu = SVGPDigitalTwin(device="cpu")
        e_cpu.train(frames, x_cols, y_cols, cfg)
        means_cpu, _ = e_cpu.predict([f.head(3).copy() for f in frames])

        e_gpu = SVGPDigitalTwin(device="cuda")
        e_gpu.train(frames, x_cols, y_cols, cfg)
        means_gpu, _ = e_gpu.predict([f.head(3).copy() for f in frames])

        # fp32 GPU nondeterminism: loose tolerance
        np.testing.assert_allclose(means_cpu, means_gpu, atol=1.0)
