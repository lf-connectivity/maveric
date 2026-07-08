"""
Validation tests for Agentic MRO — Task 3.1.

Verifies that the .py source produces correct MRO outputs aligned with the
original implementation. Tests are organised into three groups:

  1. Stop-condition logic        — pure Python, no external dependencies
  2. Parameter-validation utils  — pure Python, no external dependencies
  3. Baseline regression         — mocked LLM + real MRO evaluation on fixture

Running:
    pytest apps/mobility_robustness_optimization/agentic_mro/tests/test_validation.py -v
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — makes the agentic_mro package importable when running pytest
# from the repo root (pytest.ini sets pythonpath = .)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[5]   # …/AgenticMaveric
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    """Build a minimal AgenticMROState dict for unit tests."""
    base = {
        "llm_config": {"provider": "groq", "model": "test"},
        "input_csv_path": None,
        "raw_dataframe": None,
        "insights_dataframe": None,
        "analyzer_markdown": None,
        "strategy_json": None,
        "hyst_range": None,
        "ttt_range": None,
        "iteration_count": 0,
        "tested_parameters": [],
        "best_score": float("-inf"),
        "best_hyst": None,
        "best_ttt": None,
        "target_score": 0.80,
        "max_iterations": 3,
        "plateau_detected": False,
        "rlf_threshold": -4.0,
        "final_output": None,
    }
    base.update(overrides)
    return base


# ===========================================================================
# Group 1: Stop-condition logic
# ===========================================================================

from apps.mobility_robustness_optimization.agentic_mro.utils.stop_conditions import (
    detect_plateau,
    get_stop_reason,
    should_continue_optimization,
    update_stop_conditions,
)


class TestShouldContinueOptimization:
    """should_continue_optimization returns 'finalize' for every stop trigger
    and 'continue' when no condition is met."""

    def test_finalize_when_max_iterations_reached(self):
        state = _make_state(iteration_count=3, max_iterations=3)
        assert should_continue_optimization(state) == "finalize"

    def test_finalize_when_iterations_exceed_max(self):
        state = _make_state(iteration_count=5, max_iterations=3)
        assert should_continue_optimization(state) == "finalize"

    def test_finalize_when_target_score_achieved(self):
        state = _make_state(best_score=0.85, target_score=0.80)
        assert should_continue_optimization(state) == "finalize"

    def test_finalize_when_target_score_exactly_met(self):
        state = _make_state(best_score=0.80, target_score=0.80)
        assert should_continue_optimization(state) == "finalize"

    def test_finalize_when_plateau_detected(self):
        state = _make_state(plateau_detected=True)
        assert should_continue_optimization(state) == "finalize"

    def test_continue_when_no_condition_met(self):
        state = _make_state(
            iteration_count=1,
            max_iterations=3,
            best_score=0.50,
            target_score=0.80,
            plateau_detected=False,
        )
        assert should_continue_optimization(state) == "continue"

    def test_continue_at_zero_iterations(self):
        state = _make_state(iteration_count=0, max_iterations=3, best_score=float("-inf"))
        assert should_continue_optimization(state) == "continue"


class TestDetectPlateau:
    """detect_plateau returns True only when the last two scores show < 1%
    improvement."""

    def test_no_plateau_with_fewer_than_two_entries(self):
        state = _make_state(tested_parameters=[{"score": 0.70}])
        assert detect_plateau(state) is False

    def test_no_plateau_with_empty_history(self):
        state = _make_state(tested_parameters=[])
        assert detect_plateau(state) is False

    def test_plateau_detected_when_improvement_below_threshold(self):
        # 0.71% improvement — below 1% threshold
        state = _make_state(
            tested_parameters=[{"score": 0.70}, {"score": 0.705}]
        )
        assert detect_plateau(state) is True

    def test_no_plateau_when_improvement_above_threshold(self):
        # 5% improvement — above 1% threshold
        state = _make_state(
            tested_parameters=[{"score": 0.60}, {"score": 0.63}]
        )
        assert detect_plateau(state) is False

    def test_plateau_when_score_identical(self):
        state = _make_state(
            tested_parameters=[{"score": 0.75}, {"score": 0.75}]
        )
        assert detect_plateau(state) is True

    def test_only_last_two_entries_are_checked(self):
        # Large improvement overall, but tiny improvement in last 2 → plateau
        state = _make_state(
            tested_parameters=[
                {"score": 0.10},
                {"score": 0.50},
                {"score": 0.80},
                {"score": 0.802},   # 0.25% from 0.80 → plateau
            ]
        )
        assert detect_plateau(state) is True

    def test_no_plateau_when_prev_score_is_zero(self):
        """Division by zero guard: non-zero denominator required."""
        state = _make_state(
            tested_parameters=[{"score": 0.0}, {"score": 0.5}]
        )
        # Should not raise and should return False (improvement is infinite)
        assert detect_plateau(state) is False


class TestGetStopReason:
    """get_stop_reason returns a human-readable string matching the active
    stop condition."""

    def test_max_iterations_reason(self):
        state = _make_state(iteration_count=3, max_iterations=3)
        reason = get_stop_reason(state)
        assert "Maximum iterations" in reason
        assert "3/3" in reason

    def test_target_score_reason(self):
        state = _make_state(best_score=0.85, target_score=0.80)
        reason = get_stop_reason(state)
        assert "Target score" in reason

    def test_plateau_reason(self):
        state = _make_state(plateau_detected=True)
        reason = get_stop_reason(state)
        assert "Plateau" in reason or "plateau" in reason

    def test_in_progress_reason(self):
        state = _make_state(
            iteration_count=1,
            max_iterations=3,
            best_score=0.40,
            target_score=0.80,
            plateau_detected=False,
        )
        reason = get_stop_reason(state)
        assert "progress" in reason.lower() or "running" in reason.lower()


class TestUpdateStopConditions:
    """update_stop_conditions sets plateau_detected correctly in-place."""

    def test_plateau_flag_set_when_plateau_exists(self):
        state = _make_state(
            tested_parameters=[{"score": 0.70}, {"score": 0.701}]
        )
        updated = update_stop_conditions(state)
        assert updated["plateau_detected"] is True

    def test_plateau_flag_clear_when_no_plateau(self):
        state = _make_state(
            tested_parameters=[{"score": 0.50}, {"score": 0.60}]
        )
        updated = update_stop_conditions(state)
        assert updated["plateau_detected"] is False


# ===========================================================================
# Group 2: Parameter-validation utilities
# ===========================================================================

from apps.mobility_robustness_optimization.agentic_mro.utils.evaluation import (
    count_handovers,
    count_rlfs,
    validate_parameters,
)


class TestValidateParameters:
    """validate_parameters accepts in-range values and rejects out-of-range."""

    def test_valid_parameters_accepted(self):
        is_valid, msg = validate_parameters(3.5, 6, (0.0, 10.0), (2, 50))
        assert is_valid is True
        assert msg == ""

    def test_hysteresis_below_min_rejected(self):
        is_valid, msg = validate_parameters(-1.0, 6, (0.0, 10.0), (2, 50))
        assert is_valid is False
        assert "Hysteresis" in msg

    def test_hysteresis_above_max_rejected(self):
        is_valid, msg = validate_parameters(15.0, 6, (0.0, 10.0), (2, 50))
        assert is_valid is False
        assert "Hysteresis" in msg

    def test_ttt_below_min_rejected(self):
        is_valid, msg = validate_parameters(3.5, 1, (0.0, 10.0), (2, 50))
        assert is_valid is False
        assert "TTT" in msg

    def test_ttt_above_max_rejected(self):
        is_valid, msg = validate_parameters(3.5, 100, (0.0, 10.0), (2, 50))
        assert is_valid is False
        assert "TTT" in msg

    def test_boundary_values_accepted(self):
        is_valid, _ = validate_parameters(0.0, 2, (0.0, 10.0), (2, 50))
        assert is_valid is True

        is_valid, _ = validate_parameters(10.0, 50, (0.0, 10.0), (2, 50))
        assert is_valid is True


class TestCountHandovers:
    """count_handovers counts cell-ID changes per UE (ignoring RLF rows)."""

    def test_no_handovers_when_same_cell(self):
        df = pd.DataFrame({
            "ue_id": [0, 0, 0],
            "tick":  [0, 1, 2],
            "cell_id": ["A", "A", "A"],
        })
        assert count_handovers(df) == 0

    def test_one_handover_detected(self):
        df = pd.DataFrame({
            "ue_id": [0, 0, 0],
            "tick":  [0, 1, 2],
            "cell_id": ["A", "A", "B"],
        })
        assert count_handovers(df) == 1

    def test_rlf_destination_not_counted_but_rlf_source_is(self):
        df = pd.DataFrame({
            "ue_id": [0, 0, 0],
            "tick":  [0, 1, 2],
            "cell_id": ["A", "RLF", "B"],
        })
        # A→RLF: not a handover (destination is RLF — implementation skips it)
        # RLF→B: IS counted as a handover (reconnection after RLF)
        # Implementation: only skips transitions WHERE cell_ids[i] == 'RLF'
        assert count_handovers(df) == 1

    def test_multiple_ues_handovers_summed(self):
        df = pd.DataFrame({
            "ue_id": [0, 0, 1, 1],
            "tick":  [0, 1, 0, 1],
            "cell_id": ["A", "B", "C", "C"],
        })
        # UE 0: 1 handover; UE 1: 0 handovers → total 1
        assert count_handovers(df) == 1


class TestCountRLFs:
    """count_rlfs counts rows where cell_id == 'RLF'."""

    def test_no_rlfs(self):
        df = pd.DataFrame({"cell_id": ["A", "B", "C"]})
        assert count_rlfs(df) == 0

    def test_rlf_rows_counted(self):
        df = pd.DataFrame({"cell_id": ["A", "RLF", "RLF", "B"]})
        assert count_rlfs(df) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"cell_id": []})
        assert count_rlfs(df) == 0

    def test_missing_cell_id_column_returns_zero(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        assert count_rlfs(df) == 0


# ===========================================================================
# Group 3: Baseline regression — mocked LLM + real MRO evaluation
# ===========================================================================

# Fixture paths
_FIXTURE_CSV = FIXTURES_DIR / "sim_data_small.csv"
_EXPECTED_JSON = FIXTURES_DIR / "expected_result.json"

# Strategy JSON that the mocked strategy LLM will return
_MOCK_STRATEGY_JSON = {
    "parameter_recommendations": {
        "hysteresis": {"min": 0.0, "max": 5.0, "reasoning": "mock"},
        "time_to_trigger": {"min": 2, "max": 10, "reasoning": "mock"},
    },
    "optimization_strategy": {
        "priority": "BALANCED",
        "test_sequence": "SIMULTANEOUS",
        "predicted_optimal": {"hysteresis": 2.0, "ttt": 5},
        "reasoning": "mock strategy for baseline test",
    },
}

# Fixed parameters the mocked coordinator LLM will always suggest
_MOCK_COORD_JSON = {
    "suggested_hyst": 2.0,
    "suggested_ttt": 5,
    "reasoning": "fixed mock for baseline regression test",
}


def _make_mock_llm(generate_text="Mock network analysis.", generate_json_val=None):
    """Return a MagicMock that mimics BaseLLMProvider."""
    mock = MagicMock()
    mock.generate.return_value = generate_text
    mock.generate_json.return_value = generate_json_val or {}
    return mock


@pytest.fixture(scope="module")
def fixture_df():
    """Load the small sim-data fixture once per test-module run."""
    if not _FIXTURE_CSV.exists():
        pytest.skip(f"Fixture CSV not found: {_FIXTURE_CSV}")
    return pd.read_csv(_FIXTURE_CSV)


@pytest.fixture(scope="module")
def expected():
    """Load the golden expected-result values."""
    if not _EXPECTED_JSON.exists():
        pytest.skip(f"Expected result JSON not found: {_EXPECTED_JSON}")
    with open(_EXPECTED_JSON) as fh:
        return json.load(fh)


class TestRunAgenticMROOutputStructure:
    """Verify the final_output dict has all required keys (pipeline plumbing)."""

    def test_output_has_required_keys(self, fixture_df, expected):
        from apps.mobility_robustness_optimization.agentic_mro.main import run_agentic_mro

        analyzer_mock = _make_mock_llm(generate_text="Poor signal quality detected.")
        strategy_mock = _make_mock_llm(generate_json_val=_MOCK_STRATEGY_JSON)
        coordinator_mock = _make_mock_llm(generate_json_val=_MOCK_COORD_JSON)

        with (
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.analyzer_node.create_llm_provider",
                return_value=analyzer_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.strategy_node.create_llm_provider",
                return_value=strategy_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.coordinator_node.create_llm_provider",
                return_value=coordinator_mock,
            ),
        ):
            result = run_agentic_mro(
                csv_path=str(_FIXTURE_CSV),
                llm_config={"provider": "groq", "model": "test"},
                target_score=999_999.0,   # never hit — forces max_iterations stop
                max_iterations=1,
                rlf_threshold=-4.0,
            )

        required_keys = {
            "best_hysteresis",
            "best_ttt",
            "best_score",
            "total_iterations",
            "target_score",
            "tested_parameters",
        }
        assert required_keys.issubset(set(result.keys())), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )


class TestRunAgenticMROMatchesBaseline:
    """Run the full pipeline with a deterministic LLM mock and assert the
    output is within tolerance of the pre-computed golden values."""

    def _run(self, fixture_df, max_iterations=1):
        from apps.mobility_robustness_optimization.agentic_mro.main import run_agentic_mro

        analyzer_mock = _make_mock_llm(generate_text="Poor signal quality detected.")
        strategy_mock = _make_mock_llm(generate_json_val=_MOCK_STRATEGY_JSON)
        coordinator_mock = _make_mock_llm(generate_json_val=_MOCK_COORD_JSON)

        with (
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.analyzer_node.create_llm_provider",
                return_value=analyzer_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.strategy_node.create_llm_provider",
                return_value=strategy_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.coordinator_node.create_llm_provider",
                return_value=coordinator_mock,
            ),
        ):
            return run_agentic_mro(
                csv_path=str(_FIXTURE_CSV),
                llm_config={"provider": "groq", "model": "test"},
                target_score=999_999.0,
                max_iterations=max_iterations,
                rlf_threshold=-4.0,
            )

    def test_hysteresis_within_tolerance(self, fixture_df, expected):
        result = self._run(fixture_df)
        exp_hyst = expected["expected_hyst"]
        tolerance = expected["hyst_tolerance"]
        assert result["best_hysteresis"] is not None, "best_hysteresis is None"
        assert abs(result["best_hysteresis"] - exp_hyst) <= tolerance, (
            f"hyst {result['best_hysteresis']:.4f} not within ±{tolerance} of {exp_hyst}"
        )

    def test_ttt_within_tolerance(self, fixture_df, expected):
        result = self._run(fixture_df)
        exp_ttt = expected["expected_ttt"]
        tolerance = expected["ttt_tolerance"]
        assert result["best_ttt"] is not None, "best_ttt is None"
        assert abs(result["best_ttt"] - exp_ttt) <= tolerance, (
            f"TTT {result['best_ttt']} not within ±{tolerance} of {exp_ttt}"
        )

    def test_score_within_tolerance(self, fixture_df, expected):
        result = self._run(fixture_df)
        exp_score = expected["expected_score"]
        tolerance = expected["score_tolerance"]
        assert result["best_score"] > 0.0, "best_score should be positive"
        assert abs(result["best_score"] - exp_score) <= tolerance, (
            f"score {result['best_score']:.4f} not within ±{tolerance} of {exp_score}"
        )

    def test_stops_at_max_iterations(self, fixture_df, expected):
        result = self._run(fixture_df, max_iterations=1)
        assert result["total_iterations"] == expected["expected_iterations"]

    def test_tested_parameters_recorded(self, fixture_df, expected):
        result = self._run(fixture_df, max_iterations=1)
        params = result["tested_parameters"]
        assert len(params) == 1, f"Expected 1 entry, got {len(params)}"
        entry = params[0]
        assert "hyst" in entry
        assert "ttt" in entry
        assert "score" in entry


class TestStopConditionsEndToEnd:
    """Verify that each stop condition actually halts the pipeline loop."""

    def _run_with_target(self, target_score, max_iterations):
        from apps.mobility_robustness_optimization.agentic_mro.main import run_agentic_mro

        analyzer_mock = _make_mock_llm(generate_text="Analysis.")
        strategy_mock = _make_mock_llm(generate_json_val=_MOCK_STRATEGY_JSON)
        coordinator_mock = _make_mock_llm(generate_json_val=_MOCK_COORD_JSON)

        with (
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.analyzer_node.create_llm_provider",
                return_value=analyzer_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.strategy_node.create_llm_provider",
                return_value=strategy_mock,
            ),
            patch(
                "apps.mobility_robustness_optimization.agentic_mro.nodes.coordinator_node.create_llm_provider",
                return_value=coordinator_mock,
            ),
        ):
            return run_agentic_mro(
                csv_path=str(_FIXTURE_CSV),
                llm_config={"provider": "groq", "model": "test"},
                target_score=target_score,
                max_iterations=max_iterations,
                rlf_threshold=-4.0,
            )

    def test_iteration_cap_stops_pipeline(self):
        """Pipeline stops exactly at max_iterations=2."""
        result = self._run_with_target(target_score=999_999.0, max_iterations=2)
        assert result["total_iterations"] == 2

    def test_target_score_stops_pipeline_early(self):
        """If target_score is set below the expected evaluation score (10.0),
        the pipeline should stop after the first iteration that exceeds it."""
        # The fixture evaluation score is 10.0; set target below that
        result = self._run_with_target(target_score=5.0, max_iterations=5)
        # Must stop before max_iterations (5) because target is quickly met
        assert result["total_iterations"] < 5
        assert result["best_score"] >= 5.0
