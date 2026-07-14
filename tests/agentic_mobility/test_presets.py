"""Unit tests for scenario presets and their wiring into parameter_agent.

Covers:
  - The PRESETS registry has urban/suburban/rural with sane shape.
  - get_preset() lookup is case-insensitive and tolerant of None/unknown.
  - parameter_agent applies preset values when scenario matches and the
    user did not provide an explicit distribution.
  - parameter_agent leaves LLM output alone when the user did provide one.
  - The three presets yield distinguishable alpha + distribution values.
"""
from unittest.mock import MagicMock, patch

import pytest

from radp.digital_twin.agentic_mobility.defaults import (
    PRESETS,
    RURAL_PRESET,
    SUBURBAN_PRESET,
    URBAN_PRESET,
    get_preset,
)
from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import (
    DistributionSource,
    QueryIntent,
    ScenarioType,
    UEDistribution,
)
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_PRESET_KEYS = {
    "name",
    "ue_density_per_km2",
    "default_distribution",
    "alpha",
    "variance",
    "recommended_ticks",
    "recommended_cells",
}


def _base_state(**overrides) -> MobilityGenerationState:
    state: MobilityGenerationState = {
        "user_query": "Generate 50 UEs",
        "current_query": "Generate 50 UEs",
        "query_intent": None,
        "location_data": None,
        "gen_params": None,
        "validation_result": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def _query_intent(scenario: ScenarioType, ue_distribution=None) -> QueryIntent:
    return QueryIntent(
        scenario_type=scenario,
        location="Anywhere",
        num_ues=50,
        num_ticks=50,
        ue_distribution=ue_distribution,
        raw_query="Generate 50 UEs",
    )


def _llm_gen_params() -> GenParams:
    """Synthetic LLM-shaped output — values intentionally unlike any preset."""
    return GenParams(
        alpha=0.33,
        variance=0.44,
        ue_class_distribution={
            "stationary": 0.10,
            "pedestrian": 0.10,
            "cyclist": 0.10,
            "car": 0.70,
        },
        velocity_adjustments={
            "stationary": {"velocity": 0.0, "velocity_variance": 0.0},
            "pedestrian": {"velocity": 1.4, "velocity_variance": 0.3},
            "cyclist": {"velocity": 4.0, "velocity_variance": 0.5},
            "car": {"velocity": 13.9, "velocity_variance": 2.0},
        },
    )


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


class TestPresetRegistry:
    def test_registry_has_three_presets(self):
        assert set(PRESETS.keys()) == {"urban", "suburban", "rural"}

    @pytest.mark.parametrize("preset", [URBAN_PRESET, SUBURBAN_PRESET, RURAL_PRESET])
    def test_preset_has_required_keys(self, preset):
        assert REQUIRED_PRESET_KEYS.issubset(preset.keys())

    @pytest.mark.parametrize("preset", [URBAN_PRESET, SUBURBAN_PRESET, RURAL_PRESET])
    def test_default_distribution_sums_to_one(self, preset):
        total = sum(preset["default_distribution"].values())
        assert abs(total - 1.0) < 1e-6, f"{preset['name']} distribution sums to {total}"

    @pytest.mark.parametrize("preset", [URBAN_PRESET, SUBURBAN_PRESET, RURAL_PRESET])
    def test_alpha_in_valid_range(self, preset):
        assert 0.0 <= preset["alpha"] <= 1.0

    @pytest.mark.parametrize("preset", [URBAN_PRESET, SUBURBAN_PRESET, RURAL_PRESET])
    def test_variance_non_negative(self, preset):
        assert preset["variance"] >= 0.0

    def test_presets_distinguishable(self):
        """Acceptance criterion: each preset must yield a different alpha
        and a different distribution so end-to-end runs are visibly distinct."""
        alphas = {p["alpha"] for p in PRESETS.values()}
        assert len(alphas) == 3, "Each preset must have a unique alpha"

        distributions = {tuple(sorted(p["default_distribution"].items())) for p in PRESETS.values()}
        assert len(distributions) == 3, "Each preset must have a unique distribution"


# ---------------------------------------------------------------------------
# get_preset() lookup behavior
# ---------------------------------------------------------------------------


class TestGetPreset:
    def test_returns_preset_by_lowercase_name(self):
        assert get_preset("urban") is URBAN_PRESET
        assert get_preset("suburban") is SUBURBAN_PRESET
        assert get_preset("rural") is RURAL_PRESET

    def test_case_insensitive(self):
        assert get_preset("URBAN") is URBAN_PRESET
        assert get_preset("Suburban") is SUBURBAN_PRESET

    def test_returns_none_for_unknown(self):
        assert get_preset("highway") is None
        assert get_preset("mixed") is None

    def test_returns_none_for_none_or_empty(self):
        assert get_preset(None) is None
        assert get_preset("") is None


# ---------------------------------------------------------------------------
# parameter_agent wiring
# ---------------------------------------------------------------------------


class TestParameterAgentPresetWiring:
    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_urban_preset_applied_when_no_user_distribution(self, MockParameterChain):
        intent = _query_intent(ScenarioType.URBAN, ue_distribution=None)
        mock_chain = MagicMock()
        mock_chain.generate.return_value = (_llm_gen_params(), intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        result = parameter_agent.node(_base_state(query_intent=intent.dict()))

        assert result["gen_params"]["alpha"] == URBAN_PRESET["alpha"]
        assert result["gen_params"]["variance"] == URBAN_PRESET["variance"]
        assert result["gen_params"]["ue_class_distribution"] == URBAN_PRESET["default_distribution"]

        ue_dist = result["query_intent"]["ue_distribution"]
        assert ue_dist["pedestrian"] == URBAN_PRESET["default_distribution"]["pedestrian"]
        assert ue_dist["source"] == DistributionSource.PREDICTED.value

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_rural_preset_applied(self, MockParameterChain):
        intent = _query_intent(ScenarioType.RURAL, ue_distribution=None)
        mock_chain = MagicMock()
        mock_chain.generate.return_value = (_llm_gen_params(), intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        result = parameter_agent.node(_base_state(query_intent=intent.dict()))

        assert result["gen_params"]["alpha"] == RURAL_PRESET["alpha"]
        assert result["gen_params"]["variance"] == RURAL_PRESET["variance"]
        assert result["gen_params"]["ue_class_distribution"]["car"] == RURAL_PRESET["default_distribution"]["car"]

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_no_override_when_scenario_has_no_preset(self, MockParameterChain):
        """Highway is not a preset; LLM output should pass through."""
        intent = _query_intent(ScenarioType.HIGHWAY, ue_distribution=None)
        llm_params = _llm_gen_params()
        mock_chain = MagicMock()
        mock_chain.generate.return_value = (llm_params, intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        result = parameter_agent.node(_base_state(query_intent=intent.dict()))

        assert result["gen_params"]["alpha"] == llm_params.alpha
        assert result["gen_params"]["variance"] == llm_params.variance
        assert result["gen_params"]["ue_class_distribution"] == llm_params.ue_class_distribution

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_user_distribution_preserved_even_for_preset_scenario(self, MockParameterChain):
        """If the user gave an explicit distribution, the preset must NOT override it."""
        user_dist = UEDistribution(
            stationary=0.0,
            pedestrian=0.0,
            cyclist=0.0,
            car=1.0,
            source=DistributionSource.PARSED,
        )
        intent = _query_intent(ScenarioType.URBAN, ue_distribution=user_dist)
        llm_params = _llm_gen_params()
        mock_chain = MagicMock()
        mock_chain.generate.return_value = (llm_params, intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        result = parameter_agent.node(_base_state(query_intent=intent.dict()))

        # LLM output stays — preset wiring should be a no-op here.
        assert result["gen_params"]["alpha"] == llm_params.alpha
        assert result["gen_params"]["variance"] == llm_params.variance
        assert result["query_intent"]["ue_distribution"]["car"] == 1.0
        assert result["query_intent"]["ue_distribution"]["source"] == DistributionSource.PARSED.value

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_different_presets_yield_different_outputs(self, MockParameterChain):
        """Acceptance: same generic query under different presets ⇒ different alpha + distribution."""
        outputs = {}
        for scenario in (ScenarioType.URBAN, ScenarioType.SUBURBAN, ScenarioType.RURAL):
            intent = _query_intent(scenario, ue_distribution=None)
            mock_chain = MagicMock()
            mock_chain.generate.return_value = (_llm_gen_params(), intent)
            MockParameterChain.return_value = mock_chain

            from radp.digital_twin.agentic_mobility.nodes import parameter_agent

            result = parameter_agent.node(_base_state(query_intent=intent.dict()))
            outputs[scenario.value] = (
                result["gen_params"]["alpha"],
                tuple(sorted(result["gen_params"]["ue_class_distribution"].items())),
            )

        assert len({alpha for alpha, _ in outputs.values()}) == 3
        assert len({dist for _, dist in outputs.values()}) == 3
