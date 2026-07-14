"""Unit tests for agentic_mobility pipeline nodes.

Mocks all LLM and geocoding calls — no API keys needed.
"""
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest

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

def _base_state(**overrides) -> MobilityGenerationState:
    """Minimal valid state dict."""
    state: MobilityGenerationState = {
        "user_query": "Generate 50 UEs in Tokyo",
        "current_query": "Generate 50 UEs in Tokyo",
        "query_intent": None,
        "location_data": None,
        "gen_params": None,
        "validation_result": None,
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def _sample_query_intent() -> QueryIntent:
    return QueryIntent(
        scenario_type=ScenarioType.URBAN,
        location="Tokyo",
        num_ues=50,
        num_ticks=50,
        ue_distribution=None,
        raw_query="Generate 50 UEs in Tokyo",
    )


def _sample_gen_params() -> GenParams:
    return GenParams(
        alpha=0.5,
        variance=0.8,
        ue_class_distribution={
            "stationary": 0.20,
            "pedestrian": 0.45,
            "cyclist": 0.10,
            "car": 0.25,
        },
        velocity_adjustments={
            "stationary": {"mean": 0.0, "variance": 0.0},
            "pedestrian": {"mean": 1.4, "variance": 0.3},
            "cyclist": {"mean": 4.0, "variance": 0.5},
            "car": {"mean": 13.9, "variance": 2.0},
        },
    )


# ---------------------------------------------------------------------------
# 1. query_parser node
# ---------------------------------------------------------------------------

class TestQueryParserNode:
    """Tests for nodes/query_parser.py."""

    @patch("radp.digital_twin.agentic_mobility.nodes.query_parser.ParserChain")
    def test_returns_query_intent_dict(self, MockParserChain):
        """Node returns a dict with 'query_intent' key populated."""
        mock_chain = MagicMock()
        mock_chain.parse.return_value = _sample_query_intent()
        MockParserChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import query_parser

        result = query_parser.node(_base_state())

        assert "query_intent" in result
        assert result["query_intent"]["location"] == "Tokyo"
        assert result["query_intent"]["num_ues"] == 50
        mock_chain.parse.assert_called_once_with("Generate 50 UEs in Tokyo")

    @patch("radp.digital_twin.agentic_mobility.nodes.query_parser.ParserChain")
    def test_uses_current_query_not_user_query(self, MockParserChain):
        """Node uses current_query (augmented), not user_query."""
        mock_chain = MagicMock()
        mock_chain.parse.return_value = _sample_query_intent()
        MockParserChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import query_parser

        state = _base_state(current_query="Generate 50 UEs in Tokyo with more pedestrians")
        query_parser.node(state)

        mock_chain.parse.assert_called_once_with("Generate 50 UEs in Tokyo with more pedestrians")


# ---------------------------------------------------------------------------
# 2. location_resolver node
# ---------------------------------------------------------------------------

class TestLocationResolverNode:
    """Tests for nodes/location_resolver.py."""

    @patch("radp.digital_twin.agentic_mobility.nodes.location_resolver.GeocodingService")
    def test_returns_location_data_dict(self, MockGeocodingService):
        """Node returns location_data dict when geocoding succeeds."""
        mock_service = MagicMock()
        mock_location = MagicMock()
        mock_location.dict.return_value = {
            "center": (35.6762, 139.6503),
            "min_lat": 35.5,
            "max_lat": 35.8,
            "min_lon": 139.5,
            "max_lon": 139.8,
            "area_type": "urban",
        }
        mock_service.geocode_location.return_value = mock_location
        MockGeocodingService.return_value = mock_service

        from radp.digital_twin.agentic_mobility.nodes import location_resolver

        state = _base_state(query_intent=_sample_query_intent().dict())
        result = location_resolver.node(state)

        assert "location_data" in result
        assert result["location_data"]["area_type"] == "urban"
        mock_service.geocode_location.assert_called_once_with("Tokyo")

    @patch("radp.digital_twin.agentic_mobility.nodes.location_resolver.GeocodingService")
    def test_fallback_on_geocoding_failure(self, MockGeocodingService):
        """Node uses default bounds when geocoding returns None."""
        mock_service = MagicMock()
        mock_service.geocode_location.return_value = None
        MockGeocodingService.return_value = mock_service

        from radp.digital_twin.agentic_mobility.nodes import location_resolver

        state = _base_state(query_intent=_sample_query_intent().dict())
        result = location_resolver.node(state)

        assert "location_data" in result
        # Fallback defaults
        assert result["location_data"]["center"] == (0.0, 0.0)
        assert result["location_data"]["area_type"] == "urban"

    def test_raises_without_query_intent(self):
        """Node raises ValueError if query_intent is None."""
        from radp.digital_twin.agentic_mobility.nodes import location_resolver

        with pytest.raises(ValueError, match="query_intent must be populated"):
            location_resolver.node(_base_state(query_intent=None))


# ---------------------------------------------------------------------------
# 3. parameter_agent node
# ---------------------------------------------------------------------------

class TestParameterAgentNode:
    """Tests for nodes/parameter_agent.py."""

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_returns_gen_params_dict(self, MockParameterChain):
        """Node returns gen_params and updated query_intent."""
        mock_chain = MagicMock()
        intent = _sample_query_intent()
        mock_chain.generate.return_value = (_sample_gen_params(), intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        state = _base_state(query_intent=intent.dict())
        result = parameter_agent.node(state)

        assert "gen_params" in result
        assert "query_intent" in result
        assert result["gen_params"]["alpha"] == 0.5
        assert result["gen_params"]["variance"] == 0.8

    @patch("radp.digital_twin.agentic_mobility.nodes.parameter_agent.ParameterChain")
    def test_distribution_source_predicted_when_none(self, MockParameterChain):
        """When no distribution in query, gen_params reflects LLM-predicted values."""
        mock_chain = MagicMock()
        intent = _sample_query_intent()
        params = _sample_gen_params()

        # Simulate LLM predicting the distribution
        predicted_intent = intent.copy(update={
            "ue_distribution": UEDistribution(
                stationary=0.20, pedestrian=0.45, cyclist=0.10, car=0.25,
                source=DistributionSource.PREDICTED,
            )
        })
        mock_chain.generate.return_value = (params, predicted_intent)
        MockParameterChain.return_value = mock_chain

        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        state = _base_state(query_intent=intent.dict())
        result = parameter_agent.node(state)

        assert result["query_intent"]["ue_distribution"]["source"] == "predicted"

    def test_raises_without_query_intent(self):
        """Node raises ValueError if query_intent is None."""
        from radp.digital_twin.agentic_mobility.nodes import parameter_agent

        with pytest.raises(ValueError, match="query_intent must be populated"):
            parameter_agent.node(_base_state(query_intent=None))
