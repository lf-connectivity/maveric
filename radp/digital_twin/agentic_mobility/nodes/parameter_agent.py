"""LangGraph node for parameter generation."""
from typing import Dict, Tuple

from radp.digital_twin.agentic_mobility.chains.parameter_chain import ParameterChain
from radp.digital_twin.agentic_mobility.defaults import get_preset
from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import (
    DistributionSource,
    QueryIntent,
    UEDistribution,
)
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def node(state: MobilityGenerationState) -> Dict:
    """Parameter agent node - generates mobility parameters using LLM.

    When the parsed scenario_type matches a known preset (urban/suburban/rural)
    AND the user did not provide an explicit UE distribution, the preset
    locks in alpha, variance, and class distribution so runs are deterministic
    and distinguishable across scenarios. Explicit user-provided distributions
    are preserved untouched.

    Args:
        state: Current graph state

    Returns:
        Dict with gen_params and updated query_intent
    """
    query_intent_dict = state["query_intent"]

    if not query_intent_dict:
        raise ValueError("query_intent must be populated before parameter generation")

    query_intent = QueryIntent(**query_intent_dict)

    parameter_chain = ParameterChain()
    gen_params, updated_query_intent = parameter_chain.generate(query_intent)

    preset = get_preset(query_intent.scenario_type.value)
    if preset is not None and query_intent.ue_distribution is None:
        gen_params, updated_query_intent = _apply_preset(gen_params, updated_query_intent, preset)

    return {"gen_params": gen_params.dict(), "query_intent": updated_query_intent.dict()}


def _apply_preset(
    gen_params: GenParams, query_intent: QueryIntent, preset: Dict
) -> Tuple[GenParams, QueryIntent]:
    """Override LLM-inferred params with preset values for a matching scenario.

    Only invoked when the user did not specify a distribution — taking that
    absence as a signal that the preset should drive deterministic defaults.
    """
    distribution = dict(preset["default_distribution"])

    gen_params = gen_params.copy(
        update={
            "alpha": preset["alpha"],
            "variance": preset["variance"],
            "ue_class_distribution": distribution,
        }
    )

    query_intent = query_intent.copy(
        update={
            "ue_distribution": UEDistribution(
                stationary=distribution["stationary"],
                pedestrian=distribution["pedestrian"],
                cyclist=distribution["cyclist"],
                car=distribution["car"],
                source=DistributionSource.PREDICTED,
            )
        }
    )

    return gen_params, query_intent
