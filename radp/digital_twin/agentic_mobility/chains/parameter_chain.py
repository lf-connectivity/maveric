"""LLM chain for parameter generation."""
from typing import Tuple

from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import DistributionSource, QueryIntent, UEDistribution
from radp.digital_twin.agentic_mobility.prompts.parameter_prompts import PARAMETER_SYSTEM_PROMPT, get_parameter_prompt
from radp.digital_twin.agentic_mobility.utils.llm_client import LLMClient


class ParameterChain:
    """Chain for generating mobility parameters using LLM-based direct generation.

    NO TOOLS - uses single LLM call to generate all parameters.
    """

    def __init__(self):
        """Initialize parameter chain."""
        self.llm_client = LLMClient()

    def generate(self, query_intent: QueryIntent) -> Tuple[GenParams, QueryIntent]:
        """Generate mobility parameters from QueryIntent.

        Args:
            query_intent: Parsed query intent

        Returns:
            Tuple of (GenParams, updated QueryIntent with distribution source)

        Raises:
            Exception: If generation fails after retries
        """
        # Convert ue_distribution from UEDistribution to dict if provided
        ue_dist_dict = None
        if query_intent.ue_distribution:
            ue_dist_dict = {
                "stationary": query_intent.ue_distribution.stationary,
                "pedestrian": query_intent.ue_distribution.pedestrian,
                "cyclist": query_intent.ue_distribution.cyclist,
                "car": query_intent.ue_distribution.car,
            }

        prompt = get_parameter_prompt(
            scenario_type=query_intent.scenario_type.value,
            num_ues=query_intent.num_ues,
            num_ticks=query_intent.num_ticks,
            raw_query=query_intent.raw_query,
            ue_distribution=ue_dist_dict,
        )

        gen_params = self.llm_client.generate_structured(
            prompt=prompt, output_model=GenParams, system_message=PARAMETER_SYSTEM_PROMPT
        )

        # Update QueryIntent with predicted distribution if it was None
        updated_query_intent = query_intent
        if query_intent.ue_distribution is None:
            # LLM generated the distribution - mark as "predicted"
            updated_query_intent = query_intent.copy(
                update={
                    "ue_distribution": UEDistribution(
                        stationary=gen_params.ue_class_distribution.get("stationary", 0.0),
                        pedestrian=gen_params.ue_class_distribution.get("pedestrian", 0.0),
                        cyclist=gen_params.ue_class_distribution.get("cyclist", 0.0),
                        car=gen_params.ue_class_distribution.get("car", 0.0),
                        source=DistributionSource.PREDICTED,
                    )
                }
            )

        return gen_params, updated_query_intent
