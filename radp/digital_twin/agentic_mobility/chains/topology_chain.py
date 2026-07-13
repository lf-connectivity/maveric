"""LLM chain for topology parameter generation."""
from radp.digital_twin.agentic_mobility.models.topology_params import TopologyParams
from radp.digital_twin.agentic_mobility.prompts.topology_prompts import (
    TOPOLOGY_SYSTEM_PROMPT,
    get_topology_prompt,
)
from radp.digital_twin.agentic_mobility.utils.llm_client import LLMClient


class TopologyChain:
    """Chain for generating topology parameters using LLM-based generation."""

    def __init__(self):
        """Initialize topology chain."""
        self.llm_client = LLMClient()

    def generate(
        self,
        area_type: str,
        num_ues: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        raw_query: str,
    ) -> TopologyParams:
        """Generate topology parameters using LLM.

        Args:
            area_type: Area type (urban, suburban, rural, highway, mixed)
            num_ues: Total number of UEs in the simulation
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary
            raw_query: Original user query for context

        Returns:
            TopologyParams with LLM-generated parameters

        Raises:
            Exception: If generation fails after retries
        """
        prompt = get_topology_prompt(
            area_type=area_type,
            num_ues=num_ues,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            raw_query=raw_query,
        )

        topology_params = self.llm_client.generate_structured(
            prompt=prompt, output_model=TopologyParams, system_message=TOPOLOGY_SYSTEM_PROMPT
        )

        return topology_params
