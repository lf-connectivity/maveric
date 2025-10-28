"""Main entry point for agentic mobility generation."""
from typing import Dict

from radp.digital_twin.agentic_mobility.config import Config
from radp.digital_twin.agentic_mobility.formatters.radp_formatter import RADPFormatter
from radp.digital_twin.agentic_mobility.graph.workflow import create_workflow
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState

# Validate configuration on import
Config.validate()


class AgenticMobilityGenerator:
    """High-level API for generating mobility parameters from natural language."""

    def __init__(self):
        """Initialize the workflow."""
        self.workflow = create_workflow()
        self.formatter = RADPFormatter()

    def generate_from_query(self, user_query: str) -> Dict:
        """
        Generate RADP mobility parameters from natural language query.

        Args:
            user_query: Natural language description of mobility scenario

        Returns:
            Dictionary with:
                - status: "success" or "success_with_warnings"
                - radp_params: UETracksGenerationParams format (if success)
                - warnings: Validation warnings (if success_with_warnings)
                - failure_reasons: Structured failure metadata (if success_with_warnings)
                - metadata: Generation metadata (retry_count, etc.)
        """
        # Initialize state
        initial_state: MobilityGenerationState = {
            "user_query": user_query,
            "current_query": user_query,
            "query_intent": None,
            "location_data": None,
            "gen_params": None,
            "validation_result": None,
            "retry_count": 0,
        }

        # Run workflow
        result = self.workflow.invoke(initial_state)

        # Format results
        if result["validation_result"]["status"] == "success":
            # Success case
            radp_params = self.formatter.format_to_radp(
                gen_params=result["gen_params"],
                location_data=result["location_data"],
                query_intent=result["query_intent"],
            )

            return {
                "status": "success",
                "radp_params": radp_params,
                "metadata": {
                    "retry_count": result["retry_count"],
                    "query_intent": result["query_intent"],
                    "location_data": result["location_data"],
                    "validation_warnings": None,
                },
            }
        else:
            # Max retries reached - accept parameters with warnings (POC behavior)
            radp_params = self.formatter.format_to_radp(
                gen_params=result["gen_params"],
                location_data=result["location_data"],
                query_intent=result["query_intent"],
            )

            return {
                "status": "success_with_warnings",
                "radp_params": radp_params,
                "warnings": result["validation_result"].get("validation_errors", []),
                "failure_reasons": result["validation_result"].get("failure_reasons", {}),
                "metadata": {
                    "retry_count": result["retry_count"],
                    "query_intent": result["query_intent"],
                    "location_data": result["location_data"],
                    "validation_status": "failed_but_accepted",
                },
            }

    def visualize_graph(self, output_path: str = "workflow_graph.png"):
        """Generate visualization of the workflow graph.

        Args:
            output_path: Path to save the graph visualization
        """
        try:
            from langgraph.graph import draw_mermaid

            draw_mermaid(self.workflow, output_path=output_path)
            print(f"Workflow graph saved to {output_path}")
        except ImportError:
            print("Graph visualization requires additional dependencies")


# Convenience function for direct usage
def generate_mobility_params(query: str) -> Dict:
    """
    Generate mobility parameters from natural language query.

    Example:
        >>> params = generate_mobility_params(
        ...     "Generate 100 UEs in urban Tokyo during morning rush hour"
        ... )
        >>> print(params["status"])
        'success'

    Args:
        query: Natural language query

    Returns:
        Dictionary with status, radp_params, and metadata
    """
    generator = AgenticMobilityGenerator()
    return generator.generate_from_query(query)
