"""Conditional routing logic for LangGraph."""
from radp.digital_twin.agentic_mobility.config import Config
from radp.digital_twin.agentic_mobility.models.state import MobilityGenerationState


def should_retry(state: MobilityGenerationState) -> str:
    """
    Determine next step after validation.

    Args:
        state: Current graph state

    Returns:
        "success" - validation passed, proceed to END
        "retry" - validation failed, generate suggestion and retry
        "max_retries" - exhausted retries, accept last params (POC behavior)
    """
    validation_result = state.get("validation_result")

    if not validation_result:
        # Should not happen, but default to retry
        return "retry"

    if validation_result["status"] == "success":
        return "success"
    elif state["retry_count"] >= Config.MAX_RETRY_ATTEMPTS:
        return "max_retries"  # Accept parameters despite validation warnings
    else:
        return "retry"
