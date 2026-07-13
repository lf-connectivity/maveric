"""LLM chain for generating suggestions from validation failures."""
from typing import Dict, List

from radp.digital_twin.agentic_mobility.prompts.suggestion_prompts import (
    SUGGESTION_SYSTEM_PROMPT,
    get_suggestion_prompt,
)
from radp.digital_twin.agentic_mobility.utils.llm_client import LLMClient


class SuggestionChain:
    """Chain for generating query suggestions from validation failures."""

    def __init__(self):
        """Initialize suggestion chain."""
        self.llm_client = LLMClient()

    def generate_suggestion(
        self, original_query: str, validation_errors: List[str], failure_reasons: Dict, retry_count: int
    ) -> str:
        """Generate augmented query with suggestions.

        Args:
            original_query: Original user query
            validation_errors: List of human-readable error messages
            failure_reasons: Structured failure metadata
            retry_count: Current retry attempt number

        Returns:
            Augmented query string with suggestions

        Raises:
            Exception: If suggestion generation fails after retries
        """
        prompt = get_suggestion_prompt(
            original_query=original_query,
            validation_errors=validation_errors,
            failure_reasons=failure_reasons,
            retry_count=retry_count,
        )

        augmented_query = self.llm_client.generate_text(prompt=prompt, system_message=SUGGESTION_SYSTEM_PROMPT)

        return augmented_query
