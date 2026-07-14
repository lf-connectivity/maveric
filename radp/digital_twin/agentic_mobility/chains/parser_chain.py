"""LLM chain for query parsing."""
from radp.digital_twin.agentic_mobility.models.query_intent import QueryIntent
from radp.digital_twin.agentic_mobility.prompts.parser_prompts import PARSER_SYSTEM_PROMPT, get_parser_prompt
from radp.digital_twin.agentic_mobility.utils.llm_client import LLMClient


class ParserChain:
    """Chain for parsing natural language queries into QueryIntent."""

    def __init__(self):
        """Initialize parser chain."""
        self.llm_client = LLMClient()

    def parse(self, user_query: str) -> QueryIntent:
        """Parse user query into structured QueryIntent.

        Args:
            user_query: Natural language query from user

        Returns:
            QueryIntent object

        Raises:
            Exception: If parsing fails after retries
        """
        prompt = get_parser_prompt(user_query)

        query_intent = self.llm_client.generate_structured(
            prompt=prompt, output_model=QueryIntent, system_message=PARSER_SYSTEM_PROMPT
        )

        return query_intent
