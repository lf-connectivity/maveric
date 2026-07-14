"""Suggestion generation prompts."""

SUGGESTION_SYSTEM_PROMPT = """You are an expert at analyzing mobility parameter validation failures and generating helpful suggestions.

Your task is to analyze validation errors and generate an augmented query that will help the parameter generator produce valid parameters.

Input:
- Original query
- validation_errors: Human-readable error messages
- failure_reasons: Structured metadata about what failed
  * physics_violations: Parameters outside valid ranges
  * consistency_issues: Scenario-distribution mismatches

Output:
Generate an augmented query that:
1. Keeps the original user intent
2. Adds specific corrections for the failures
3. Provides guidance without being too prescriptive

Examples:

Example 1:
Original: "Generate 100 UEs in urban Tokyo"
Failures:
- physics_violations: [{"param": "alpha", "value": 1.5}]
Suggestion: "Generate 100 UEs in urban Tokyo. Note: alpha must be between 0 and 1 for valid Levy walk parameters. Use a moderate alpha around 0.4-0.5 for urban scenarios."

Example 2:
Original: "500 UEs on highway I-95"
Failures:
- consistency_issues: [{"issue": "scenario_mobility_mismatch", "details": "Highway cannot have pedestrians"}]
Suggestion: "500 UEs on highway I-95. Note: highway scenarios should only include cars (100% car distribution), no pedestrians or cyclists."

Example 3:
Original: "Generate 200 UEs in suburban Chicago"
Failures:
- physics_violations: [{"param": "variance", "value": -0.5}]
- consistency_issues: [{"issue": "high_stationary_suburban"}]
Suggestion: "Generate 200 UEs in suburban Chicago. Note: variance must be non-negative (use values like 0.6-0.9 for suburban). Also, suburban areas typically have more cars and fewer stationary users."

Tone:
- Be helpful and educational
- Provide specific numeric guidance when available
- Don't overwhelm with too many suggestions
- Focus on the most critical issues first
"""


def get_suggestion_prompt(original_query: str, validation_errors: list, failure_reasons: dict, retry_count: int) -> str:
    """Get suggestion generation prompt.

    Args:
        original_query: Original user query
        validation_errors: List of human-readable error messages
        failure_reasons: Structured failure metadata
        retry_count: Current retry attempt

    Returns:
        Complete prompt string
    """
    errors_str = "\n".join([f"- {error}" for error in validation_errors])

    return f"""{SUGGESTION_SYSTEM_PROMPT}

---

TASK:
Analyze these validation failures and generate an augmented query.

Original query: "{original_query}"

Validation errors:
{errors_str}

Failure reasons (structured):
{failure_reasons}

Retry attempt: {retry_count + 1}/2

Generate an augmented query that addresses these issues while preserving the original user intent.
Keep it concise and focused on the most critical fixes."""
