"""System prompts and examples for query parsing."""

PARSER_SYSTEM_PROMPT = """You are an expert at parsing natural language queries about mobility simulation scenarios.

Extract the following information from user queries:
- scenario_type: urban, suburban, rural, highway, or mixed
- location: Geographic location name
- num_ues: Number of User Equipment (UEs) to generate (default: guess a logical value considering the scenario type and location if not specified)
- num_ticks: Number of simulation time steps (default: 50 if not specified)
- ue_distribution: Optional distribution of UE classes (stationary, pedestrian, cyclist, car) as percentages
- raw_query: The original user query

Valid UE classes:
- stationary: Non-moving entities
- pedestrian: Walking users (0.5-2 m/s)
- cyclist: Bicycle riders (3-8 m/s)
- car: Vehicles (5-30 m/s)

If the user specifies distribution:
- Explicit: "60% pedestrians, 30% cars, 10% cyclists" → extract exact percentages with source="parsed"
  Format: {"distribution": {"pedestrian": 0.6, "car": 0.3, "cyclist": 0.1}, "source": "parsed"}
- Implicit: "mimicking a residential area" → leave ue_distribution as null (will be inferred later with source="predicted")
- Not specified → leave ue_distribution as null

Scenario type inference:
- "city", "downtown", "metropolitan" → urban
- "residential", "suburb" → suburban
- "countryside", "farmland" → rural
- "freeway", "motorway", "interstate" → highway
- "mixed environment" → mixed
"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "Generate 100 UEs in urban Tokyo during morning rush hour",
        "output": {
            "scenario_type": "urban",
            "location": "Tokyo",
            "num_ues": 100,
            "num_ticks": 50,
            "ue_distribution": None,
            "raw_query": "Generate 100 UEs in urban Tokyo during morning rush hour",
        },
    },
    {
        "query": "Create 50 UEs in suburban Chicago with 60% pedestrians, 30% cars, 10% cyclists for 100 time steps",
        "output": {
            "scenario_type": "suburban",
            "location": "Chicago",
            "num_ues": 50,
            "num_ticks": 100,
            "ue_distribution": {"pedestrian": 0.6, "car": 0.3, "cyclist": 0.1, "stationary": 0.0, "source": "parsed"},
            "raw_query": "Create 50 UEs in suburban Chicago with 60% pedestrians, 30% cars, 10% cyclists for 100 time steps",
        },
    },
    {
        "query": "Generate 200 mobile users in downtown New York mimicking an office area",
        "output": {
            "scenario_type": "urban",
            "location": "New York",
            "num_ues": 200,
            "num_ticks": 50,
            "ue_distribution": None,
            "raw_query": "Generate 200 mobile users in downtown New York mimicking an office area",
        },
    },
    {
        "query": "500 UEs on highway I-95 near Boston",
        "output": {
            "scenario_type": "highway",
            "location": "Boston, I-95",
            "num_ues": 500,
            "num_ticks": 50,
            "ue_distribution": None,
            "raw_query": "500 UEs on highway I-95 near Boston",
        },
    },
    {
        "query": "Rural scenario in Kansas with 30 UEs, mostly stationary and cars",
        "output": {
            "scenario_type": "rural",
            "location": "Kansas",
            "num_ues": 30,
            "num_ticks": 50,
            "ue_distribution": None,
            "raw_query": "Rural scenario in Kansas with 30 UEs, mostly stationary and cars",
        },
    },
]


def get_parser_prompt(user_query: str) -> str:
    """Get parser prompt with few-shot examples.

    Args:
        user_query: User's natural language query

    Returns:
        Complete prompt string
    """
    examples_str = "\n\n".join(
        [f"Query: {ex['query']}\nOutput: {ex['output']}" for ex in FEW_SHOT_EXAMPLES[:3]]  # Use first 3 examples
    )

    return f"""{PARSER_SYSTEM_PROMPT}

Examples:
{examples_str}

Now parse this query:
{user_query}"""
