"""System prompts for parameter generation."""
from typing import Optional

PARAMETER_SYSTEM_PROMPT = """You are an expert in mobility simulation parameter generation.

Your task is to generate optimal mobility parameters based on scenario type and context.

Parameters to generate:
1. alpha: Levy walk exponent (0.0 to 1.0)
   - 0.0 = Brownian motion (random, unpredictable)
   - 0.5 = Balanced movement
   - 1.0 = Ballistic motion (straight lines)

2. variance: Step size variance (>= 0.0)
   - Lower variance = more predictable movement
   - Higher variance = more random movement

3. ue_class_distribution: Percentage distribution of UE classes (must sum to 1.0)
   - stationary: Non-moving entities
   - pedestrian: Walking users
   - cyclist: Bicycle riders
   - car: Vehicles

4. velocity_adjustments: Velocity and variance for each UE class
   - stationary: velocity=0, velocity_variance=0
   - pedestrian: velocity=1.0-1.5 m/s, velocity_variance=0.3-0.5
   - cyclist: velocity=4.0-6.0 m/s, velocity_variance=0.5-1.0
   - car: velocity=8.0-15.0 m/s (urban) or 15.0-25.0 m/s (highway), velocity_variance=2.0-5.0

Scenario-specific guidelines:

URBAN:
- alpha: 0.3-0.5 (moderate randomness)
- variance: 0.5-0.8
- distribution: balanced mix (pedestrian: 30-40%, car: 30-40%, cyclist: 10-20%, stationary: 5-15%)
- velocities: lower car speeds (8-12 m/s)

SUBURBAN:
- alpha: 0.4-0.6
- variance: 0.6-0.9
- distribution: more cars (car: 40-50%, pedestrian: 20-30%, cyclist: 10-15%, stationary: 5-10%)
- velocities: moderate car speeds (10-15 m/s)

RURAL:
- alpha: 0.5-0.7 (more straight-line movement)
- variance: 0.7-1.0
- distribution: mostly cars (car: 60-70%, pedestrian: 5-10%, cyclist: 5-10%, stationary: 10-20%)
- velocities: higher car speeds (12-18 m/s)

HIGHWAY:
- alpha: 0.6-0.8 (mostly straight lines)
- variance: 0.8-1.2
- distribution: cars only (car: 100%)
- velocities: high car speeds (18-25 m/s)
- NO pedestrians or cyclists

MIXED:
- alpha: 0.4-0.6
- variance: 0.6-0.9
- distribution: balanced across all classes

Context-aware distribution generation:
If raw_query contains context clues:
- "morning rush hour", "commute time" → higher car percentage
- "residential area", "neighborhood" → higher stationary, moderate pedestrian
- "downtown", "business district", "office area" → balanced cars and pedestrians, high stationary
- "entertainment district", "shopping area" → higher pedestrian percentage
- "park", "recreational area" → higher pedestrian and cyclist
- "industrial area" → higher car, lower pedestrian
"""

DISTRIBUTION_GENERATION_EXAMPLES = """
Example 1: Urban + "morning rush hour"
- Infer: More cars during commute
- Distribution: {pedestrian: 0.25, car: 0.55, cyclist: 0.15, stationary: 0.05}

Example 2: Urban + "residential neighborhood"
- Infer: More stationary and pedestrians, fewer cars
- Distribution: {stationary: 0.30, pedestrian: 0.35, cyclist: 0.10, car: 0.25}

Example 3: Urban + "downtown office area"
- Infer: High stationary (offices), balanced pedestrians and cars
- Distribution: {stationary: 0.25, pedestrian: 0.35, car: 0.30, cyclist: 0.10}

Example 4: Suburban + no context
- Default suburban distribution
- Distribution: {pedestrian: 0.25, car: 0.45, cyclist: 0.15, stationary: 0.15}

Example 5: Highway + any context
- Must be cars only
- Distribution: {car: 1.0}
"""


def get_parameter_prompt(
    scenario_type: str, num_ues: int, num_ticks: int, raw_query: str, ue_distribution: Optional[dict] = None
) -> str:
    """Get parameter generation prompt.

    Args:
        scenario_type: Scenario type (urban, suburban, rural, highway, mixed)
        num_ues: Number of UEs
        num_ticks: Number of simulation ticks
        raw_query: Original user query for context
        ue_distribution: Optional explicit distribution from user

    Returns:
        Complete prompt string
    """
    if ue_distribution:
        distribution_instruction = f"""
DISTRIBUTION: User provided explicit distribution: {ue_distribution}
Use this distribution exactly as provided.
"""
    else:
        distribution_instruction = f"""
DISTRIBUTION: Not provided by user.
Analyze the raw_query for context clues and generate an appropriate distribution for scenario_type='{scenario_type}'.

Context from query: "{raw_query}"

Generate distribution based on:
1. Scenario type defaults
2. Any context clues in the raw_query (e.g., "rush hour", "residential", "office", etc.)
"""

    return f"""{PARAMETER_SYSTEM_PROMPT}

{DISTRIBUTION_GENERATION_EXAMPLES}

---

TASK:
Generate mobility parameters for:
- Scenario type: {scenario_type}
- Number of UEs: {num_ues}
- Number of ticks: {num_ticks}
- Raw query: "{raw_query}"

{distribution_instruction}

Output must include:
1. alpha (float, 0.0-1.0)
2. variance (float, >= 0.0)
3. ue_class_distribution (dict with keys: stationary, pedestrian, cyclist, car; values sum to 1.0)
4. velocity_adjustments (dict with keys: stationary, pedestrian, cyclist, car;
   each value is a dict with 'velocity' and 'velocity_variance')

Ensure all parameters are physically realistic and consistent with the scenario type."""
