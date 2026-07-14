"""Helper validation functions."""
from typing import Dict, List, Tuple

from radp.digital_twin.agentic_mobility.defaults.parameter_ranges import (
    ALPHA_RANGE,
    SIMULATION_CONSTRAINTS,
    VARIANCE_RANGE,
    VELOCITY_RANGES,
)


def validate_alpha(alpha: float) -> Tuple[bool, str]:
    """Validate alpha parameter.

    Returns:
        (is_valid, error_message)
    """
    if alpha < ALPHA_RANGE["min"] or alpha > ALPHA_RANGE["max"]:
        return False, f"Alpha value {alpha} is outside valid range [{ALPHA_RANGE['min']}, {ALPHA_RANGE['max']}]"
    return True, ""


def validate_variance(variance: float) -> Tuple[bool, str]:
    """Validate variance parameter.

    Returns:
        (is_valid, error_message)
    """
    if variance < VARIANCE_RANGE["min"]:
        return False, f"Variance value {variance} must be non-negative"
    return True, ""


def validate_velocity(mobility_class: str, velocity: float) -> Tuple[bool, str]:
    """Validate velocity for a mobility class.

    Returns:
        (is_valid, error_message)
    """
    if mobility_class not in VELOCITY_RANGES:
        return False, f"Unknown mobility class: {mobility_class}"

    range_info = VELOCITY_RANGES[mobility_class]
    if velocity < range_info["min"] or velocity > range_info["max"]:
        return (
            False,
            f"Velocity {velocity} m/s for {mobility_class} is outside valid range [{range_info['min']}, {range_info['max']}]",
        )
    return True, ""


def validate_num_ues(num_ues: int) -> Tuple[bool, str]:
    """Validate number of UEs.

    Returns:
        (is_valid, error_message)
    """
    if num_ues < SIMULATION_CONSTRAINTS["min_ues"] or num_ues > SIMULATION_CONSTRAINTS["max_ues"]:
        return (
            False,
            f"Number of UEs {num_ues} is outside valid range [{SIMULATION_CONSTRAINTS['min_ues']}, {SIMULATION_CONSTRAINTS['max_ues']}]",
        )
    return True, ""


def validate_num_ticks(num_ticks: int) -> Tuple[bool, str]:
    """Validate number of ticks.

    Returns:
        (is_valid, error_message)
    """
    if num_ticks < SIMULATION_CONSTRAINTS["min_ticks"] or num_ticks > SIMULATION_CONSTRAINTS["max_ticks"]:
        return (
            False,
            f"Number of ticks {num_ticks} is outside valid range [{SIMULATION_CONSTRAINTS['min_ticks']}, {SIMULATION_CONSTRAINTS['max_ticks']}]",
        )
    return True, ""


def validate_distribution(distribution: Dict[str, float]) -> Tuple[bool, str]:
    """Validate UE distribution percentages.

    Returns:
        (is_valid, error_message)
    """
    total = sum(distribution.values())
    if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
        return False, f"Distribution percentages must sum to 1.0 (got {total})"

    for class_name, percentage in distribution.items():
        if percentage < 0 or percentage > 1:
            return False, f"Distribution percentage for {class_name} must be between 0 and 1 (got {percentage})"

    return True, ""


def check_scenario_consistency(scenario_type: str, distribution: Dict[str, float]) -> List[str]:
    """Check consistency between scenario type and UE distribution.

    Returns:
        List of warning messages (empty if consistent)
    """
    warnings = []

    if scenario_type == "highway":
        if distribution.get("pedestrian", 0) > 0:
            warnings.append("Highway scenarios typically should not include pedestrians")
        if distribution.get("cyclist", 0) > 0:
            warnings.append("Highway scenarios typically should not include cyclists")

    if scenario_type == "rural":
        if distribution.get("stationary", 0) > 0.5:
            warnings.append("Rural scenarios typically have lower stationary UE percentages")

    return warnings
