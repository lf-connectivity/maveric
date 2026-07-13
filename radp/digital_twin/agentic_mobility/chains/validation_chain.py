"""Validation chain for parameter checking."""
from typing import Dict

from radp.digital_twin.agentic_mobility.models.generation_params import GenParams
from radp.digital_twin.agentic_mobility.models.query_intent import QueryIntent
from radp.digital_twin.agentic_mobility.utils.validators import (
    check_scenario_consistency,
    validate_alpha,
    validate_distribution,
    validate_variance,
    validate_velocity,
)


class ValidationChain:
    """Chain for validating generated parameters.

    Performs two-stage validation:
    1. Physics validation (critical)
    2. Consistency validation (warnings)
    """

    def validate(self, gen_params: GenParams, query_intent: QueryIntent) -> Dict:
        """Validate generated parameters.

        Args:
            gen_params: Generated parameters to validate
            query_intent: Original query intent for consistency checking

        Returns:
            Dict with validation result:
            {
                "status": "success" | "failed",
                "validated_params": {...},  # if success
                "validation_errors": [...],  # if failed
                "failure_reasons": {...}     # if failed
            }
        """
        validation_errors = []
        physics_violations = []
        consistency_issues = []

        # 1. Physics Validation (CRITICAL)

        # Validate alpha
        alpha_valid, alpha_error = validate_alpha(gen_params.alpha)
        if not alpha_valid:
            validation_errors.append(alpha_error)
            physics_violations.append(
                {"param": "alpha", "value": gen_params.alpha, "allowed_range": [0.0, 1.0], "severity": "critical"}
            )

        # Validate variance
        variance_valid, variance_error = validate_variance(gen_params.variance)
        if not variance_valid:
            validation_errors.append(variance_error)
            physics_violations.append(
                {
                    "param": "variance",
                    "value": gen_params.variance,
                    "allowed_range": [0.0, float("inf")],
                    "severity": "critical",
                }
            )

        # Validate distribution
        dist_valid, dist_error = validate_distribution(gen_params.ue_class_distribution)
        if not dist_valid:
            validation_errors.append(dist_error)
            physics_violations.append(
                {
                    "param": "ue_class_distribution",
                    "value": sum(gen_params.ue_class_distribution.values()),
                    "allowed_range": [1.0, 1.0],
                    "severity": "critical",
                }
            )

        # Validate velocities
        for ue_class, adjustments in gen_params.velocity_adjustments.items():
            velocity = adjustments.get("velocity", 0)
            velocity_valid, velocity_error = validate_velocity(ue_class, velocity)
            if not velocity_valid:
                validation_errors.append(velocity_error)
                physics_violations.append(
                    {
                        "param": f"velocity_{ue_class}",
                        "value": velocity,
                        "allowed_range": "see velocity ranges",
                        "severity": "critical",
                    }
                )

            # Validate velocity_variance is non-negative
            velocity_variance = adjustments.get("velocity_variance", 0)
            if velocity_variance < 0:
                error_msg = f"Velocity variance for {ue_class} must be non-negative (got {velocity_variance})"
                validation_errors.append(error_msg)
                physics_violations.append(
                    {
                        "param": f"velocity_variance_{ue_class}",
                        "value": velocity_variance,
                        "allowed_range": [0.0, float("inf")],
                        "severity": "critical",
                    }
                )

        # 2. Consistency Validation (WARNINGS)
        scenario_warnings = check_scenario_consistency(
            query_intent.scenario_type.value, gen_params.ue_class_distribution
        )
        for warning in scenario_warnings:
            validation_errors.append(warning)
            consistency_issues.append(
                {"issue": "scenario_mobility_mismatch", "details": warning, "severity": "warning"}
            )

        # Determine overall result
        if physics_violations:
            # Critical failures
            return {
                "status": "failed",
                "validation_errors": validation_errors,
                "failure_reasons": {
                    "physics_violations": physics_violations,
                    "consistency_issues": consistency_issues,
                    "severity": "critical",
                },
            }
        elif consistency_issues:
            # Only warnings - still considered failed for retry
            return {
                "status": "failed",
                "validation_errors": validation_errors,
                "failure_reasons": {
                    "physics_violations": [],
                    "consistency_issues": consistency_issues,
                    "severity": "warning",
                },
            }
        else:
            # All validations passed
            return {"status": "success", "validated_params": gen_params.dict()}
