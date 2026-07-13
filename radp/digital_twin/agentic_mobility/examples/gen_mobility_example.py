"""Basic usage example for Agentic Mobility Generation."""
import json

from radp.digital_twin.agentic_mobility.api import generate_mobility_params


def main():
    """Demonstrate basic usage of agentic mobility generation."""

    print("=" * 80)
    print("Agentic Mobility Generation - Basic Usage Example")
    print("=" * 80)

    # Example 1: Simple urban scenario
    print("\n1. Simple urban scenario:")
    print("-" * 80)
    # query1 = "Generate 100 UEs in urban Tokyo during morning rush hour"
    query1 = "Give me for Dhaka. consider it as an suburban area with lots of pedestrians and cars."
    print(f"Query: {query1}")

    result1 = generate_mobility_params(query1)
    print(f"Status: {result1['status']}")
    print(f"Retries: {result1['metadata']['retry_count']}")
    print(f"Generated parameters:\n{json.dumps(result1['radp_params'], indent=2)[:500]}...")
    print("\n\n")

    print(result1)

    # # Example 2: Suburban with explicit distribution
    # print("\n\n2. Suburban with explicit distribution:")
    # print("-" * 80)
    # query2 = "Create 50 UEs in suburban Chicago with 60% pedestrians, 30% cars, 10% cyclists for 100 time steps"
    # print(f"Query: {query2}")

    # result2 = generate_mobility_params(query2)
    # print(f"Status: {result2['status']}")
    # print(f"Retries: {result2['metadata']['retry_count']}")

    # # Example 3: Highway scenario
    # print("\n\n3. Highway scenario:")
    # print("-" * 80)
    # query3 = "500 UEs on highway I-95 near Boston"
    # print(f"Query: {query3}")

    # result3 = generate_mobility_params(query3)
    # print(f"Status: {result3['status']}")
    # if result3['status'] == 'success_with_warnings':
    #     print(f"Warnings: {result3['warnings']}")
    # print(f"Retries: {result3['metadata']['retry_count']}")

    # # Example 4: Implicit distribution context
    # print("\n\n4. Implicit distribution (residential area):")
    # print("-" * 80)
    # query4 = "Generate 200 mobile users in downtown New York mimicking a residential neighborhood"
    # print(f"Query: {query4}")

    # result4 = generate_mobility_params(query4)
    # print(f"Status: {result4['status']}")
    # print(f"UE Distribution:")
    # ue_dist = result4['radp_params']['ue_tracks_generation']['params']['ue_class_distribution']
    # for ue_class, info in ue_dist.items():
    #     if info['count'] > 0:
    #         print(f"  - {ue_class}: {info['count']} UEs (velocity: {info['velocity']} m/s)")

    # print("\n" + "=" * 80)
    # print("Examples complete!")
    # print("=" * 80)


if __name__ == "__main__":
    # Note: Requires .env file with GROQ_API_KEY in radp/digital_twin/agentic_mobility/
    main()
