"""End-to-end example: Natural language → Mobility simulation."""
import json
from pathlib import Path

from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration


def main():
    """Demonstrate complete pipeline from natural language to mobility DataFrame."""

    # Create output directory for generated UEs
    output_dir = Path(__file__).parent / "generated_ues"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Agentic Mobility Generation - End-to-End Example")
    print("Natural Language → Parameters → Simulation → DataFrame")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)

    # Example 1: Urban scenario
    print("\nExample 1: Urban scenario")
    print("-" * 80)

    query1 = "Create 25 devices in Austin, Texas. Generate for 20 ticks"
    print(f"Query: '{query1}'")
    print("\nProcessing...")

    df1, metadata1 = AgenticMobilityIntegration.generate_from_natural_language(query1)

    print("\n✓ Successfully generated mobility tracks!")
    print(f"  - Number of UEs: {metadata1['query_intent']['num_ues']}")
    print(f"  - Number of ticks: {metadata1['query_intent']['num_ticks']}")
    print(f"  - Total position points: {len(df1)}")
    print(f"  - Retry count: {metadata1['retry_count']}")
    print(f"  - Location: {metadata1['query_intent']['location']}")
    print(f"  - Scenario type: {metadata1['query_intent']['scenario_type']}")

    print("\n  DataFrame preview:")
    print(df1.head(10))

    print(f"\n  DataFrame shape: {df1.shape}")
    print(f"  Columns: {list(df1.columns)}\n")

    print("\n  DataFrame summary:")
    print(df1.describe(), end="\n\n")

    print("\n  Metadata:")
    print(json.dumps(metadata1, indent=2))

    # Save CSV to generated_ues directory
    csv_filename = (
        f"agentic_mobility_{metadata1['query_intent']['num_ues']}UE_{metadata1['query_intent']['num_ticks']}ticks.csv"
    )
    csv_path = output_dir / csv_filename
    df1.to_csv(csv_path, index=False)

    # Save metadata as JSON
    metadata_filename = f"agentic_mobility_{metadata1['query_intent']['num_ues']}UE_{metadata1['query_intent']['num_ticks']}ticks_metadata.json"
    metadata_path = output_dir / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata1, f, indent=2)

    print(f"\n✓ Saved CSV to: {csv_path}")
    print(f"✓ Saved metadata to: {metadata_path}")

    # # Example 2: Highway scenario
    # print("\n\n2. Highway Scenario: I-95 Boston")
    # print("-" * 80)

    # query2 = "Generate 50 UEs on highway I-95 near Boston"
    # print(f"Query: '{query2}'")
    # print("\nProcessing...")

    # df2, metadata2 = AgenticMobilityIntegration.generate_from_natural_language(query2)

    # print(f"\n✓ Successfully generated mobility tracks!")
    # print(f"  - Number of UEs: {metadata2['query_intent']['num_ues']}")
    # print(f"  - Total position points: {len(df2)}")
    # print(f"  - Location: {metadata2['query_intent']['location']}")

    # # Example 3: Parameters only (no simulation)
    # print("\n\n3. Parameters Generation Only (No Simulation)")
    # print("-" * 80)

    # query3 = "Generate 100 UEs in suburban Chicago"
    # print(f"Query: '{query3}'")
    # print("\nGenerating parameters...")

    # params3, metadata3 = AgenticMobilityIntegration.generate_params_only(query3)

    # print(f"\n✓ Parameters generated successfully!")
    # print("\n  Generated Parameters (excerpt):")
    # print(json.dumps(params3["ue_tracks_generation"]["params"], indent=2)[:500] + "...")

    # print("\n" + "=" * 80)
    # print("Examples complete!")
    # print("=" * 80)
    # print("\nKey Takeaways:")
    # print("  1. Natural language queries are automatically converted to valid parameters")
    # print("  2. The system generates realistic mobility distributions based on context")
    # print("  3. Full simulation produces pandas DataFrames ready for analysis")
    # print("  4. You can generate parameters only for inspection/modification")


if __name__ == "__main__":
    # Note: Requires .env file with GROQ_API_KEY in radp/digital_twin/agentic_mobility/
    main()
