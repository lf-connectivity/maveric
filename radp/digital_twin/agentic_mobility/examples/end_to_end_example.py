"""End-to-end example: Natural language → Mobility simulation.

Supports a --preset {urban,suburban,rural} flag that swaps in a generic
preset-driven query so the chosen scenario's default alpha, variance, and
class distribution are applied deterministically.
"""
import argparse
import json
from pathlib import Path

from radp.digital_twin.agentic_mobility.defaults import PRESETS
from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration


PRESET_QUERIES = {
    "urban": "Generate 50 UEs in an urban environment for 50 ticks",
    "suburban": "Generate 50 UEs in a suburban environment for 50 ticks",
    "rural": "Generate 50 UEs in a rural environment for 50 ticks",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic mobility end-to-end pipeline.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Scenario preset to apply (urban, suburban, or rural).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Custom natural-language query. Overrides the preset's default query if both are given.",
    )
    return parser.parse_args()


def main():
    """Demonstrate complete pipeline from natural language to mobility DataFrame."""
    args = _parse_args()

    if args.query:
        query = args.query
    elif args.preset:
        query = PRESET_QUERIES[args.preset]
    else:
        query = "Create 25 devices in Austin, Texas. Generate for 20 ticks"

    output_dir = Path(__file__).parent / "generated_ues"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Agentic Mobility Generation - End-to-End Example")
    print("Natural Language → Parameters → Simulation → DataFrame")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print("=" * 80)

    print(f"\nQuery: '{query}'")
    print("\nProcessing...")

    df, metadata = AgenticMobilityIntegration.generate_from_natural_language(query)

    print("\n✓ Successfully generated mobility tracks!")
    print(f"  - Number of UEs: {metadata['query_intent']['num_ues']}")
    print(f"  - Number of ticks: {metadata['query_intent']['num_ticks']}")
    print(f"  - Total position points: {len(df)}")
    print(f"  - Retry count: {metadata['retry_count']}")
    print(f"  - Location: {metadata['query_intent']['location']}")
    print(f"  - Scenario type: {metadata['query_intent']['scenario_type']}")

    print("\n  DataFrame preview:")
    print(df.head(10))

    print(f"\n  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}\n")

    print("\n  DataFrame summary:")
    print(df.describe(), end="\n\n")

    print("\n  Metadata:")
    print(json.dumps(metadata, indent=2))

    name_suffix = f"_{args.preset}" if args.preset else ""
    csv_filename = (
        f"agentic_mobility{name_suffix}_"
        f"{metadata['query_intent']['num_ues']}UE_"
        f"{metadata['query_intent']['num_ticks']}ticks.csv"
    )
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)

    metadata_filename = (
        f"agentic_mobility{name_suffix}_"
        f"{metadata['query_intent']['num_ues']}UE_"
        f"{metadata['query_intent']['num_ticks']}ticks_metadata.json"
    )
    metadata_path = output_dir / metadata_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved CSV to: {csv_path}")
    print(f"✓ Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    # Note: Requires .env file with GROQ_API_KEY in radp/digital_twin/agentic_mobility/
    main()
