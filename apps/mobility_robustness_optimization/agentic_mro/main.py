"""
Main Entry Point for Agentic MRO System

Provides command-line interface and programmatic API for running
the agentic MRO optimization pipeline.

Usage:
    python main.py --csv data.csv --topology topology.csv --provider groq

Or programmatically:
    from agentic_mro.main import run_agentic_mro
    result = run_agentic_mro(csv_path, topology_path, llm_config)
"""

import argparse
import os
import json
import pandas as pd
from typing import Dict, Optional

from state import create_initial_state
from graph import create_agentic_mro_graph


def run_agentic_mro(
    csv_path: str,
    llm_config: Dict,
    target_score: float = 0.80,
    max_iterations: int = 3,
    rlf_threshold: float = -4.0
) -> Dict:
    """
    Run the Agentic MRO optimization pipeline.

    Args:
        csv_path: Path to preprocessed CSV with UE and cell data
        llm_config: LLM provider configuration
        target_score: Target MRO score for early stopping (default: 0.80)
        max_iterations: Maximum optimization iterations (default: 3)
        rlf_threshold: RLF threshold in dB (default: -4.0)

    Returns:
        Dictionary with final output:
        {
            "best_hysteresis": float,
            "best_ttt": int,
            "best_score": float,
            "total_iterations": int,
            "tested_parameters": list
        }

    Example:
        >>> llm_config = {
        ...     "provider": "groq",
        ...     "model": "llama-3.1-70b-versatile",
        ...     "temperature": 0.2
        ... }
        >>> result = run_agentic_mro("simulation_data_initial.csv", llm_config)
        >>> print(f"Optimal hyst: {result['best_hysteresis']}")
    """
    print("\n" + "="*70)
    print(" "*20 + "AGENTIC MRO SYSTEM")
    print("="*70)
    print(f"Input CSV: {csv_path}")
    print(f"LLM Provider: {llm_config.get('provider', 'unknown')}")
    print(f"Target Score: {target_score}")
    print(f"Max Iterations: {max_iterations}")
    print("="*70)

    # Create initial state
    initial_state = create_initial_state(
        llm_config=llm_config,
        input_csv_path=csv_path,
        target_score=target_score,
        max_iterations=max_iterations,
        rlf_threshold=rlf_threshold
    )

    # Create graph
    graph = create_agentic_mro_graph()

    # Run graph
    print("\n🚀 Starting optimization pipeline...\n")
    final_state = graph.invoke(initial_state)

    # Extract final output
    final_output = final_state.get("final_output")

    print("\n" + "="*70)
    print(" "*25 + "OPTIMIZATION COMPLETE")
    print("="*70)

    return final_output


def main():
    """Command-line interface for Agentic MRO."""
    parser = argparse.ArgumentParser(
        description="Agentic MRO: Intelligent Mobility Robustness Optimization"
    )

    # Data arguments
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to preprocessed CSV file with UE and cell data"
    )

    # LLM arguments
    parser.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "bedrock", "openai"],
        help="LLM provider to use (default: groq)"
    )
    parser.add_argument(
        "--model",
        help="Model name (default: provider-specific default)"
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set environment variable)"
    )

    # Optimization arguments
    parser.add_argument(
        "--target-score",
        type=float,
        default=0.80,
        help="Target MRO score for early stopping (default: 0.80)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum optimization iterations (default: 3)"
    )
    parser.add_argument(
        "--rlf-threshold",
        type=float,
        default=-4.0,
        help="RLF threshold in dB (default: -4.0)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        help="Path to save output JSON (optional)"
    )

    args = parser.parse_args()

    # Build LLM config
    llm_config = {
        "provider": args.provider,
        "temperature": 0.2,
        "max_tokens": 2000
    }

    if args.model:
        llm_config["model"] = args.model

    if args.api_key:
        llm_config["api_key"] = args.api_key

    # Run optimization
    result = run_agentic_mro(
        csv_path=args.csv,
        llm_config=llm_config,
        target_score=args.target_score,
        max_iterations=args.max_iterations,
        rlf_threshold=args.rlf_threshold
    )

    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    print(json.dumps(result, indent=2, default=str))
    print("="*70)

    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
