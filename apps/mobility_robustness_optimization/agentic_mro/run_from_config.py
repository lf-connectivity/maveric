"""
Run Agentic MRO using configuration from config.yaml

This script loads all settings from config.yaml including API keys,
making it easy to run without command-line arguments.

Usage:
    python run_from_config.py
"""

import yaml
import os
from main import run_agentic_mro


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run Agentic MRO using config.yaml settings."""
    # Load configuration from the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    # Get default provider settings
    default_provider = config.get("default_provider", "groq")
    provider_config = config["providers"][default_provider]

    # Get optimization settings
    opt_config = config.get("optimization", {})

    # Get data paths
    data_config = config.get("data", {})
    csv_path_from_config = data_config.get("input_csv", "notebooks/simulation_data_initial.csv")

    # Resolve CSV path relative to repository root (maveric directory)
    # The script is in apps/mobility_robustness_optimization/agentic_mro/
    # So we go up 3 levels to get to repo root
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    csv_path = os.path.join(repo_root, csv_path_from_config)

    print("="*70)
    print("Running Agentic MRO with config.yaml settings")
    print("="*70)
    print(f"Provider: {default_provider}")
    print(f"Model: {provider_config.get('model')}")
    print(f"CSV: {csv_path_from_config}")
    print(f"Resolved path: {csv_path}")
    print(f"Target Score: {opt_config.get('target_score', 0.80)}")
    print(f"Max Iterations: {opt_config.get('max_iterations', 3)}")
    print("="*70 + "\n")

    # Check if API key is set
    if provider_config.get("api_key") == "PUT_YOUR_GROQ_API_KEY_HERE":
        print("⚠️  WARNING: API key not set in config.yaml!")
        print("Please edit config.yaml and replace 'PUT_YOUR_GROQ_API_KEY_HERE' with your actual Groq API key.")
        return

    # Run optimization
    result = run_agentic_mro(
        csv_path=csv_path,
        llm_config=provider_config,
        target_score=opt_config.get("target_score", 0.80),
        max_iterations=opt_config.get("max_iterations", 3),
        rlf_threshold=opt_config.get("rlf_threshold", -4.0)
    )

    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best Hysteresis: {result['best_hysteresis']:.4f} dB")
    print(f"Best TTT: {result['best_ttt']} ticks")
    print(f"Best Score: {result['best_score']:.4f}")
    print(f"Total Iterations: {result['total_iterations']}")
    print("="*70)


if __name__ == "__main__":
    main()
