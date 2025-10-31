"""
Run Agentic MRO using configuration from config.yaml

This script loads all settings from config.yaml including API keys,
making it easy to run without command-line arguments.

Usage:
    python run_from_config.py
"""

import os

import pandas as pd
import yaml
from main import run_agentic_mro

from apps.mobility_robustness_optimization.agentic_mro.utils.visuals import (
    add_sinr_column,
    mro_plot_scatter,
    plot_sinr_db_by_ue,
)
from notebooks.radp_library import preprocess_ue_data
from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt
from radp.digital_twin.utils.constants import RLF_THRESHOLD


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run Agentic MRO using config.yaml settings."""
    # Load configuration from the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    config = load_config(config_path)

    # Get default provider settings
    # Priority: 1. LLM_PROVIDER env var, 2. config.yaml default_provider
    default_provider = os.getenv("LLM_PROVIDER") or config.get("default_provider", "groq")
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

    print("=" * 70)
    print("Running Agentic MRO with config.yaml settings")
    print("=" * 70)
    print(f"Provider: {default_provider}")
    print(f"Model: {provider_config.get('model')}")
    print(f"CSV: {csv_path_from_config}")
    print(f"Resolved path: {csv_path}")
    print(f"Max Iterations: {opt_config.get('max_iterations', 3)}")
    print("=" * 70 + "\n")

    # Check if API key is set
    if provider_config.get("api_key") == "PUT_YOUR_GROQ_API_KEY_HERE":
        print("⚠️  WARNING: API key not set in config.yaml!")
        print("Please edit config.yaml and replace 'PUT_YOUR_GROQ_API_KEY_HERE' with your actual Groq API key.")
        return

    print("=" * 70)
    print("Preprocessing Data...")
    print("=" * 70)

    opt_config["rlf_threshold"] = RLF_THRESHOLD

    ue_data = pd.read_csv(csv_path)
    topology = pd.read_csv(config["data"]["topology_csv"])

    ue_data = ue_data.rename(columns={"lat": "latitude", "lon": "longitude"})

    sim_data = preprocess_ue_data(ue_data, topology)
    sim_data = sim_data.rename(
        columns={"longitude": "loc_x", "latitude": "loc_y", "mock_ue_id": "ue_id", "cell_rxpwr_dbm": "cell_rxpower_dbm"}
    )

    sim_data = add_sinr_column(sim_data)
    sim_data.to_csv("apps/mobility_robustness_optimization/agentic_mro/data/sim_data.csv", index=False)
    print(sim_data.head(10))


    ticks = len(sim_data["tick"].unique())
    # Assuming each tick represents 50ms (this value may need to be adjusted based on actual data characteristics)
    tick_duration_seconds = 1  # 1 second per tick
    T = ticks * tick_duration_seconds
    opt_config["target_score"] = T

    print("Data Preprocessing Complete.")
    print("=" * 70)

    # Run optimization
    result = run_agentic_mro(
        csv_path="apps/mobility_robustness_optimization/agentic_mro/data/sim_data.csv",
        llm_config=provider_config,
        target_score=opt_config.get("target_score", 0.80),
        max_iterations=opt_config.get("max_iterations", 3),
        rlf_threshold=opt_config.get("rlf_threshold", -4.0),
    )

    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best Hysteresis: {result['best_hysteresis']:.4f} dB")
    print(f"Best TTT: {result['best_ttt']} ticks")
    print(f"Best Score: {result['best_score']:.4f}")
    print(f"Total Iterations: {result['total_iterations']}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Attempting Visualization")
    print("=" * 70)

    optimal_data = perform_attachment_hyst_ttt(
        sim_data, hyst=result["best_hysteresis"], ttt=result["best_ttt"], rlf_threshold=RLF_THRESHOLD
    )

    mro_plot_scatter(
        optimal_data, topology, save_path="apps/mobility_robustness_optimization/agentic_mro/plots/mro_plot_scatter.png"
    )

    unique_ue_ids = optimal_data["ue_id"].unique()
    for ue_id in unique_ue_ids:
        plot_sinr_db_by_ue(
            optimal_data,
            sim_data,
            ue_id,
            save_path=f"apps/mobility_robustness_optimization/agentic_mro/plots/sinr_db_by_ue_{ue_id}.png",
        )


if __name__ == "__main__":
    main()
