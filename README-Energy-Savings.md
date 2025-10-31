# Energy Savings rApp Application

> **Version:** 1.0 | **Date:** October 2, 2025

## Table of Contents

1. **[Overview](#overview)**
2. **[What Energy Savings Does](#what-energy-savings-does-intelligent-network-energy-optimization)**
3. **[Architecture & Components](#architecture--components)**
4. **[Directory Structure](#directory-structure)**
5. **[Prerequisites and Setup](#prerequisites-and-setup)**
6. **[Quick Start Example](#quick-start-example)**
7. **[Step-by-Step Usage Guide](#step-by-step-usage-guide)**


---
> **Prerequisites:** Complete the setup steps in [DEVELOPER-GUIDE.md](DEVELOPER-GUIDE.md) before proceeding.

# **1. Overview**

The Energy Savings rApp is an intelligent network optimization application that uses reinforcement learning to minimize cellular network energy consumption while maintaining service quality. The application optimizes cell tower configurations (ON/OFF states and antenna tilts) based on real-time traffic patterns and user equipment (UE) locations.

The system leverages **Bayesian Digital Twin (BDT)** models for RF prediction and **Proximal Policy Optimization (PPO)** reinforcement learning to learn optimal energy-saving strategies across different time periods and traffic conditions.

---

# **2. What Energy Savings Does: Intelligent Network Energy Optimization**

**Core Function:** Optimizes cellular network energy consumption by intelligently controlling cell tower states and antenna configurations based on traffic patterns.

**Key Logic:**
- **Problem:** Balance energy savings with service quality - turning off cells saves energy but may degrade coverage
- **Solution:** Use RL agent to learn optimal cell ON/OFF states and antenna tilts for different traffic scenarios
- **Method:** PPO reinforcement learning with BDT-based RF simulation
- **Process:** Train BDT for RF prediction → Preprocess UE data → Train RL agent → Predict optimal configurations → Visualize results

**Key Parameters:**
- **Cell State:** ON/OFF control for individual cell towers
- **Antenna Tilt:** Electrical tilt adjustment for coverage optimization
- **Energy Consumption:** Power usage minimization objective
- **Service Quality:** Coverage and capacity maintenance constraints

---

# **3. Architecture & Components**

## System Pipeline

The energy savings application follows a linear pipeline:

```mermaid
flowchart TD
        A[Raw UE Data (Multi-Day)] --> B[Step 1: Preprocess Data]
        B --> C[Gym-Ready UE Data]
        C --> D[Step 2: Train BDT]
        D --> E[BDT Model Pickle]
        E --> F[Step 3: Train RL Agent]
        F --> G[Trained RL Agent (.zip)]
        G --> H[Step 4: Inference]
        H --> I[Console Output (Config Table)]
        I --> J[Step 5: Visualize]
        J --> K[Comparison Plot (.png)]
```

## Core Components

- **`main_app.py`**: Main orchestrator script controlling the entire pipeline
- **`bdt_manager.py`**: Manages BDT model training and Docker communication
- **`data_preprocessor.py`**: Prepares UE data for the Gym environment
- **`rl_trainer.py`**: RL training logic using PPO algorithm
- **`rl_energy_saving_env.py`**: Custom Gymnasium environment for energy optimization
- **`rl_predictor.py`**: Inference using the trained RL agent
- **`energy_saving_visualizer.py`**: Generates comparison plots and visualizations

---

# **4. Directory Structure**

```
apps/energy_savings/
│
├── main_app.py                 # Main orchestrator script
├── bdt_manager.py              # BDT model training and Docker communication
├── data_preprocessor.py        # UE data preprocessing for Gym environment
├── rl_trainer.py               # RL training logic
├── rl_energy_saving_env.py     # Custom Gymnasium environment
├── rl_predictor.py             # Inference using trained RL agent
├── energy_saving_visualizer.py # Visualization and plotting
│
├── data/                       # Static input files
│   ├── topology.csv                # Cell tower layout and configuration
│   ├── config.csv                  # Initial cell tower configuration
│   └── dummy_ue_training_data.csv  # BDT model training data
│
├── generated_data/             # Day-wise UE datasets
│   └── Day_*/
│       ├── ue_data_per_tick/       # Raw UE location data per hour
│       │   ├── generated_ue_data_for_cco_0.csv
│       │   └── ... (up to 23)
│       └── ue_data_gym_ready/      # Preprocessed UE data for RL
│           ├── ue_data_gym_ready_0.csv
│           └── ... (up to 23)
│
└── (Generated Outputs)/
        ├── bdt_model_map.pickle        # Trained BDT model
        ├── energy_saver_agent.zip      # Trained RL agent
        ├── rl_training_logs/           # Training logs and checkpoints
        └── plots/                      # Visualization outputs
```

---

# **5. Prerequisites and Setup**

For complete setup instructions including Python environment, Docker services, and dependencies, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

## Quick Setup Summary

**System Requirements:**
- Python 3.9.x to 3.10.x
- RADP Core libraries installed
- Minimum 8GB RAM (16GB recommended)



**For detailed installation steps, environment setup, and troubleshooting, refer to the [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).**

---


# **6. Quick Start Example**

**Prepare Sample Data:**

```bash
# Install unzip if needed
# Ubuntu/Debian: sudo apt-get install unzip
# macOS: brew install unzip
# Windows: unzip is built-in

# Extract sample data
cd notebooks/data

# Linux/macOS:
unzip energy_saving_data.zip
cp -r energy_saving_data/* ../../apps/energy_savings/

# Windows (PowerShell):
Expand-Archive -Path energy_saving_data.zip -DestinationPath .
Copy-Item -Recurse energy_saving_data/* ../../apps/energy_savings/
```

**Data Generation**

If you need to generate training data, follow these steps:

## Install Dependencies

```bash
pip install -r radp/digital_twin/requirements.txt
pip3 install tensorboard
```

## Generate Traffic Data

```bash
cd radp/digital_twin/traffic_load

# Linux/macOS/WSL:
python traffic_demand_app.py \
  --generate_config_flag \
  --num_sites 5 \
  --cells_per_site 3 \
  --lat_range 240.7 -140 \
  --lon_range -74.05 100 \
  --num_ues 500 \
  --generate_dummy_training_flag \
  --generate_plots_flag \
  --plot_max_ticks 0

# Windows (PowerShell):
python traffic_demand_app.py `
  --generate_config_flag `
  --num_sites 5 `
  --cells_per_site 3 `
  --lat_range 240.7 -140 `
  --lon_range -74.05 100 `
  --num_ues 500 `
  --generate_dummy_training_flag `
  --generate_plots_flag `
  --plot_max_ticks 0
```

## Copy Generated Data

```bash
# Copy generated data to energy savings app
# macOS/Linux
cp -r radp/digital_twin/traffic_load/generated_data/* apps/energy_savings/generated_data/

# Windows
Copy-Item -Recurse radp/digital_twin/traffic_load/generated_data/* apps/energy_savings/generated_data/

# Copy required CSV files to data directory
# macOS/Linux
cp apps/energy_savings/generated_data/topology.csv apps/energy_savings/generated_data/config.csv apps/energy_savings/generated_data/dummy_ue_training_data.csv apps/energy_savings/data/

# Windows
New-Item -ItemType Directory -Force -Path apps/energy_savings/data | Out-Null
Copy-Item apps/energy_savings/generated_data/topology.csv -Destination apps/energy_savings/data/
Copy-Item apps/energy_savings/generated_data/config.csv -Destination apps/energy_savings/data/
Copy-Item apps/energy_savings/generated_data/dummy_ue_training_data.csv -Destination apps/energy_savings/data/
```
Working Example:

```bash
cd apps/energy_savings

# 1. Prepare UE data for training (days 0-2) and testing (day 3)
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3

# 2. Train the BDT model (ensure Docker container is running)
python main_app.py --train-bdt --bdt-model-id "bdt_for_energy_saving" --container "radp_prod-training-1"

# 3. Train the RL agent
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 25000

# 4. Run inference for a specific hour
python main_app.py --infer --tick 3

# 5. Visualize results
python main_app.py --visualize --test-day 3 --tick 3
```

---

# **7. Step-by-Step Usage Guide**

## Step 1: Preprocess UE Data

Prepares raw, per-hour UE location data for simulation.

```bash
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3
```

- **Input:** `generated_data/Day_*/ue_data_per_tick/`
- **Output:** `generated_data/Day_*/ue_data_gym_ready/`

## Step 2: Train Bayesian Digital Twin (BDT)

Trains the RF simulation model using Docker backend service.

**Prerequisites:** Docker container must be running.

```bash
python main_app.py --train-bdt --bdt-model-id "bdt_for_energy_saving" --container "radp_prod-training-1"
```

- **Inputs:** `data/topology.csv`, `data/dummy_ue_training_data.csv`
- **Output:** `bdt_model_map.pickle`

## Step 3: Train RL Energy Saving Agent

Trains the PPO agent using preprocessed data and BDT model.

```bash
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 25000
```

- **Inputs:** `bdt_model_map.pickle`, `generated_data/Day_*/ue_data_gym_ready/`, `data/topology.csv`, `data/config.csv`
- **Outputs:** `energy_saver_agent.zip`, `rl_training_logs/`

## Step 4: Run Inference

Uses trained agent to predict optimal network configuration.

```bash
python main_app.py --infer --tick <T>
```

- **Inputs:** `energy_saver_agent.zip`, `data/topology.csv`
- **Output:** Console table of predicted optimal state (ON/OFF, tilt) for each cell

## Step 5: Visualize Results

Generates comparison plots showing optimization impact.

```bash
python main_app.py --visualize --test-day <D> --tick <T>
```

- **Inputs:** `energy_saver_agent.zip`, `bdt_model_map.pickle`, `data/topology.csv`, `data/config.csv`, `generated_data/Day_<D>/ue_data_gym_ready/`
- **Output:** `.png` image in `plots/` directory

---

**For questions or contributions, please refer to the main project documentation.**
