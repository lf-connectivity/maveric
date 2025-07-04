# Energy Saving Application using Reinforcement Learning

**Version:** 1.0  
**Date:** June 20, 2025

---

## 1. Overview

This application is a comprehensive, modular pipeline designed to train and deploy a Reinforcement Learning (RL) agent for dynamic cellular network optimization. The core objective is to significantly reduce power consumption by intelligently turning cell sectors on/off and adjusting their antenna tilts based on the time of day, while maintaining network coverage and Quality of Service (QoS).

The system leverages a **Bayesian Digital Twin (BDT)** RF model, enabling the RL training loop to perform rapid, local RF simulations. This decouples the agent's learning process from backend latency, allowing efficient training on multi-day traffic patterns. The workflow is orchestrated through `main_app.py` with a clear command-line interface.

---

## 2. Key Features

- **Modular Pipeline:** Each major stage—data preprocessing, BDT model management, RL training, inference, and visualization—is encapsulated in its own Python module for clarity and maintainability.
- **Time-Aware Energy Saving:** The RL agent learns a policy dependent on the hour of the day (tick 0-23), with actions including setting antenna tilts (`cell_el_deg`) and toggling cell sectors ON/OFF.
- **Multi-Day Training & Testing:** Uses distinct datasets for training and testing, allowing the agent to learn from recurring daily patterns and be evaluated on unseen data.
- **Local BDT-based Simulation:** The RL environment uses a pre-trained BDT model to run local RF simulations, providing immediate reward feedback without API calls.
- **Multi-Objective Reward Function:** Balances:
    - **Energy Saving:** Rewarded by turning off more cells.
    - **Network Coverage:** Penalizes UEs in weak coverage zones.
    - **QoS:** Scores based on SINR of connected UEs.
    - **Load Balance:** Discourages overloading active cells.
- **Comparative Visualization:** Generates side-by-side plots comparing baseline and optimized scenarios.

---

## 3. System Architecture

The application follows a linear pipeline, where the output of one stage becomes the input for the next.

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

---

## 4. Directory Structure

```
energy_saving_app/
│
├── main_app.py                 # Main orchestrator script
├── bdt_manager.py              # Manages BDT model training and Docker communication
├── data_preprocessor.py        # Prepares UE data for the Gym environment
├── rl_trainer.py               # RL training logic
├── rl_energy_saving_env.py     # Custom Gymnasium environment
├── rl_predictor.py             # Inference using the trained RL agent
├── energy_saving_visualizer.py # Generates comparison plots
│
├── topology.csv                # Cell tower layout
├── config.csv                  # Initial cell tower configuration
├── dummy_ue_training_data.csv  # Training data for BDT model
│
├── generated_data/
│   └── Day_*/                  # Data for each day
│       └── ue_data_per_tick/   # Raw UE location data per hour
│           ├── generated_ue_data_for_cco_0.csv
│           └── ... (up to 23)
│
└── (Generated Outputs)/
        ├── generated_data/
        │   └── Day_*/
        │       └── ue_data_gym_ready/  # Processed UE data
        ├── bdt_model_map.pickle        # Trained BDT model
        ├── energy_saver_agent.zip      # Trained RL agent
        ├── rl_training_logs/           # RL training logs and checkpoints
        └── plots/                      # Visualization outputs
```

---

## 5. Prerequisites

- **Python 3.8+**
- **Docker:** BDT model training runs inside a Docker container. Ensure Docker daemon is running.
- **Required Python Packages:** Create a `requirements.txt` file:

        ```text
        pandas
        numpy
        gymnasium
        stable-baselines3[extra]
        matplotlib
        torch
        gpytorch
        # any other specific libraries like radp_client
        ```

        Install packages:

        ```bash
        pip install -r requirements.txt
        ```

---

## 6. Application Workflow & Usage

The application is run as a pipeline, with each step triggered by a specific flag to `main_app.py`.

### **Step 1: Preprocess UE Data**

Prepares raw, per-hour UE location data for simulation.

```bash
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3
```

- **Input:** `generated_data/Day_*/ue_data_per_tick/`
- **Output:** `generated_data/Day_*/ue_data_gym_ready/`

---

### **Step 2: Train the Bayesian Digital Twin (BDT)**

Trains the RF simulation model using a backend service in Docker.

- **Prerequisites:** Docker container (e.g., `radp_dev-training-1`) must be running.

```bash
python main_app.py --train-bdt --bdt-model-id "bdt_for_energy_saving" --container "radp_dev-training-1"
```

- **Inputs:** `topology.csv`, `dummy_ue_training_data.csv`
- **Output:** `bdt_model_map.pickle`

---

### **Step 3: Train the RL Energy Saving Agent**

Trains the PPO agent using preprocessed data and the BDT model.

```bash
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 25000
```

- **Inputs:** `bdt_model_map.pickle`, `generated_data/Day_*/ue_data_gym_ready/`, `topology.csv`, `config.csv`
- **Outputs:** `energy_saver_agent.zip`, `rl_training_logs/`

---

### **Step 4: Run Inference**

Uses the trained agent to predict the optimal network configuration for a specific hour.

```bash
python main_app.py --infer --tick <T>
```

- **Inputs:** `energy_saver_agent.zip`, `topology.csv`
- **Output:** Console table of predicted optimal state (ON/OFF, tilt) for each cell.

---

### **Step 5: Visualize the Results**

Generates a side-by-side plot comparing network state before and after optimization.

```bash
python main_app.py --visualize --test-day <D> --tick <T>
```

- **Inputs:** `energy_saver_agent.zip`, `bdt_model_map.pickle`, `topology.csv`, `config.csv`, `generated_data/Day_<D>/ue_data_gym_ready/`
- **Output:** `.png` image in `plots/` directory.

---

### **Full Pipeline Example**

```bash
# 1. Prepare UE data for training (days 0-2) and testing (day 3)
python main_app.py --preprocess-data --train-days 0 1 2 --test-day 3

# 2. Train the core RF simulation model (ensure Docker container is running)
python main_app.py --train-bdt --bdt-model-id "bdt_for_energy_saving" --container "radp_dev-training-1"

# 3. Train the RL agent on the first 3 days of data
python main_app.py --train-rl --train-days 0 1 2 --total-timesteps 25000

# 4. Predict the optimal configuration for a late-night hour (e.g., 3 AM)
python main_app.py --infer --tick 3

# 5. Visualize the impact of the optimization on the test data for that hour
python main_app.py --visualize --test-day 3 --tick 3
```

---

## 7. Detailed Module Breakdown

### **data_preprocessor.py**
- **Function:** Prepares raw UE data for the RL environment.
- **Logic:** Reads per-tick CSV files, renames `lon` to `loc_x` and `lat` to `loc_y`, saves to `ue_data_gym_ready/`.

### **bdt_manager.py**
- **Function:** Manages backend-intensive training of the RF model.
- **Logic:** Uses a client (e.g., `radp_client`) to send topology and training data to a backend service. Downloads the trained model from Docker.

### **rl_trainer.py & rl_energy_saving_env.py**
- **Function:** Orchestrates PPO agent training.
- **Logic:** Loads BDT model and preprocessed UE data, initializes custom environment, trains agent with multi-objective reward.

### **rl_predictor.py**
- **Function:** Uses the trained agent for immediate recommendations.
- **Logic:** Loads PPO agent, predicts best action for a target tick, outputs a human-readable table.

### **energy_saving_visualizer.py**
- **Function:** Qualitative assessment of RL agent's performance.
- **Logic:** Simulates baseline and optimized scenarios, generates side-by-side plots of tower status and UE signal strength.
