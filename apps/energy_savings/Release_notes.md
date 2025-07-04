# Energy Saving Application - README

**Version:** 1.0
**Date:** June 18, 2025

## 1. Overview

The Energy Saving Application is a comprehensive, modular pipeline designed to train and deploy a Reinforcement Learning (RL) agent for dynamic network optimization. The core objective is to significantly reduce power consumption by intelligently turning cell sectors on or off and adjusting their antenna tilts based on the time of day, while maintaining acceptable network coverage and Quality of Service (QoS).

This system leverages a pre-trained Bayesian Digital Twin (BDT) RF model, which allows the RL training loop to perform rapid, local RF simulations. This architecture decouples the RL agent's learning process from the latency of live backend services, enabling efficient training on multi-day traffic patterns. The entire workflow is orchestrated through a main application with a clear command-line interface.

## 2. Key Features

- **Modular Pipeline:** Each major stage—data preprocessing, BDT model management, RL environment definition, RL agent training, inference, and visualization—is encapsulated in its own dedicated Python module for clarity and maintainability.
- **Time-Aware Energy Saving Policy:** The RL agent learns a policy that is dependent on the hour of the day (tick 0-23). Its actions include both setting the `cell_el_deg` (tilt) and turning individual cell sectors completely ON or OFF.
- **Multi-Day Training & Testing:** The pipeline is designed to use distinct sets of data for training and testing. The RL agent can be trained on several days of UE data to learn recurring patterns, and its performance can be evaluated on a separate, unseen test day.
- **Local BDT-based Simulation:** The RL training environment (`rl_energy_saving_env.py`) loads a pre-trained BDT model (pickled from the backend). It uses this model to run local RF simulations, providing immediate reward feedback to the agent without needing to make an API call for every simulation step.
- **Multi-Objective Reward Function:** The RL agent is trained to optimize a complex reward function that balances:
  - **Energy Saving:** The primary goal, rewarded by turning off more cells.
  - **Network Coverage:** A penalty for UEs falling into weak coverage zones.
  - **QoS:** A score based on the SINR of connected UEs.
  - **Load Balance:** A component to discourage overloading the remaining active cells.
- **Comparative Visualization:** Includes a dedicated visualizer to generate side-by-side plots that clearly show the impact of the RL agent's decisions by comparing a baseline scenario (all cells on) against the energy-saving optimized scenario.

---

## 3. Application Pipeline & Module Breakdown

The application is designed to be run as a sequence of steps, orchestrated by `main_app.py`.

### **Step 1: Preprocess UE Data**

- **Module:** `data_preprocessor.py`
- **Function:** Prepares the raw, multi-day UE location data for the RL environment.
- **Input:** Reads per-tick CSV files (e.g., `generated_ue_data_for_cco_0.csv`) from directories like `generated_data/Day_*/ue_data_per_tick/`. These files must have `lon` and `lat` columns.
- **Logic:** Iterates through all specified day directories, reads each CSV, renames the `lon` column to `loc_x` and `lat` to `loc_y` (as expected by the Gym's internal BDT prediction frame generator), and saves the result.
- **Output:** Creates a new set of directories and files (e.g., `generated_data/Day_*/ue_data_gym_ready/`) containing the preprocessed data.

### **Step 2: Train Bayesian Digital Twin (BDT) Model**

- **Module:** `bdt_manager.py`
- **Function:** Manages the one-time, backend-intensive training of the core RF model.
- **Input:**
  - `topology.csv`: The full network topology.
  - `dummy_ue_training_data.csv`: **This must be a realistic, high-quality dataset** that maps cell configurations (including different tilts) and UE locations to measured RSRP values. The "dummy" name is a placeholder; the data content must be real for a useful model.
- **Logic:**
  1.  Uses `radp_client` to send the topology and training data to the backend `training` service.
  2.  Waits for the backend to train the BDT model and save it. The backend saves the model as a pickled Python dictionary (`Dict[str, BayesianDigitalTwin]`).
  3.  After successful training, it uses a `docker cp` command to download the saved `model.pickle` file from inside the specified Docker container to a local path (e.g., `./bdt_model_map.pickle`).
- **Output:** A `bdt_model_map.pickle` file, stored locally, containing the trained BDT models for all cells.

### **Step 3: Train the RL Agent**

- **Module:** `rl_trainer.py`
- **Function:** The core of the machine learning process. It orchestrates the training of the PPO agent for energy saving.
- **Input:**
  - The `bdt_model_map.pickle` file generated in Step 2.
  - The preprocessed per-tick UE data from the `ue_data_gym_ready` directories for the specified training days.
  - `topology.csv` and `config.csv` (for initial state).
- **Logic:**
  1.  Loads the BDT model map from the pickle file.
  2.  Loads all per-tick UE DataFrames from the specified training day directories into a dictionary: `{tick: [df_day0, df_day1, ...]}`.
  3.  Initializes the custom RL environment, `TickAwareEnergyEnv` (defined in `rl_energy_saving_env.py`), passing it the loaded BDT models and the pool of UE data.
  4.  Initializes a Stable Baselines3 PPO agent.
  5.  Calls `model.learn()`. The PPO agent then interacts with the `TickAwareEnergyEnv` for thousands of steps:
      - The agent observes the current `tick`.
      - It outputs an `action` (a set of tilts and on/off states for all cells).
      - The environment's `step` method applies the action, **samples a random day's UE data for the current tick**, and runs a **local RF simulation** using the loaded BDT models.
      - A multi-objective reward is calculated and returned to the agent.
  6.  The agent updates its policy based on the rewards received.
- **Output:** A trained RL agent saved as a `.zip` file (e.g., `./energy_saver_agent.zip`).

### **Step 4: Run Inference with the RL Agent**

- **Module:** `rl_predictor.py`
- **Function:** Uses the trained RL agent to get an immediate recommendation for a specific hour.
- **Input:**
  - The path to the saved RL agent (`.zip` file).
  - The `topology.csv` file (to get the correct cell order).
  - A target `tick` (0-23) from the command line.
- **Logic:**
  1.  Loads the trained PPO agent.
  2.  Calls `model.predict(target_tick)` to get the deterministic best action for that hour.
  3.  Maps the numerical action array to a human-readable DataFrame showing each `cell_id`, its predicted `state` (ON/OFF), and its `predicted_cell_el_deg`.
- **Output:** Prints the recommended energy-saving configuration to the console.

### Step 5: Visualize Performance

- **Module:** `energy_saving_visualizer.py`
- **Function:** Provides a qualitative assessment of the RL agent's performance by comparing it against a baseline.
- **Input:**
  - The BDT model pickle file.
  - The trained RL agent zip file.
  - `topology.csv` and `config.csv`.
  - A specific `test-day` and `tick` to evaluate.
- **Logic:**
  1.  Loads all necessary models and data.
  2.  **Scenario A (Baseline):** Simulates RF performance for the specified tick using the _initial_ configuration from `config.csv` with all cells turned ON.
  3.  **Scenario B (Optimized):** Uses the RL agent to predict the best configuration (tilts and on/off states) for that tick, then simulates RF performance using _that_ configuration.
  4.  **Plotting:** Generates a side-by-side plot showing the UE coverage map for both scenarios. The plot visualizes which towers are Fully Active, Partially Active, or Fully Inactive, and colors UEs by their received signal strength (RSRP).
- **Output:** A `.png` comparison image saved to the `./plots/` directory.

---

## 4. How to Use

The pipeline is controlled via command-line arguments to `main_app.py`.

**Example Full Workflow:**

```bash
# Set your Maveric project root if not already set
export MAVERIC_ROOT=/path/to/your/maveric/project

# Step 1: Preprocess UE data for training (days 0-3) and testing (day 4)
python main_app.py --preprocess-data --train-days 0 1 2 3 --test-day 4

# Step 2: Train the core BDT model and download it from the 'radp_dev-training-1' container
# Note: You may need to change the --container name to match your Docker setup.
python main_app.py --train-bdt --bdt-model-id "bdt_for_energy_saving" --container "radp_dev-training-1"

# Step 3: Train the RL agent on the first 4 days of data
python main_app.py --train-rl --train-days 0 1 2 3 --total-timesteps 25000

# Step 4: Get a specific energy-saving recommendation for a peak hour (e.g., tick 18)
python main_app.py --infer --tick 18

# Step 5: Visualize how the agent's recommendation performs on the test day data for that hour
python main_app.py --visualize --test-day 4 --tick 18
```
