# Mobility Robustness Optimization (MRO) Application

> **Version:** 1.0 | **Date:** October 2, 2025

## Table of Contents

1. **[Overview](#1-overview)**
2. **[Architecture & Components](#2-architecture--components)**
3. **[What MRO Does](#3-what-mro-does)**
4. **[Directory Structure](#4-directory-structure)**
5. **[Prerequisites and Setup](#5-prerequisites-and-setup)**
6. **[Quick Start Example](#6-quick-start-example)**
7. **[Step-by-Step Usage Guide](#7-step-by-step-usage-guide)**
8. **[Detailed Code Examples](#8-detailed-code-examples)**
9. **[Related Notebooks](#9-related-notebooks)**
10. **[Troubleshooting](#10-troubleshooting)**

---
> **Prerequisites:** Complete the setup steps in [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) before proceeding.

# **1. Overview**

The Mobility Robustness Optimization (MRO) application is a comprehensive, modular solution designed to optimize cellular network handover parameters to ensure seamless mobility for User Equipment (UE). **The primary objective is to minimize Radio Link Failures (RLF) while maximizing cell_rx_pwr_dbm and reducing handover interruptions by intelligently tuning hysteresis and Time-to-Trigger (TTT) parameters.**

MRO leverages **Bayesian Digital Twin (BDT)** RF models to predict received signal strength across the network and simulates UE mobility patterns to evaluate different handover configurations. The application supports three optimization approaches: random/grid search, Bayesian optimization, and reinforcement learning.

---

# **2. Architecture & Components**

## Base Class: `MobilityRobustnessOptimization`

The abstract base class ([mobility_robustness_optimization.py](mobility_robustness_optimization.py)) provides core functionality:

- **RF Twin Management:**
  - `train_or_update_rf_twins()`: Train new or update existing Bayesian Digital Twins
  - `save_bdt()` / `load_bdt()`: Persist and load trained models

- **Prediction Engine:**
  - `_predictions()`: Predict received power for UE locations using BDTs
  - `_preprocess_simulation_data()`: Prepare data for MRO analysis

- **Utility Methods:**
  - `_add_sinr_column()`: Calculate Signal-to-Interference-plus-Noise Ratio
  - `_prepare_train_or_update_data()`: Process training data with features

- **PRediciting Optimal Hysterisis and Time-To-Trigger:**
  - `solve()`: It pruduces synthetic data based on simulation ran on real data to predict the optimal hysterisis (hyst) & time-to-trigger (ttt) for all user equipments in the simulative enviroment

## Optimization Implementations

#### 1. SimpleMRO ([simple_mro.py](simple_mro.py:17))
Iterative random search optimization that:
- Randomly samples hysteresis and TTT values within valid ranges
- Evaluates each configuration using the MRO metric
- Tracks all trials in a score DataFrame
- Returns the best-performing configuration

**Best for:** Quick exploration, baseline comparisons, understanding parameter sensitivity

#### 2. ReinforcedMRO ([mro_rl.py](mro_rl.py:20))
Reinforcement learning approach using:
- **PPO (Proximal Policy Optimization)** agent
- Custom Gymnasium environment (`ReinforcedMROEnv`)
- Continuous action space for hysteresis and TTT
- Episode-based learning with configurable timesteps

**Best for:** Complex optimization landscapes, learning from sequential decisions

## Key Utility Functions

- `train_or_update_rf_twins`: Train new or update existing Bayesian Digital Twins with UE measurement data
- `save_bdt`: Persist trained digital twin models to disk for reuse
- `load_bdt`: Load pre-trained digital twin models from disk
- `solve`: Execute optimization algorithm to find optimal hysteresis and TTT parameters 


---

# **3. What MRO Does: Intelligent Handover Parameter Optimization**

**Core Function:** MRO optimizes cellular network handover parameters (hysteresis and Time-to-Trigger) to minimize connection failures while reducing interruptions during user mobility.

**Key Logic:**
- **Problem:** Balance competing objectives - too aggressive handovers → many interruptions (50ms each), too conservative handovers → Radio Link Failures (1000ms recovery)
- **Solution:** Find optimal parameters that maximize effective operational time: `D = Total Simulation Time - (successful_handovers × 50ms + RLF_events × 1000ms)`
- **Methods:** SimpleMRO (random search), ReinforcedMRO (PPO reinforcement learning)
- **Process:** Train Bayesian Digital Twins for RF prediction → Simulate UE mobility → Calculate MRO metric → Return optimal parameters
- **Cell Attachment Logic:** SINR calculation across all cells → Apply hysteresis to serving cell → TTT countdown before handover → RLF detection when signal drops below threshold

**Key Parameters:**
- **Hysteresis (dB):** Offset preventing ping-pong handovers between cells
- **Time-to-Trigger (TTT):** Duration condition must be met before triggering handover
- **Radio Link Failure (RLF):** Connection drop when signal below threshold (-120 dBm)

---

# **4. Directory Structure**

```
apps/mobility_robustness_optimization/
├── mobility_robustness_optimization.py  # Abstract base class
├── simple_mro.py                        # Random/grid search optimization
├── mro_ml.py                            # Bayesian optimization (GPR/XGBoost)
├── mro_rl.py                            # Reinforcement learning (PPO)
└── tests/                               # Unit tests
    ├── test_mobility_robustness_optimization.py
    └── test_mro_ml.py
```

**Related Resources:**
- [notebooks/mro.ipynb](../../notebooks/mro.ipynb): Interactive MRO workflow demonstration
- [notebooks/mobility_model.ipynb](../../notebooks/mobility_model.ipynb): UE mobility pattern analysis
- [notebooks/radp_library.py](../../notebooks/radp_library.py): Shared utility functions

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


# **6. Related Notebooks**

### [notebooks/mro.ipynb](../../notebooks/mro.ipynb)

**Purpose:** Interactive demonstration of the complete MRO workflow

**Contents:**
- Data loading and preprocessing
- Bayesian Digital Twin training
- SimpleMRO optimization walkthrough
- BayesianMRO comparison
- Visualization of results:
  - UE tracks with cell assignments
  - Handover events over time
  - 3D plots (hysteresis × TTT × MRO score)
  - SINR distribution analysis

**When to Use:**
- Learning how MRO works
- Experimenting with parameters
- Visualizing optimization results
- Debugging data issues

---

### [notebooks/mobility_model.ipynb](../../notebooks/mobility_model.ipynb)

**Purpose:** Explore and understand UE mobility patterns

**Contents:**
- Gauss-Markov mobility model
- UE track generation
- Velocity distribution analysis
- Spatial pattern visualization
- Mobility class configuration

**When to Use:**
- Designing realistic mobility scenarios
- Understanding UE movement patterns
- Tuning mobility parameters for optimization
- Generating training data

---

### [notebooks/radp_library.py](../../notebooks/radp_library.py)

**Purpose:** Shared utility functions used across notebooks and applications


---

# **7. Troubleshooting**

### Common Issues

#### Issue 1: Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'radp'
```

**Solution:**
```bash
# Set PYTHONPATH to include project root
export PYTHONPATH="$(pwd)":$PYTHONPATH

# Verify
echo $PYTHONPATH
```

---

#### Issue 2: Cartesian Format Validation Failure

**Error:**
```
ValueError: The input DataFrame is not in the expected cartesian format
```

**Solution:**

Ensure your data has one row per UE-cell pair:

```python
# Check your data
print(ue_data.groupby(['longitude', 'latitude']).size().unique())
# Should return array with consistent cell count (e.g., [3] for 3 cells)

# Convert if needed
from notebooks.radp_library import get_ues_cells_cartesian_df

ue_data_cartesian = get_ues_cells_cartesian_df(ue_data, topology)
```

---

#### Issue 3: Cell ID Format Mismatch

**Error:**
```
KeyError: 'cell_1' or ValueError during cell lookup
```

**Solution:**

Normalize cell IDs consistently:

```python
from notebooks.radp_library import normalize_cell_ids

# Normalize both dataframes
topology = normalize_cell_ids(topology)
ue_data = normalize_cell_ids(ue_data)

# Verify format consistency
print(f"Topology cell_id dtype: {topology['cell_id'].dtype}")
print(f"UE data cell_id dtype: {ue_data['cell_id'].dtype}")
```

---

#### Issue 4: CUDA/GPU Issues

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

Force CPU usage:

```python
import torch

# Set device to CPU
device = torch.device("cpu")

# Or set environment variable before import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

Install CPU-only PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

#### Issue 5: Simulation Boundary Issues

**Error:**
```
ValueError: lat_lon_boundaries not properly set
```

**Solution:**

Let MRO auto-calculate boundaries:

```python
from notebooks.radp_library import find_sim_boundary

# Calculate boundaries from topology and UE data
bounds = find_sim_boundary(topology, ue_data)

# Update mobility parameters
mobility_params["ue_tracks_generation"]["params"]["lat_lon_boundaries"] = bounds

print(f"Simulation boundaries: {bounds}")
```

---

#### Issue 6: Empty or NaN Predictions

**Error:**
```
ValueError: NaN values in prediction results
```

**Solution:**

Check Digital Twin training:

```python
# Verify twins are trained
print(f"Trained cells: {list(mro.bayesian_digital_twins.keys())}")

# Check for sufficient training data
for cell_id, twin in mro.bayesian_digital_twins.items():
    print(f"{cell_id}: {twin.data_in[0].shape[0]} samples")

# Minimum ~50-100 samples per cell recommended
```

---

#### Issue 7: Optimization Returns Same Parameters

**Error:** All optimization runs return identical hysteresis/TTT

**Solution:**

Check parameter ranges:

```python
from radp.digital_twin.utils.cell_selection import find_hyst_diff

# Calculate valid hysteresis range
max_diff = find_hyst_diff(simulation_data)
print(f"Hysteresis range: [0, {max_diff}]")

# Ensure TTT range is valid
num_ticks = simulation_data['tick'].nunique()
print(f"TTT range: [2, {num_ticks + 1}]")

# Ranges too narrow? Increase simulation ticks or check data
```

---

### Getting Help

If issues persist:

1. **Check logs:** Review detailed error messages
2. **Verify data:** Ensure input data formats are correct
3. **Test notebooks:** Run [mro.ipynb](../../notebooks/mro.ipynb) with sample data
4. **Review tests:** Check test cases for examples
5. **Consult DEVELOPER_GUIDE.md:** For general setup issues

---





**For questions or contributions, please refer to the main project documentation.**
