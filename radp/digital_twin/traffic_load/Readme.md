# Maveric Traffic Load Simulation Framework

This framework simulates User Equipment (UE) traffic demand across a configurable mobile network topology over multiple days. It generates realistic UE distributions based on spatial and temporal parameters, creating datasets for:

- Network performance analysis
- Machine learning model training
- Coverage optimization studies

The simulation core uses Voronoi tessellation to create a spatial layout from cell tower locations, with each Voronoi cell representing a distinct geographical area. UEs are distributed across these areas according to time-varying weights, simulating population movement patterns (e.g., between residential and commercial zones).

---

## Key Features

- **Multi-Day Simulation:** Generate traffic demand over multiple consecutive days for extensive datasets.
- **Configurable Network Topology:** Automatically generate or use an existing topology. Configure number of sites, cells per site, locations, and azimuths.
- **Voronoi-Based Spatial Modeling:** Create spatial layouts using Voronoi polygons around cell sites, assigning area types (e.g., residential, commercial).
- **Time-Varying UE Distribution:** Distribute UEs based on configurable time-based weights to simulate daily traffic patterns.
- **Data Generation:** Produce per-tick UE location data, initial network configuration files, and consolidated dummy training datasets.
- **Visualization:** Generate plots for specified simulation ticks, showing UE locations, cell towers, and shaded spatial areas.
- **Modular & Extensible:** Logical code structure for configuration, simulation, and visualization, making it easy to extend.

---

## Project Structure

The framework consists of several key Python scripts:

- `traffic_demand_app.py`: Main application entry point; orchestrates the simulation workflow, handles CLI arguments, and calls other modules.
- `traffic_demand_simulation.py`: Contains the `TrafficDemandModel` class for generating the spatial layout and distributing UEs over time.
- `config_gen.py`: Contains the `ScenarioConfigurationGenerator` class for creating network topology files, initial cell configurations, and dummy training data.
- `plot_gen.py`: Contains the `TrafficDataVisualizer` class for generating plots of UE distribution and network layout.

---

## Prerequisites

Requires Python 3 and the following libraries:

- pandas
- numpy
- matplotlib
- scipy
- shapely

Install dependencies with:

```bash
pip install -r requirements.txt
```

The scripts also attempt to import a custom library, `radp`. If unavailable, the scripts fall back to basic definitions for required constants and tools.

---

## Configuration

Before running the simulation, provide two JSON configuration files defining the spatial and temporal behavior of UEs.

### 1. Spatial Parameters (`spatial_params.json`)

Defines area types and their proportions (should sum to 1.0):

```json
{
    "types": [
        "type_1",
        "type_2",
        "type_3"
    ],
    "proportions": [
        0.35,
        0.25,
        0.40
    ]
}
```

### 2. Time Parameters (`time_params.json`)

Defines simulation time parameters. `total_ticks` sets the number of discrete time steps in a day. `time_weights` provides UE density multipliers for each area type at each tick.

```json
{
    "total_ticks": 24,
    "tick_duration_min": 60,
    "time_weights": {
        "type_1": [
            0.8901912520505434, 0.9366538454508161, 0.8850210544911438, 0.08576316377644322,
            0.3550281976989599, 0.29496579053939276, 0.26392356242241855, 0.5789886718891961,
            0.020840126847845863, 0.12615412294825634, 0.9370664665163935, 0.5120685223556638,
            0.9073482730676804, 0.8431515482719586, 0.7898642692222879, 0.7480359899839738,
            0.04184346977408948, 0.9912107812398174, 0.5458574787429198, 0.5665166694689662,
            0.3600255537355952, 0.6038182888071592, 0.4785634191863364, 0.450319584914449
        ],
        "type_2": [
            0.19092628845957593, 0.891720202967957, 0.7998009318567456, 0.7843241337227397,
            0.01210799524341144, 0.6995705348847321, 0.22373246860333673, 0.7527482860656918,
            0.5843323776566927, 0.1851647504528009, 0.5175249229907024, 0.027781626031334605,
            0.4929267605677564, 0.0980224595897754, 0.18340423700207498, 0.050242480109088494,
            0.5484725076757121, 0.3566541042902558, 0.31365975225586795, 0.8085132916576125,
            0.5346845420623789, 0.25307294842490335, 0.3058281611231879, 0.3556946549918253
        ],
        "type_3": [
            0.3148618990521689, 0.5585637679959404, 0.6911034203948746, 0.40737247747835126,
            0.9576154739922703, 0.8945550914400008, 0.2558435878354406, 0.7485802785655512,
            0.5756290174625156, 0.38788784770825546, 0.43402745898272055, 0.39076030055192634,
            0.18349175135576679, 0.985018398444745, 0.6702421652700735, 0.3119475806020052,
            0.36951873929843715, 0.011155794738183622, 0.04263007284029874, 0.6489752515416065,
            0.8899141816333724, 0.3030836625085872, 0.06641282939127147, 0.31214241358754147
        ]
    }
}
```

---

## Usage

### Step 1: Generate Initial Configuration (Optional)

If you do not have a `topology.csv`, generate one along with a default `config.csv`:

```bash
python traffic_demand_app.py \
    --generate_config_flag \
    --num_sites 10 \
    --cells_per_site 3 \
    --output_dir "./generated_data"
```

This creates `topology.csv` and `config.csv` inside `./generated_data`.

### Step 2: Run the Simulation

Run the main application, pointing to your configuration files. Example for a 3-day simulation with plots and dummy training data:

```bash
python traffic_demand_app.py \
    --days 3 \
    --num_ues 1000 \
    --output_dir "./generated_data" \
    --topology_csv "topology.csv" \
    --spatial_params_json "spatial_params.json" \
    --time_params_json "time_params.json" \
    --generate_plots_flag \
    --generate_dummy_training_flag
```

### Example: 
*Whole Flow in One Command*

```bash
python traffic_demand_app.py \
    --generate_config_flag \
    --num_sites 5 \
    --cells_per_site 3 \
    --lat_range 240.7 -140 \
    --lon_range -74.05 100 \
    --num_ues 500 \
    --generate_plots_flag \
    --generate_dummy_training_flag \
    --generate_plots_flag \
    --plot_max_ticks 0
```

---

## Output Structure

The application generates the following directory structure and files inside the specified `--output_dir` (e.g., `./generated_data`):

```text
generated_data/
│
├── Day_0/
│   ├── ue_data_per_tick/
│   │   ├── generated_ue_data_for_cco_0.csv
│   │   ├── generated_ue_data_for_cco_1.csv
│   │   └── ...
│   └── plots/
│       ├── ue_distribution_tick_0.png
│       ├── ue_distribution_tick_1.png
│       └── ...
│
├── Day_1/
│   └── ... (similar structure)
│
├── topology.csv                # Network topology file.
├── config.csv                  # Initial cell configuration file.
└── dummy_ue_training_data.csv  # Combined dummy data for training.
```