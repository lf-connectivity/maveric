# Coverage and Capacity Optimization (CCO) - Simplified Modular Structure

This is a refactored version of the Coverage and Capacity Optimization application with a clean, modular architecture following the existing patterns from the energy savings module.

## Architecture Overview

The application has been restructured into the following simple modules:

```
apps/coverage_capacity_optimization/
├── bdt_manager.py          # Base station data management (same as energy_savings)
├── data_preprocessing.py   # Data cleaning and preparation (same as energy_savings)
├── cco_env.py             # CCO environment based on dGPCO algorithm
├── cco_trainer.py         # CCO training using dGPCO
├── cco_prediction.py      # Prediction of optimized el_deg values
├── main_app.py            # Main CLI application
└── data/                  # Data directory
    ├── topology.csv
    ├── config.csv
    ├── ue_data.csv
    └── ue_training_data.csv
```

## Module Descriptions

### 1. bdt_manager.py
- **Purpose**: Identical to energy_savings version
- **Key Features**:
  - BDT model training using RADP client
  - Model downloading from Docker container
  - Simple file path management

### 2. data_preprocessing.py
- **Purpose**: Identical to energy_savings version
- **Key Features**:
  - UE data preprocessing for CCO environment
  - Coordinate transformation (lon/lat → loc_x/loc_y)
  - Batch processing of data files

### 3. cco_env.py
- **Purpose**: CCO environment based on dGPCO algorithm
- **Key Features**:
  - Core functionality from original DgpcoCCO class
  - Metric calculation using CCO engine
  - Configuration management
  - Simulation execution

### 4. cco_trainer.py
- **Purpose**: CCO training using dGPCO algorithm
- **Key Features**:
  - dGPCO training implementation
  - Epsilon-greedy exploration
  - Model saving/loading
  - Training history tracking

### 5. cco_prediction.py
- **Purpose**: Prediction of optimized el_deg values
- **Key Features**:
  - Load trained models
  - Generate optimized el_deg predictions
  - Compare with current configuration
  - Generate recommendations
  - Export results

### 6. main_app.py
- **Purpose**: CLI application to run the complete pipeline
- **Key Features**:
  - Command-line interface
  - Full pipeline orchestration
  - Data preprocessing support
  - Results export

## Usage

### Command Line Interface

#### Run Full Pipeline
```bash
python main_app.py \
  --topology data/topology.csv \
  --training data/ue_training_data.csv \
  --prediction data/ue_data.csv \
  --config data/config.csv \
  --model-id cco_model
```

#### Preprocess Data Only
```bash
python main_app.py --preprocess --base-dir data --days 1 2 3
```

#### With Custom Parameters
```bash
python main_app.py \
  --topology data/topology.csv \
  --training data/ue_training_data.csv \
  --prediction data/ue_data.csv \
  --config data/config.csv \
  --model-id cco_model \
  --epochs 50 \
  --container my_radp_container
```

### Programmatic Usage

```python
from apps.coverage_capacity_optimization.main_app import CCOMainApp

# Initialize application
app = CCOMainApp(
    topology_path="data/topology.csv",
    training_data_path="data/ue_training_data.csv", 
    prediction_data_path="data/ue_data.csv",
    config_path="data/config.csv",
    model_id="my_model"
)

# Run full pipeline
results = app.run_full_pipeline()

# Or run individual steps
app.train_bdt(container_name="radp_training")
cco_results = app.train_cco()
predictions = app.predict_el_deg()
```

## Data Format Requirements

### Topology Data (topology.csv)
Required columns:
- `cell_id`: Unique cell identifier
- `lat`: Cell latitude
- `lon`: Cell longitude
- `cell_az_deg`: Cell azimuth angle
- `cell_carrier_freq_mhz`: Carrier frequency

### UE Data (ue_data.csv, ue_training_data.csv)
Required columns:
- `lon`: Longitude
- `lat`: Latitude
- Optional: `mock_ue_id`, `tick`

### Configuration Data (config.csv)
Required columns:
- `cell_id`: Cell identifier (must match topology)
- `cell_el_deg`: Elevation tilt angle

## Output Files

The pipeline generates the following output files:

### Training Outputs
- `models/{model_id}.pickle`: Trained BDT model
- `models/cco_{model_id}.pickle`: Trained CCO model

### Prediction Outputs
- `results/el_deg_predictions_{model_id}.csv`: Optimized el_deg values
- `results/el_deg_comparison_{model_id}.csv`: Comparison with current config
- `results/el_deg_recommendations_{model_id}.csv`: Recommendations for changes

## Algorithm Details

### dGPCO (Distributed Gaussian Process Coverage Optimization)
- Greedy optimization algorithm
- Epsilon-greedy exploration for robustness
- Cell-by-cell optimization approach
- Based on original DgpcoCCO implementation

### Pipeline Flow
1. **BDT Training**: Train Bayesian Digital Twin model using RADP
2. **CCO Training**: Train CCO optimization using dGPCO algorithm
3. **Prediction**: Generate optimized el_deg values for all cells
4. **Export**: Save predictions and recommendations

## Configuration

The application uses command-line arguments for configuration:

- `--topology`: Path to topology CSV file
- `--training`: Path to training data CSV file
- `--prediction`: Path to prediction data CSV file
- `--config`: Path to configuration CSV file
- `--model-id`: Model identifier for BDT training
- `--container`: Docker container name (default: radp_training)
- `--epochs`: Number of training epochs (default: 20)

## Dependencies

- pandas
- numpy
- radp.client (RADP client)
- python-dotenv

## Error Handling

- Comprehensive logging at all levels
- Graceful failure with informative error messages
- File existence validation
- Data format validation

## Migration from Original Code

The refactored code preserves the original dGPCO algorithm while providing:
- Better modularity and maintainability
- Command-line interface
- Consistent patterns with energy_savings module
- Simplified configuration
- Better error handling and logging

## Key Differences from Original

1. **Simplified Structure**: Follows the same pattern as energy_savings module
2. **CLI Interface**: Easy-to-use command-line interface
3. **Modular Design**: Clear separation of concerns
4. **Prediction Focus**: Main output is optimized el_deg values
5. **Export Capabilities**: Results exported in multiple formats

## Troubleshooting

### Common Issues

1. **RADP Client Issues**
   - Ensure RADP client is properly configured
   - Check Docker container is running
   - Verify container name is correct

2. **Data Format Issues**
   - Validate CSV file formats
   - Check column names match requirements
   - Ensure file paths are correct

3. **Memory Issues**
   - Reduce number of training epochs
   - Check available system memory
   - Process data in smaller batches

### Debug Mode
```bash
python main_app.py --verbose --log-level DEBUG [other args...]
```

## License

This code is licensed under the MIT license as specified in the original files.