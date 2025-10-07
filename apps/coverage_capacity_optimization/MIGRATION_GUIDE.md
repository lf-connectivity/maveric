# Migration Guide: From Original CCO to Refactored Modular Structure

This guide helps users migrate from the original CCO codebase to the new modular structure.

## Overview of Changes

The original CCO application has been refactored from a monolithic structure into a clean, modular architecture with proper separation of concerns.

### Original Structure
```
apps/coverage_capacity_optimization/
├── dgpco_cco.py           # Main algorithm implementation
├── cco_engine.py          # Core CCO calculations
├── cco_example_app.py     # Example usage
├── cco_anp_app.py         # ANP-specific example
├── constants.py           # Constants and enums
└── data/                  # Data files
```

### New Modular Structure
```
apps/coverage_capacity_optimization/
├── bdt_manager.py          # Base station data management
├── data_preprocessing.py   # Data cleaning and preparation
├── cco_env.py             # Environment/simulation setup
├── cco_trainer.py         # Model training logic
├── cco_prediction.py      # Prediction and inference
├── main_app.py            # Main orchestrator
├── config.yaml            # Configuration file
├── README.md              # Documentation
├── test_integration.py    # Integration tests
└── requirements.txt       # Dependencies
```

## Key Changes

### 1. Functionality Mapping

| Original Component | New Module | Purpose |
|-------------------|------------|---------|
| `DgpcoCCO` class | `CCOTrainer` | Training logic |
| `CcoEngine` class | `cco_env.py` + `cco_prediction.py` | CCO calculations |
| Data loading code | `data_preprocessing.py` | Data handling |
| RADP client usage | `bdt_manager.py` | Model management |
| Example apps | `main_app.py` | Orchestration |

### 2. Configuration Changes

**Before**: Hardcoded parameters in scripts
```python
VALID_CONFIGURATION_VALUES = {
    "cell_el_deg": [0.0, 1.0, 2.0, ..., 20.0]
}
```

**After**: YAML configuration file
```yaml
cco:
  lambda_: 0.5
  weak_coverage_threshold: -90
  over_coverage_threshold: 0
  growth_rate: 1.0
```

### 3. Data Processing Changes

**Before**: Manual data loading in each script
```python
topology = pd.read_csv(TOPOLOGY_FILE)
training_data = pd.concat([pd.read_csv(file) for file in TRAINING_DATA_FILES])
```

**After**: Structured preprocessing pipeline
```python
ue_preprocessor = UEDataPreprocessor()
training_data = ue_preprocessor.load_ue_data(training_data_files)
training_data = ue_preprocessor.clean_ue_data(training_data)
```

## Migration Steps

### Step 1: Update Imports

**Before**:
```python
from apps.coverage_capacity_optimization.dgpco_cco import DgpcoCCO
```

**After**:
```python
from apps.coverage_capacity_optimization.main_app import CCOMainApp
# or
from apps.coverage_capacity_optimization.cco_trainer import CCOTrainer
```

### Step 2: Update Initialization

**Before**:
```python
dgpco_cco = DgpcoCCO(
    topology=topology,
    valid_configuration_values=VALID_CONFIGURATION_VALUES,
    bayesian_digital_twin_id=MODEL_ID,
    ue_data=prediction_data,
    config=prediction_config,
)
```

**After**:
```python
app = CCOMainApp(config_path="config.yaml")
# or
trainer = CCOTrainer(environment, algorithm="dgpco")
```

### Step 3: Update Execution

**Before**:
```python
rf_dataframe_per_epoch, coverage_dataframe_per_epoch, cco_objective_per_epoch, opt_per_epoch = dgpco_cco.run(num_epochs=20)
```

**After**:
```python
# Command line
python main_app.py train --config config.yaml

# Programmatic
training_results = app.train()
```

### Step 4: Update Configuration

Create a `config.yaml` file with your parameters:

```yaml
data:
  topology_file: "data/topology.csv"
  training_data_files: ["data/ue_training_data.csv"]
  prediction_data_files: ["data/ue_data.csv"]
  config_file: "data/config.csv"

model:
  model_id: "your_model_name"
  algorithm: "dgpco"

training:
  num_epochs: 20
  epsilon: 0.1
  seed: 42

cco:
  lambda_: 0.5
  weak_coverage_threshold: -90
  over_coverage_threshold: 0
```

## Common Migration Patterns

### 1. Simple Training Script

**Before**:
```python
# Load data
topology = pd.read_csv("topology.csv")
training_data = pd.read_csv("training_data.csv")
config = pd.read_csv("config.csv")

# Initialize and train
dgpco_cco = DgpcoCCO(topology, valid_values, model_id, training_data, config)
results = dgpco_cco.run(num_epochs=100)
```

**After**:
```python
# Method 1: Command line
python main_app.py train --config config.yaml

# Method 2: Programmatic
app = CCOMainApp("config.yaml")
results = app.train()
```

### 2. Custom Training Parameters

**Before**:
```python
results = dgpco_cco.run(
    num_epochs=50,
    lambda_=0.3,
    weak_coverage_threshold=-85,
    seed=123
)
```

**After**:
```python
# Update config.yaml
training:
  num_epochs: 50
  seed: 123
cco:
  lambda_: 0.3
  weak_coverage_threshold: -85

# Run
app = CCOMainApp("config.yaml")
results = app.train()
```

### 3. Model Evaluation

**Before**:
```python
# Manual evaluation logic
final_reward = results[-1]  # Last epoch reward
```

**After**:
```python
# Built-in evaluation
evaluation_results = app.evaluate(num_episodes=10)
print(f"Average reward: {evaluation_results['average_reward']}")
```

### 4. Custom Callbacks

**Before**:
```python
# Manual logging
print(f"Epoch {epoch}: reward = {reward}")
```

**After**:
```python
from apps.coverage_capacity_optimization.cco_trainer import LoggingCallback

trainer = CCOTrainer(environment, "dgpco")
trainer.add_callback(LoggingCallback(log_interval=10))
```

## Backward Compatibility

The refactored code preserves all original functionality while providing:

1. **Same Algorithm**: dGPCO algorithm implementation is preserved
2. **Same Results**: Output format and calculations remain identical
3. **Same Parameters**: All CCO parameters are supported
4. **Enhanced Features**: Additional capabilities like visualization, export, etc.

## Troubleshooting Migration

### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'apps.coverage_capacity_optimization.dgpco_cco'`

**Solution**: Update imports to use new modules:
```python
# Old
from apps.coverage_capacity_optimization.dgpco_cco import DgpcoCCO

# New
from apps.coverage_capacity_optimization.main_app import CCOMainApp
```

### Issue 2: Configuration Errors
**Error**: Missing configuration parameters

**Solution**: Create `config.yaml` file with required parameters:
```yaml
data:
  topology_file: "path/to/topology.csv"
  # ... other parameters
```

### Issue 3: Data Format Changes
**Error**: Data loading fails

**Solution**: Use preprocessing pipeline:
```python
preprocessor = UEDataPreprocessor()
data = preprocessor.load_ue_data("data.csv")
data = preprocessor.clean_ue_data(data)
```

### Issue 4: Parameter Changes
**Error**: Unknown parameter in function call

**Solution**: Check parameter names in new API:
```python
# Old
dgpco_cco.run(num_epochs=100, lambda_=0.5)

# New - via config file
training:
  num_epochs: 100
cco:
  lambda_: 0.5
```

## Testing Migration

Use the integration test to verify your migration:

```bash
cd apps/coverage_capacity_optimization
python test_integration.py
```

This will test:
- Module imports
- Data preprocessing
- BDT manager functionality
- Main app initialization

## Support

If you encounter issues during migration:

1. Check the README.md for detailed documentation
2. Review the integration test for usage examples
3. Verify your configuration file format
4. Ensure all dependencies are installed

## Benefits of Migration

After migration, you'll have:

1. **Better Maintainability**: Modular structure with clear responsibilities
2. **Enhanced Configuration**: YAML-based configuration management
3. **Improved Error Handling**: Comprehensive error checking and logging
4. **Command Line Interface**: Easy-to-use CLI for common operations
5. **Better Testing**: Structured test framework
6. **Documentation**: Comprehensive documentation and examples
7. **Extensibility**: Easy to add new algorithms and features

The migration effort is worthwhile for the improved code quality, maintainability, and user experience.
