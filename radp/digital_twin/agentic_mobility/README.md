# Agentic Mobility Generation

Transform RADP's mobility generator from **parameter-driven JSON** to **natural language queries** using LLM-powered workflows.

## Quick Start

### 1. Install & Configure

```bash
# Install dependencies
pip install -r radp/digital_twin/requirements.txt

# Set up API key
cp radp/digital_twin/agentic_mobility/.env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

### 2. Basic Usage

```python
from radp.digital_twin.agentic_mobility.integration import AgenticMobilityIntegration

# Natural language → mobility DataFrame
df, metadata = AgenticMobilityIntegration.generate_from_natural_language(
    "Generate 100 UEs in urban Tokyo during morning rush hour"
)

print(df.head())  # pandas DataFrame: mock_ue_id, lon, lat, tick
```

---

## Examples

Run these from the project root:

### 1. Agentic Mobility Pipeline
```bash
python3 radp/digital_twin/agentic_mobility/examples/gen_mobility_example.py
```
- Natural language → RADP parameters
- Shows query parsing, location resolution, parameter generation
- Outputs: JSON parameters

### 2. Mobility + Data Simulation
```bash
python3 radp/digital_twin/agentic_mobility/examples/end_to_end_example.py
```
- Natural language → mobility DataFrame
- Complete pipeline with data generation
- Outputs: CSV files with UE tracks

### 3. Mobility + Simulation + Topology + Visualization
```bash
python3 radp/digital_twin/agentic_mobility/examples/end_to_end_viz_example.py
```
- End-to-end: NL → mobility → topology → visualization
- Generates cell towers based on UE distribution
- Outputs: CSV files + PNG visualizations

### 4. Interactive Visualization Demo (Notebook)
```bash
jupyter notebook radp/digital_twin/agentic_mobility/examples/interactive_visualization_demo.ipynb
```
- Location validation with reverse geocoding
- World map visualization with spatial bounds
- Interactive Plotly dashboards (UE-wise & tick-wise views)
- Exports to standalone HTML

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Natural Language Parsing** | "Generate 100 UEs in Tokyo" → valid parameters |
| **Auto-Geocoding** | Location names → lat/lon bounds (cached) |
| **Context-Aware** | LLM infers distributions from context (e.g., "rush hour" → high car %) |
| **Distribution Tracking** | Tracks if distribution was "parsed" or "predicted" |
| **Self-Correction** | Auto-validates + retries with LLM suggestions (max 2 retries) |
| **Parallel Execution** | Location + parameter generation run simultaneously |
| **End-to-End** | Complete NL → DataFrame pipeline |

---

## Visualization Suite

### Static Visualizations
```python
from radp.digital_twin.agentic_mobility.visualization import (
    plot_ue_tracks,
    plot_ue_tracks_comparison,
)

# All UEs without legend
plot_ue_tracks(df, legend=False)

# Specific UEs with legend
plot_ue_tracks(df, legend=True, ue_ids=[0, 1, 2, 3, 4])

# Compare two scenarios
plot_ue_tracks_comparison(df1, df2, legend=False)
```

### Interactive Visualizations
```python
from radp.digital_twin.agentic_mobility.visualization import (
    plot_ue_wise_interactive,
    plot_tick_wise_interactive,
    validate_location_bounds,
    plot_bounds_on_map,
)

# UE-wise: dropdown to select UE, see full track
fig = plot_ue_wise_interactive(df)
fig.show()

# Tick-wise: slider to animate through time
fig = plot_tick_wise_interactive(df)
fig.show()

# Location validation with world map
validation = validate_location_bounds(metadata)
fig = plot_bounds_on_map(validation)
fig.show()
```

See `visualization/README.md` for detailed documentation.

---

## Architecture

```
Natural Language Query
    ↓
Parser (LLM) → QueryIntent
    ↓
Location Resolver ∥ Parameter Agent (parallel)
    ↓
Validator → Suggestion (if failed) → Retry
    ↓
RADP Formatter → JSON
    ↓
Integration → UETracksGenerator → DataFrame
    ↓
Topology Generator (optional)
    ↓
Visualization (optional)
```

---

## Output Structure

```python
{
  "status": "success" | "success_with_warnings" | "failed",
  "radp_params": {/* RADP JSON format */},
  "metadata": {
    "retry_count": 0,
    "query_intent": {
      "scenario_type": "urban",
      "location": "Tokyo",
      "num_ues": 100,
      "num_ticks": 50,
      "ue_distribution": {
        "source": "parsed" | "predicted",
        "distribution": {...}
      }
    },
    "spatial_bounds": {
      "requested": {...},
      "actual": {...}
    },
    "location_data": {...}
  }
}
```

---

## Technology Stack

- **LangGraph**: Workflow orchestration with parallel execution
- **LangChain**: LLM chain implementations
- **Groq API**: LLM provider (llama-3.1-70b-versatile)
- **Pydantic**: Structured output validation
- **Geopy/Nominatim**: Free geocoding (1 req/sec, cached)
- **Plotly**: Interactive visualizations

---

## Performance

| Metric | Value |
|--------|-------|
| Query Parsing | ~2-3s |
| Location Resolution | ~0.5-1s (cached: <0.01s) |
| Parameter Generation | ~2-3s |
| Validation | <0.1s |
| **Total (no retry)** | **~5-7s** |
| End-to-End (NL → DataFrame) | ~7-12s |

---

## File Structure

```
radp/digital_twin/agentic_mobility/
├── api.py                      # Public API
├── integration.py              # End-to-end integration
├── topology_generator.py       # Cell tower generation
├── models/                     # Pydantic models
├── chains/                     # LangChain implementations
├── nodes/                      # LangGraph nodes
├── graph/                      # Workflow orchestration
├── formatters/                 # RADP formatter
├── utils/                      # Utilities (LLM, geocoding, validators)
├── prompts/                    # LLM prompts
├── visualization/              # Visualization suite
│   ├── tracks.py               # Static matplotlib plots
│   ├── interactive.py          # Interactive Plotly plots
│   ├── geographic.py           # Location validation & maps
│   └── README.md               # Visualization docs
├── examples/
│   ├── gen_mobility_example.py         # NL → parameters
│   ├── end_to_end_example.py           # NL → DataFrame
│   ├── end_to_end_viz_example.py       # NL → mobility + topology + viz
│   └── interactive_visualization_demo.ipynb  # Interactive demos
└── tests/
```

---

## Support

- **Examples**: See `examples/` directory
- **Visualization**: See `visualization/README.md`
- **Tests**: Run `pytest radp/digital_twin/agentic_mobility/tests/`

---

**Status**: Production-Ready
**Last Updated**: 2025-10-30
