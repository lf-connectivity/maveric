# Agentic Mobility Visualization

Comprehensive visualization suite for UE mobility tracks with static matplotlib plots, interactive Plotly dashboards, and geographic validation.

---

## Quick Start

```python
from radp.digital_twin.agentic_mobility.visualization import (
    # Static plots
    plot_ue_tracks,
    plot_ue_tracks_comparison,

    # Interactive plots
    plot_ue_wise_interactive,
    plot_tick_wise_interactive,

    # Geographic validation
    validate_location_bounds,
    plot_bounds_on_map,
)

# Load data
import pandas as pd
df = pd.read_csv("generated_ues/agentic_mobility_75UE_50ticks.csv")

# Static: All UEs without legend (recommended for 20+ UEs)
plot_ue_tracks(df, legend=False)

# Static: Specific UEs with legend
plot_ue_tracks(df, legend=True, ue_ids=[0, 1, 2, 3, 4])

# Interactive: UE-wise view with dropdown
fig = plot_ue_wise_interactive(df)
fig.show()

# Interactive: Tick-wise animation with slider
fig = plot_tick_wise_interactive(df)
fig.show()

# Geographic validation
validation = validate_location_bounds(metadata)
fig = plot_bounds_on_map(validation)
fig.show()
```

---

## Static Visualizations (Matplotlib)

### `plot_ue_tracks()`

Main plotting function for UE mobility tracks.

**Parameters:**
```python
plot_ue_tracks(
    df,                          # DataFrame with ['mock_ue_id', 'tick', 'lat', 'lon']
    legend=False,                # Show/hide legend (False recommended for 20+ UEs)
    ue_ids=None,                 # List of UE IDs to plot, or None for all
    figsize=(12, 8),             # Figure size
    title=None,                  # Custom title
    show_start_points=True,      # Mark starting points
    show_end_points=False,       # Mark ending points
)
```

**Examples:**
```python
# All UEs, no legend
plot_ue_tracks(df, legend=False)

# First 5 UEs with legend
plot_ue_tracks(df, legend=True, ue_ids=[0, 1, 2, 3, 4])

# Single UE with end marker
plot_ue_tracks(df, legend=True, ue_ids=[7], show_end_points=True)
```

### `plot_ue_tracks_comparison()`

Side-by-side comparison of two datasets.

```python
# Compare two scenarios
plot_ue_tracks_comparison(
    df1, df2,
    legend=False,
    ue_ids=[0, 1, 2],  # Optional: filter to specific UEs
    titles=("Urban", "Rural")
)
```

---

## Interactive Visualizations (Plotly)

### `plot_ue_wise_interactive()`

Interactive plot with dropdown menu to select UE and view full track.

**Features:**
- Dropdown to switch between UEs
- Start point (green circle) and end point (red square) markers
- Hover for details (UE ID, tick, lat/lon)
- Exportable to HTML

```python
fig = plot_ue_wise_interactive(
    df,
    ue_ids=None,         # List of UEs or None for all
    show_arrows=True,    # Show direction arrows
    figsize=(1000, 700)
)
fig.show()

# Export to HTML
fig.write_html("ue_tracks.html")
```

### `plot_tick_wise_interactive()`

Interactive animation with slider to view all UEs at each tick.

**Features:**
- Slider to select tick (with PLAY button for animation)
- Color-coded by UE ID
- Hover for UE details
- Optional trail mode

```python
fig = plot_tick_wise_interactive(
    df,
    initial_tick=0,      # Starting tick
    color_by_ue=True,    # Color by UE ID
    show_trails=False    # Show movement trails
)
fig.show()
```

---

## Geographic Validation

### `validate_location_bounds()`

Validates that generated spatial bounds match the intended location using reverse geocoding.

**Process:**
1. Extracts 5 points: center + 4 corners (NW, NE, SW, SE)
2. Reverse geocodes each point (OpenStreetMap/Nominatim)
3. Compares detected locations vs. query intent
4. Returns validation result with confidence scores

```python
# Assuming metadata from AgenticMobilityIntegration
validation_result = validate_location_bounds(
    metadata,
    threshold=0.7  # Minimum confidence score for match
)

print(validation_result['is_match'])              # True/False
print(validation_result['overall_confidence'])    # 0.0 to 1.0
print(validation_result['detected_locations'])    # 5 geocoded points
print(validation_result['warnings'])              # List of issues
```

### `plot_bounds_on_map()`

Visualizes spatial bounds on an interactive world map.

**Features:**
- Red rectangle showing spatial bounds
- 5 validation points with markers (blue center, green corners)
- Legend with full addresses
- Auto-zoom based on coordinate span
- Hover for detailed info

```python
validation_result = validate_location_bounds(metadata)

fig = plot_bounds_on_map(
    validation_result,
    title="Spatial Bounds Validation",
    figsize=(1200, 800)
)
fig.show()

# Export to HTML
fig.write_html("bounds_map.html")
```

---

## Interactive Demo Notebook

Run the comprehensive interactive demo:

```bash
jupyter notebook radp/digital_twin/agentic_mobility/examples/interactive_visualization_demo.ipynb
```

**Includes:**
1. **Location Validation**: Reverse geocoding with confidence scores
2. **World Map View**: Spatial bounds with 5 validation points
3. **UE-wise Dashboard**: Dropdown menu to explore individual UE tracks
4. **Tick-wise Animation**: Slider to animate through time

**Outputs:**
- Interactive Plotly figures (embedded in notebook)
- Exportable to standalone HTML files
- Full validation reports

---

## Solutions to Common Issues

### Issue 1: Legend Too Long (75+ UEs)

```python
# Solution 1: Disable legend (recommended)
plot_ue_tracks(df, legend=False)

# Solution 2: Plot subset with legend
plot_ue_tracks(df, legend=True, ue_ids=[0, 1, 2, 3, 4])
```

### Issue 2: Filtering Specific UEs

```python
# Correct way
plot_ue_tracks(df, legend=True, ue_ids=[5, 10, 15])

# Don't filter DataFrame first (causes batch detection issues)
# filtered_df = df[df['mock_ue_id'] == 5]  # Avoid this!
```

### Issue 3: Location Validation Fails

**Possible causes:**
- Invalid coordinates (e.g., ocean, unpopulated areas)
- Typo in location name ("Argentica" instead of "Argentina")
- Coordinates don't match query intent

**Solution:** Check the validation warnings and visualize on map:
```python
validation = validate_location_bounds(metadata)
print(validation['warnings'])

# Visualize to verify
fig = plot_bounds_on_map(validation)
fig.show()
```

---

## File Structure

```
visualization/
├── __init__.py              # Public exports
├── tracks.py                # Static matplotlib plots
├── interactive.py           # Interactive Plotly plots
├── geographic.py            # Location validation & maps
├── legacy.py                # Backward compatibility
└── README.md                # This file
```

---

## Performance

- **Static plots**: Fast (<1s for 100 UEs)
- **Interactive plots**: ~1-2s for 100 UEs, 50 ticks
- **Location validation**: ~5-7s (5 geocoding API calls with rate limiting)
- **World map rendering**: <1s

---

## Tips & Best Practices

1. **For 20+ UEs**: Use `legend=False` in static plots
2. **For detailed analysis**: Use `ue_ids` to focus on specific UEs
3. **For presentations**: Use interactive plots with selective UE filtering
4. **For comparisons**: Use `plot_ue_tracks_comparison()` with matching `ue_ids`
5. **For location debugging**: Always visualize bounds on map with `plot_bounds_on_map()`

---

**Status**: Production-Ready
**Last Updated**: 2025-10-30
