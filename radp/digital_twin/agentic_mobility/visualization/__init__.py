"""Visualization suite for agentic mobility simulations."""

# Enhanced track visualization (matplotlib)
# Geographic validation
from radp.digital_twin.agentic_mobility.visualization.geographic import (
    plot_bounds_on_map,
    reverse_geocode_bounds,
    validate_location_bounds,
)

# Interactive visualization (plotly)
from radp.digital_twin.agentic_mobility.visualization.interactive import (
    plot_tick_wise_interactive,
    plot_ue_wise_interactive,
)

# Legacy functions (maintained for backward compatibility)
from radp.digital_twin.agentic_mobility.visualization.legacy import plot_ue_tracks as legacy_plot_ue_tracks
from radp.digital_twin.agentic_mobility.visualization.legacy import plot_ue_tracks_on_axis, plot_ue_tracks_side_by_side
from radp.digital_twin.agentic_mobility.visualization.tracks import plot_ue_tracks, plot_ue_tracks_comparison

__all__ = [
    # Enhanced matplotlib visualization
    "plot_ue_tracks",
    "plot_ue_tracks_comparison",
    # Interactive plotly visualization
    "plot_ue_wise_interactive",
    "plot_tick_wise_interactive",
    # Geographic validation
    "validate_location_bounds",
    "reverse_geocode_bounds",
    "plot_bounds_on_map",
    # Legacy functions
    "legacy_plot_ue_tracks",
    "plot_ue_tracks_side_by_side",
    "plot_ue_tracks_on_axis",
]
