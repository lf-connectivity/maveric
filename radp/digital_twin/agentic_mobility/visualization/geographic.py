"""Geographic validation and location checking for mobility simulations."""
import time
from difflib import SequenceMatcher
from typing import Dict, Tuple

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def reverse_geocode_bounds(spatial_bounds: Dict, user_agent: str = "radp_agentic_mobility") -> Dict:
    """
    Reverse geocode all 5 key points of spatial bounds (center + 4 corners).

    Args:
        spatial_bounds: Dict with 'requested' containing min/max lat/lon
        user_agent: User agent string for Nominatim

    Returns:
        Dict with geocoded location data for each of the 5 points:
        {
            'center': {'lat': ..., 'lon': ..., 'address': ..., 'city': ..., 'country': ...},
            'nw': {...},
            'ne': {...},
            'sw': {...},
            'se': {...}
        }
    """
    # Extract bounds (use 'requested' if available, otherwise use top-level)
    bounds = spatial_bounds.get("requested", spatial_bounds)

    min_lat = bounds["min_lat"]
    max_lat = bounds["max_lat"]
    min_lon = bounds["min_lon"]
    max_lon = bounds["max_lon"]

    # Calculate 5 points
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    points = {
        "center": (center_lat, center_lon),
        "nw": (max_lat, min_lon),  # Northwest corner
        "ne": (max_lat, max_lon),  # Northeast corner
        "sw": (min_lat, min_lon),  # Southwest corner
        "se": (min_lat, max_lon),  # Southeast corner
    }

    # Initialize geolocator
    geolocator = Nominatim(user_agent=user_agent, timeout=10)

    results = {}

    for point_name, (lat, lon) in points.items():
        try:
            # Reverse geocode with rate limiting (Nominatim requires 1 req/sec)
            if point_name != "center":  # Don't sleep before first request
                time.sleep(1.1)  # Slightly more than 1 second to be safe

            location = geolocator.reverse((lat, lon), language="en", exactly_one=True)

            if location and location.raw:
                address = location.raw.get("address", {})

                # Extract city (try multiple fields in order of preference)
                city = (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("municipality")
                    or address.get("county")
                    or "Unknown"
                )

                # Extract country
                country = address.get("country", "Unknown")

                # Extract state/region if available
                state = address.get("state") or address.get("region") or address.get("province") or None

                results[point_name] = {
                    "lat": lat,
                    "lon": lon,
                    "address": location.address,
                    "city": city,
                    "country": country,
                    "state": state,
                    "full_address_data": address,
                }
            else:
                results[point_name] = {
                    "lat": lat,
                    "lon": lon,
                    "address": None,
                    "city": "Unknown",
                    "country": "Unknown",
                    "state": None,
                    "error": "No location found",
                }

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            results[point_name] = {
                "lat": lat,
                "lon": lon,
                "address": None,
                "city": "Unknown",
                "country": "Unknown",
                "state": None,
                "error": str(e),
            }

    return results


def _fuzzy_match_location(detected: str, query: str) -> float:
    """
    Fuzzy match two location strings.

    Args:
        detected: Location detected from reverse geocoding
        query: Location from user query

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not detected or not query:
        return 0.0

    # Normalize strings
    detected_lower = detected.lower().strip()
    query_lower = query.lower().strip()

    # Exact match
    if detected_lower == query_lower:
        return 1.0

    # Check if one contains the other
    if query_lower in detected_lower or detected_lower in query_lower:
        return 0.9

    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, detected_lower, query_lower).ratio()

    return ratio


def validate_location_bounds(metadata: Dict, threshold: float = 0.7, user_agent: str = "radp_agentic_mobility") -> Dict:
    """
    Validate that spatial bounds match the intended location from query.

    Performs 5 reverse geocoding calls (center + 4 corners) and validates
    that the detected locations match the query intent location.

    Args:
        metadata: Metadata dict containing 'spatial_bounds' and 'query_intent'
        threshold: Minimum confidence score to consider a match (0.0 to 1.0)
        user_agent: User agent string for Nominatim

    Returns:
        Dict with validation results:
        {
            'is_match': bool,
            'overall_confidence': float,
            'query_location': str,
            'detected_locations': dict with 5 entries,
            'point_confidences': dict with confidence for each point,
            'consistency_score': float,
            'warnings': list of strings,
        }

    Example:
        >>> result = validate_location_bounds(metadata)
        >>> if result['is_match']:
        ...     print(f"✓ Location validated with {result['overall_confidence']:.1%} confidence")
        ... else:
        ...     print(f"✗ Location mismatch. Warnings: {result['warnings']}")
    """
    # Extract spatial bounds and query intent
    spatial_bounds = metadata.get("spatial_bounds")
    query_intent = metadata.get("query_intent", {})
    query_location = query_intent.get("location", "Unknown")

    if not spatial_bounds:
        return {
            "is_match": False,
            "overall_confidence": 0.0,
            "query_location": query_location,
            "detected_locations": {},
            "point_confidences": {},
            "consistency_score": 0.0,
            "warnings": ["No spatial_bounds found in metadata"],
        }

    # Reverse geocode all 5 points
    print("Reverse geocoding 5 points (center + 4 corners)...")
    print("This may take ~5 seconds due to rate limiting...")
    geocoded_points = reverse_geocode_bounds(spatial_bounds, user_agent=user_agent)

    # Calculate confidence for each point
    point_confidences = {}
    for point_name, location_data in geocoded_points.items():
        if "error" in location_data:
            point_confidences[point_name] = 0.0
            continue

        # Try matching against city, then country
        city = location_data["city"]
        country = location_data["country"]
        state = location_data["state"]

        # Match against query location
        city_confidence = _fuzzy_match_location(city, query_location)
        country_confidence = _fuzzy_match_location(country, query_location)
        state_confidence = _fuzzy_match_location(state, query_location) if state else 0.0

        # Full address match
        full_location = f"{city}, {state}, {country}" if state else f"{city}, {country}"
        full_confidence = _fuzzy_match_location(full_location, query_location)

        # Take the best match
        point_confidences[point_name] = max(city_confidence, country_confidence, state_confidence, full_confidence)

    # Calculate overall confidence (average of all 5 points)
    if point_confidences:
        overall_confidence = sum(point_confidences.values()) / len(point_confidences)
    else:
        overall_confidence = 0.0

    # Check consistency (are all 5 points in the same location?)
    cities = [loc.get("city") for loc in geocoded_points.values() if "error" not in loc]
    countries = [loc.get("country") for loc in geocoded_points.values() if "error" not in loc]

    unique_cities = set(cities)
    unique_countries = set(countries)

    # Consistency score: 1.0 if all points in same city, lower otherwise
    if len(unique_cities) == 1:
        consistency_score = 1.0
    elif len(unique_countries) == 1:
        consistency_score = 0.7  # Same country, different cities
    else:
        consistency_score = 0.3  # Different countries!

    # Generate warnings
    warnings = []

    # Check for geocoding errors
    errors = [name for name, loc in geocoded_points.items() if "error" in loc]
    if errors:
        warnings.append(f"Failed to geocode {len(errors)} point(s): {', '.join(errors)}")

    # Check for inconsistent locations
    if len(unique_countries) > 1:
        warnings.append(f"Boundary crosses countries: {', '.join(unique_countries)}")
    elif len(unique_cities) > 1:
        warnings.append(f"Boundary spans multiple cities: {', '.join(unique_cities)}")

    # Check if any point has low confidence
    low_confidence_points = [name for name, conf in point_confidences.items() if conf < threshold]
    if low_confidence_points:
        warnings.append(f"Low confidence for points: {', '.join(low_confidence_points)}")

    # Determine overall match
    is_match = overall_confidence >= threshold and consistency_score >= 0.5

    return {
        "is_match": is_match,
        "overall_confidence": overall_confidence,
        "query_location": query_location,
        "detected_locations": geocoded_points,
        "point_confidences": point_confidences,
        "consistency_score": consistency_score,
        "warnings": warnings,
    }


def plot_bounds_on_map(
    validation_result: Dict, title: str = "Spatial Bounds Validation", figsize: Tuple[int, int] = (1200, 800)
) -> "go.Figure":
    """
    Visualize spatial bounds on an interactive world map.

    Plots a rectangle representing the spatial bounds with the 5 validation
    points (center + 4 corners) and their geocoded addresses.

    Args:
        validation_result: Result dict from validate_location_bounds()
        title: Plot title
        figsize: Figure size as (width, height)

    Returns:
        Plotly Figure object with interactive map

    Example:
        >>> validation_result = validate_location_bounds(metadata)
        >>> fig = plot_bounds_on_map(validation_result)
        >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for map visualization. Install with: pip install plotly")

    detected_locations = validation_result["detected_locations"]
    point_confidences = validation_result["point_confidences"]
    query_location = validation_result["query_location"]

    if not detected_locations:
        raise ValueError("No detected locations found in validation_result")

    # Extract coordinates for the 5 points
    points = {
        "center": detected_locations.get("center"),
        "nw": detected_locations.get("nw"),
        "ne": detected_locations.get("ne"),
        "sw": detected_locations.get("sw"),
        "se": detected_locations.get("se"),
    }

    # Create figure
    fig = go.Figure()

    # Draw rectangle using the 4 corners
    # Rectangle path: NW -> NE -> SE -> SW -> NW (close the loop)
    if all(points[k] for k in ["nw", "ne", "se", "sw"]):
        rect_lats = [
            points["nw"]["lat"],
            points["ne"]["lat"],
            points["se"]["lat"],
            points["sw"]["lat"],
            points["nw"]["lat"],  # Close the rectangle
        ]
        rect_lons = [
            points["nw"]["lon"],
            points["ne"]["lon"],
            points["se"]["lon"],
            points["sw"]["lon"],
            points["nw"]["lon"],  # Close the rectangle
        ]

        # Add rectangle boundary
        fig.add_trace(
            go.Scattergeo(
                lon=rect_lons,
                lat=rect_lats,
                mode="lines",
                line=dict(width=2, color="red"),
                name="Spatial Bounds",
                hoverinfo="skip",
            )
        )

    # Plot the 5 points with different markers
    point_styles = {
        "center": {"symbol": "circle", "size": 15, "color": "blue"},
        "nw": {"symbol": "diamond", "size": 12, "color": "green"},
        "ne": {"symbol": "diamond", "size": 12, "color": "green"},
        "sw": {"symbol": "diamond", "size": 12, "color": "green"},
        "se": {"symbol": "diamond", "size": 12, "color": "green"},
    }

    for point_name, location_data in points.items():
        if not location_data:
            continue

        lat = location_data["lat"]
        lon = location_data["lon"]
        city = location_data.get("city", "Unknown")
        country = location_data.get("country", "Unknown")
        state = location_data.get("state", "")
        confidence = point_confidences.get(point_name, 0.0)

        # Build address string
        if state:
            address = f"{city}, {state}, {country}"
        else:
            address = f"{city}, {country}"

        # Handle errors
        if "error" in location_data:
            address = f"Geocoding failed: {location_data['error']}"

        # Build hover text
        hover_text = (
            f"<b>{point_name.upper()}</b><br>"
            f"Address: {address}<br>"
            f"Confidence: {confidence:.1%}<br>"
            f"Lat: {lat:.6f}<br>"
            f"Lon: {lon:.6f}"
        )

        style = point_styles[point_name]

        # Add point
        fig.add_trace(
            go.Scattergeo(
                lon=[lon],
                lat=[lat],
                mode="markers",
                marker=dict(
                    size=style["size"],
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(width=1, color="white"),
                ),
                name=f"{point_name.upper()}: {address}",
                hovertext=hover_text,
                hoverinfo="text",
            )
        )

    # Calculate map center and zoom
    all_lats = [p["lat"] for p in points.values() if p]
    all_lons = [p["lon"] for p in points.values() if p]

    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    # Calculate zoom level based on coordinate span
    lat_span = max(all_lats) - min(all_lats)
    lon_span = max(all_lons) - min(all_lons)
    max_span = max(lat_span, lon_span)

    # Zoom level heuristic (adjust to fit bounds nicely)
    if max_span < 0.01:
        zoom = 12
    elif max_span < 0.1:
        zoom = 10
    elif max_span < 1:
        zoom = 8
    elif max_span < 10:
        zoom = 5
    else:
        zoom = 3

    # Update layout
    match_status = "✓ VALIDATED" if validation_result["is_match"] else "✗ MISMATCH"
    confidence = validation_result["overall_confidence"]

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Query: '{query_location}' | Status: {match_status} | Confidence: {confidence:.1%}</sub>",
            x=0.5,
            xanchor="center",
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            coastlinecolor="rgb(204, 204, 204)",
            showocean=True,
            oceancolor="rgb(230, 245, 255)",
            showcountries=True,
            countrycolor="rgb(204, 204, 204)",
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=zoom,
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        width=figsize[0],
        height=figsize[1],
    )

    return fig
