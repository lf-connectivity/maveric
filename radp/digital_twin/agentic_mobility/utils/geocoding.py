"""Geocoding utilities using geopy/Nominatim."""
import math
from typing import Optional

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from radp.digital_twin.agentic_mobility.models.location_data import LocationData
from radp.digital_twin.agentic_mobility.utils.cache import geocoding_cache


class GeocodingService:
    """Geocoding service wrapper for geopy/Nominatim."""

    def __init__(self, user_agent: str = "radp_agentic_mobility"):
        """Initialize geocoding service.

        Args:
            user_agent: User agent string for Nominatim
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)

    def geocode_location(self, location: str, bounds_size_km: float = 5.0) -> Optional[LocationData]:
        """Geocode a location string to geographic bounds.

        Args:
            location: Location string (e.g., "Tokyo", "New York City")
            bounds_size_km: Size of bounding box in kilometers (default 5km radius)

        Returns:
            LocationData object or None if geocoding fails
        """
        # Check cache first
        if geocoding_cache:
            cache_key = f"{location}_{bounds_size_km}"
            cached = geocoding_cache.get(cache_key)
            if cached:
                return cached

        try:
            # Geocode location
            location_data = self.geolocator.geocode(location, addressdetails=True)

            if not location_data:
                return None

            # Extract coordinates
            lat = location_data.latitude
            lon = location_data.longitude

            # Calculate bounds (approximate, using lat/lon degrees)
            # 1 degree of latitude ≈ 111 km
            # 1 degree of longitude ≈ 111 km * cos(latitude)
            lat_offset = bounds_size_km / 111.0
            lon_offset = bounds_size_km / (111.0 * abs(math.cos(math.radians(lat))))

            min_lat = lat - lat_offset
            max_lat = lat + lat_offset
            min_lon = lon - lon_offset
            max_lon = lon + lon_offset

            # Infer area type from address details
            area_type = self._infer_area_type(location_data.raw.get("address", {}))

            result = LocationData(
                center=(lat, lon),
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                area_type=area_type,
            )

            # Cache result
            if geocoding_cache:
                geocoding_cache.set(cache_key, result)

            return result

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error for '{location}': {e}")
            return None

    def _infer_area_type(self, address: dict) -> str:
        """Infer area type from address details.

        Args:
            address: Address dictionary from Nominatim

        Returns:
            Area type string ("urban", "suburban", "rural")
        """
        # Simple heuristic based on address components
        if "city" in address or "town" in address:
            # Check for city_district or neighbourhood to distinguish urban/suburban
            if "city_district" in address or "neighbourhood" in address:
                return "urban"
            return "suburban"
        elif "village" in address or "hamlet" in address:
            return "rural"
        else:
            # Default to suburban if unclear
            return "suburban"
