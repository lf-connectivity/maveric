"""Topology generator for cell tower placement based on LLM-generated parameters."""
import math
from typing import List

import numpy as np
import pandas as pd

from radp.digital_twin.agentic_mobility.chains.topology_chain import TopologyChain
from radp.digital_twin.agentic_mobility.models.topology_params import TopologyParams


class TopologyGenerator:
    """Generates cell topology data using LLM-determined parameters."""

    @staticmethod
    def generate_from_llm(
        area_type: str,
        num_ues: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        raw_query: str,
    ) -> pd.DataFrame:
        """Generate cell topology using LLM to determine parameters.

        Args:
            area_type: Area type (urban, suburban, rural, highway, mixed)
            num_ues: Total number of UEs in the simulation
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary
            raw_query: Original user query for context

        Returns:
            DataFrame with columns: cell_lat, cell_lon, cell_id, cell_az_deg, cell_carrier_freq_mhz
        """
        # Use LLM to generate topology parameters
        topology_chain = TopologyChain()
        params = topology_chain.generate(
            area_type=area_type,
            num_ues=num_ues,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            raw_query=raw_query,
        )

        print(f"\n[Topology Generation - LLM Output]")
        print(f"  Reasoning: {params.reasoning}")
        print(f"  num_cells: {params.num_cells}")
        print(f"  frequency: {params.cell_carrier_freq_mhz} MHz")
        print(f"  azimuth_strategy: {params.azimuth_strategy}")
        print(f"  cell_placement_strategy: {params.cell_placement_strategy}")

        # Calculate actual sites
        if params.azimuth_strategy == "sectored":
            num_sites = max(1, params.num_cells // 3)
            print(f"  → Will generate {num_sites} sites × 3 sectors = {num_sites * 3} total cells")
        else:
            num_sites = params.num_cells
            print(f"  → Will generate {num_sites} sites")

        # Generate topology using LLM parameters
        return TopologyGenerator.generate_with_params(
            params=params,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
        )

    @staticmethod
    def generate_with_params(
        params: TopologyParams,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> pd.DataFrame:
        """Generate cell topology using provided parameters.

        Args:
            params: TopologyParams with generation parameters
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary

        Returns:
            DataFrame with cell topology data
        """
        # Calculate cell sites based on azimuth strategy
        # num_cells represents TOTAL sectors, so we need to calculate actual sites
        if params.azimuth_strategy == "sectored":
            sectors_per_site = 3
            # num_cells = total sectors, so divide by 3 to get number of sites
            num_sites = max(1, params.num_cells // sectors_per_site)
        else:
            sectors_per_site = 1
            num_sites = params.num_cells

        # Generate site locations based on placement strategy
        if params.cell_placement_strategy == "grid":
            site_locations = TopologyGenerator._generate_grid_locations(
                min_lat, max_lat, min_lon, max_lon, num_sites
            )
        elif params.cell_placement_strategy == "cluster":
            site_locations = TopologyGenerator._generate_cluster_locations(
                min_lat, max_lat, min_lon, max_lon, num_sites
            )
        elif params.cell_placement_strategy == "boundary":
            site_locations = TopologyGenerator._generate_boundary_locations(
                min_lat, max_lat, min_lon, max_lon, num_sites
            )
        else:
            # Default to grid
            site_locations = TopologyGenerator._generate_grid_locations(
                min_lat, max_lat, min_lon, max_lon, num_sites
            )

        # Generate cell data
        cells = []
        cell_counter = 1

        for site_lat, site_lon in site_locations:
            # Determine azimuths based on strategy
            if params.azimuth_strategy == "sectored":
                azimuths = [0, 120, 240]
            elif params.azimuth_strategy == "random":
                azimuths = [np.random.randint(0, 360)]
            else:  # omnidirectional
                azimuths = [0]

            for azimuth in azimuths:
                cells.append(
                    {
                        "cell_lat": round(site_lat, 6),
                        "cell_lon": round(site_lon, 6),
                        "cell_id": f"cell_{cell_counter}",
                        "cell_az_deg": azimuth,
                        "cell_carrier_freq_mhz": params.cell_carrier_freq_mhz,
                    }
                )
                cell_counter += 1

                # Stop if we've reached the desired number of cells
                if cell_counter > params.num_cells:
                    break

            if cell_counter > params.num_cells:
                break

        return pd.DataFrame(cells)

    @staticmethod
    def _generate_grid_locations(
        min_lat: float, max_lat: float, min_lon: float, max_lon: float, num_sites: int
    ) -> List[tuple]:
        """Generate cell site locations using industry-standard hexagonal grid.

        Based on cellular network planning theory:
        - Hexagonal cells provide optimal coverage efficiency
        - Inter-site distance (ISD) determines cell radius
        - Uses offset coordinate system for hexagonal tiling
        - Reference: "Cellular Network Planning" by Mischa Schwartz
        """
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        center_lat = (min_lat + max_lat) / 2

        # Check if area is very narrow (aspect ratio > 5:1)
        # Convert to approximate km for comparison
        lat_km = lat_range * 111
        lon_km = lon_range * 111 * math.cos(math.radians(center_lat))
        aspect_ratio = max(lat_km / lon_km, lon_km / lat_km) if min(lat_km, lon_km) > 0 else 1

        # For narrow corridors (highways, rivers, etc.), use linear placement
        if aspect_ratio > 5:
            print(f"  [Info] Narrow area detected (aspect {aspect_ratio:.1f}:1), using linear placement")
            return TopologyGenerator._generate_linear_locations(
                min_lat, max_lat, min_lon, max_lon, num_sites
            )

        # Standard hexagonal grid for regular areas
        # Calculate Inter-Site Distance (ISD) based on coverage area
        area = lat_range * lon_range

        # Hexagonal cell: Area = (3√3/2) × r²  where r = cell radius
        # ISD (distance between centers) ≈ √(2 × area / (√3 × num_sites))
        isd_lat = math.sqrt(2 * area / (math.sqrt(3) * num_sites))
        isd_lon = isd_lat  # Keep cells roughly circular

        # Hexagonal grid: row offset and vertical spacing
        row_offset = isd_lon / 2
        row_spacing = isd_lat * math.sqrt(3) / 2

        locations = []
        row = 0
        lat = min_lat + row_spacing / 2

        # Generate hexagonal grid using offset coordinates
        while lat <= max_lat and len(locations) < num_sites:
            # Alternate rows are offset horizontally (creates hexagonal pattern)
            lon_start = min_lon + (row_offset if row % 2 == 1 else 0) + isd_lon / 2
            lon = lon_start

            while lon <= max_lon and len(locations) < num_sites:
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    locations.append((lat, lon))
                lon += isd_lon

            lat += row_spacing
            row += 1

        # If hexagonal grid didn't generate enough sites, use simple grid
        if len(locations) < num_sites:
            print(f"  [Info] Hexagonal grid insufficient ({len(locations)}/{num_sites}), using simple grid")
            return TopologyGenerator._generate_simple_grid(min_lat, max_lat, min_lon, max_lon, num_sites)

        return locations[:num_sites]

    @staticmethod
    def _generate_linear_locations(
        min_lat: float, max_lat: float, min_lon: float, max_lon: float, num_sites: int
    ) -> List[tuple]:
        """Generate linear cell placement for narrow corridors (highways, railways, etc.)."""
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        # Determine if corridor is horizontal or vertical
        if lat_range > lon_range:
            # Vertical corridor - space along latitude
            lat_points = np.linspace(min_lat, max_lat, num_sites)
            lon_center = (min_lon + max_lon) / 2
            locations = [(lat, lon_center) for lat in lat_points]
        else:
            # Horizontal corridor - space along longitude
            lon_points = np.linspace(min_lon, max_lon, num_sites)
            lat_center = (min_lat + max_lat) / 2
            locations = [(lat_center, lon) for lon in lon_points]

        return locations

    @staticmethod
    def _generate_simple_grid(
        min_lat: float, max_lat: float, min_lon: float, max_lon: float, num_sites: int
    ) -> List[tuple]:
        """Generate simple square grid pattern (fallback)."""
        grid_size = math.ceil(math.sqrt(num_sites))

        lat_points = np.linspace(min_lat, max_lat, grid_size)
        lon_points = np.linspace(min_lon, max_lon, grid_size)

        locations = []
        for lat in lat_points:
            for lon in lon_points:
                locations.append((lat, lon))
                if len(locations) >= num_sites:
                    break
            if len(locations) >= num_sites:
                break

        return locations[:num_sites]

    @staticmethod
    def _generate_cluster_locations(
        min_lat: float, max_lat: float, min_lon: float, max_lon: float, num_sites: int
    ) -> List[tuple]:
        """Generate cell site locations clustered around center (hotspot)."""
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        locations = []

        # First cell at center
        if num_sites >= 1:
            locations.append((center_lat, center_lon))

        # Remaining cells in concentric circles around center
        num_rings = math.ceil(math.sqrt(num_sites))
        cells_per_ring = num_sites // num_rings if num_rings > 1 else 0

        for ring in range(1, num_rings + 1):
            radius_lat = (lat_range / (num_rings + 1)) * ring * 0.4
            radius_lon = (lon_range / (num_rings + 1)) * ring * 0.4

            angles = np.linspace(0, 2 * np.pi, cells_per_ring + 1)[:-1]

            for angle in angles:
                lat = center_lat + radius_lat * np.sin(angle)
                lon = center_lon + radius_lon * np.cos(angle)

                # Ensure within boundaries
                lat = np.clip(lat, min_lat, max_lat)
                lon = np.clip(lon, min_lon, max_lon)

                locations.append((lat, lon))

                if len(locations) >= num_sites:
                    break

            if len(locations) >= num_sites:
                break

        return locations[:num_sites]

    @staticmethod
    def _generate_boundary_locations(
        min_lat: float, max_lat: float, min_lon: float, max_lon: float, num_sites: int
    ) -> List[tuple]:
        """Generate cell site locations around the boundary perimeter."""
        # Distribute cells around the perimeter (rectangle)
        perimeter_points = []

        cells_per_side = math.ceil(num_sites / 4)

        # Top edge
        for i in range(cells_per_side):
            lon = min_lon + (max_lon - min_lon) * i / max(1, cells_per_side - 1) if cells_per_side > 1 else (min_lon + max_lon) / 2
            perimeter_points.append((max_lat, lon))

        # Right edge
        for i in range(cells_per_side):
            lat = max_lat - (max_lat - min_lat) * i / max(1, cells_per_side - 1) if cells_per_side > 1 else (min_lat + max_lat) / 2
            perimeter_points.append((lat, max_lon))

        # Bottom edge
        for i in range(cells_per_side):
            lon = max_lon - (max_lon - min_lon) * i / max(1, cells_per_side - 1) if cells_per_side > 1 else (min_lon + max_lon) / 2
            perimeter_points.append((min_lat, lon))

        # Left edge
        for i in range(cells_per_side):
            lat = min_lat + (max_lat - min_lat) * i / max(1, cells_per_side - 1) if cells_per_side > 1 else (min_lat + max_lat) / 2
            perimeter_points.append((lat, min_lon))

        # Return only the requested number
        return perimeter_points[:num_sites]


def generate_topology_from_mobility_data(
    mobility_df: pd.DataFrame,
    area_type: str = "suburban",
    raw_query: str = "",
) -> pd.DataFrame:
    """Generate topology data from mobility DataFrame using LLM intelligence.

    This is the main convenience function that extracts boundaries from mobility data
    and uses LLM to generate contextually appropriate topology parameters.

    Args:
        mobility_df: DataFrame with 'lat' and 'lon' columns from mobility simulation
        area_type: Type of area - "urban", "suburban", "rural", etc.
        raw_query: Original user query for additional context

    Returns:
        DataFrame with cell topology data
    """
    # Extract boundaries from mobility data
    min_lat = mobility_df["lat"].min()
    max_lat = mobility_df["lat"].max()
    min_lon = mobility_df["lon"].min()
    max_lon = mobility_df["lon"].max()

    # Get number of UEs from mobility data
    num_ues = mobility_df["mock_ue_id"].nunique() if "mock_ue_id" in mobility_df.columns else len(mobility_df)

    # Generate topology using LLM
    return TopologyGenerator.generate_from_llm(
        area_type=area_type,
        num_ues=num_ues,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        raw_query=raw_query,
    )
