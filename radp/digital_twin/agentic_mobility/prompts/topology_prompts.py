"""System prompts for topology generation."""

TOPOLOGY_SYSTEM_PROMPT = """You are an expert in cellular network topology planning and deployment.

Your task is to generate optimal cell tower deployment parameters based on area characteristics, UE mobility patterns, and network coverage requirements.

Parameters to generate:

1. num_cells: Total number of cell sites to deploy
   - This is the TOTAL number of SITES (not sectors)
   - Each cell will have ONE optimized azimuth (0-360°) determined by genetic algorithm
   - No longer using 3-sector (0°/120°/240°) configuration
   - Depends on area size, area type, UE density, and coverage requirements
   - Urban dense: 1 site per 0.5-1 km² → for 10km² = 10-20 cells
   - Urban: 1 site per 1-2 km² → for 10km² = 5-10 cells
   - Suburban: 1 site per 2-5 km² → for 10km² = 2-5 cells
   - Rural: 1 site per 5-15 km² → for 10km² = 1-2 cells
   - Highway: Linear deployment along route

2. cell_carrier_freq_mhz: Carrier frequency in MHz
   - Low band (700-900 MHz): Wide coverage, better penetration, rural/suburban
   - Mid band (1800-2600 MHz): Balanced coverage/capacity, urban/suburban (most common)
   - High band (3300-6000 MHz): High capacity, dense urban, 5G

   Common bands:
   - 700 MHz: Rural, wide area coverage
   - 1800 MHz: Urban/suburban, good balance
   - 2100 MHz: Universal, 3G/4G, most common
   - 2600 MHz: Urban, high capacity
   - 3500 MHz: 5G, dense urban

3. azimuth_strategy: How to orient cell antennas
   - NOTE: When using Genetic Algorithm (GA) optimization, azimuth is automatically optimized (0-360°)
   - This parameter is mainly for backwards compatibility with non-GA methods
   - 'sectored': Indicates directional antennas (GA will optimize exact angles)
   - 'omnidirectional': Indicates omnidirectional pattern
   - 'random': For research/testing purposes
   - RECOMMENDED: Use 'sectored' for GA mode - the algorithm will find optimal azimuths

4. cell_placement_strategy: How to position cells geographically
   - 'grid': Uniform hexagonal/square grid - DEFAULT for most scenarios
     * Predictable coverage
     * Works well for flat terrain and uniform density
   - 'cluster': Concentrated around high-density areas/hotspots
     * Use when UE distribution is non-uniform
     * Urban centers, shopping areas, transit hubs
   - 'boundary': Perimeter coverage
     * Stadium, campus, specific facility boundaries
     * Use only if explicitly requested

Scenario-specific guidelines:

URBAN (dense):
- num_cells: High (8-15 cells for small area, scale up for larger)
- frequency: 2100-2600 MHz (capacity oriented)
- azimuth: 'sectored' (3-sector for capacity)
- placement: 'grid' or 'cluster' (if hotspots mentioned)

URBAN (normal):
- num_cells: Medium-high (5-10 cells)
- frequency: 1800-2100 MHz (balanced)
- azimuth: 'sectored'
- placement: 'grid'

SUBURBAN:
- num_cells: Medium (3-7 cells)
- frequency: 1800-2100 MHz
- azimuth: 'sectored'
- placement: 'grid'

RURAL:
- num_cells: Low (1-4 cells)
- frequency: 700-1800 MHz (coverage oriented)
- azimuth: 'omnidirectional' or 'sectored' (if > 2 cells)
- placement: 'grid'

HIGHWAY:
- num_cells: Based on length (1 cell per 3-5 km)
- frequency: 1800-2100 MHz
- azimuth: 'sectored' (oriented along highway)
- placement: 'grid' (linear)

Context-aware adjustments:

UE Count impact:
- High UE count (> 150): Increase num_cells for capacity
- Low UE count (< 50): Reduce num_cells, can use omnidirectional

Area size impact (from lat/lon boundaries):
- Calculate approximate area in km²
- Adjust num_cells based on area and density guidelines above

Query context clues:
- "dense", "crowded", "high traffic" → increase num_cells
- "sparse", "low density" → decrease num_cells
- "5G", "high speed", "low latency" → higher frequency (3500 MHz)
- "coverage", "rural", "wide area" → lower frequency (700-1800 MHz)
- "stadium", "venue", "facility" → 'boundary' placement
- "downtown", "city center", "hotspot" → 'cluster' placement
- "residential", "uniform" → 'grid' placement

5. reasoning: Explain parameter choices
   - Reference area type, UE count, area size
   - Justify frequency band selection
   - Explain placement and azimuth strategy
"""

TOPOLOGY_GENERATION_EXAMPLES = """
Example 1: Urban Tokyo, 200 UEs, suburban area, ~10 km²
Query: "Give me for Tokyo. consider it as a suburban area with lots of pedestrians and cars."
→ num_cells: 5 (5 cell sites with GA-optimized azimuths)
→ frequency: 2100 MHz (universal, good urban/suburban balance)
→ azimuth: 'sectored' (GA will optimize exact angles based on UE distribution)
→ placement: 'grid' (GA will refine placement based on SINR optimization)
→ reasoning: "Suburban area ~10km² needs 5 cells at ~2km²/site density. With 200 UEs, 5 cells with optimized azimuths provide excellent coverage and capacity. 2100 MHz is optimal for urban/suburban mixed use. GA will optimize both positions and azimuths for maximum SINR."

Example 2: Rural area, 30 UEs, ~50 km²
Query: "Generate 30 UEs in rural countryside"
→ num_cells: 3 (3 cell sites with GA-optimized azimuths)
→ frequency: 1800 MHz (good rural coverage)
→ azimuth: 'sectored' (GA will optimize directions)
→ placement: 'grid' (GA will refine based on UE locations)
→ reasoning: "Rural area 50km² with low UE density needs 3 cells (~17km²/site). GA will optimize azimuth angles to point toward UE concentrations. 1800 MHz provides good propagation for wide areas."

Example 3: Dense urban, 500 UEs, small area ~5 km²
Query: "Generate 500 UEs in downtown Manhattan during rush hour"
→ num_cells: 30 (10 sites × 3 sectors, very high density for capacity)
→ frequency: 2600 MHz (high capacity for dense urban)
→ azimuth: 'sectored' (maximize capacity)
→ placement: 'cluster' (concentrated in downtown)
→ reasoning: "Very high UE density (500 in 5km²) requires 10 sites for capacity (~0.5km²/site). With sectored deployment, 30 total sectors provide sufficient capacity. 2600 MHz provides high throughput. Clustered placement focuses coverage on downtown hotspot."

Example 4: Highway corridor, 100 UEs, linear ~20 km
Query: "100 cars on highway route"
→ num_cells: 5 (linear deployment, ~4 km spacing)
→ frequency: 2100 MHz (good highway coverage)
→ azimuth: 'sectored' (oriented along highway direction)
→ placement: 'grid' (linear along route)
→ reasoning: "Highway deployment needs linear cell placement along route. 5 cells at ~4km spacing provide continuous coverage. Sectored antennas oriented along highway direction."

Example 5: Small area, very few UEs, rural
Query: "Generate 10 UEs in small village"
→ num_cells: 1 (minimal deployment)
→ frequency: 1800 MHz (good coverage)
→ azimuth: 'omnidirectional' (simple single site)
→ placement: 'grid' (center placement)
→ reasoning: "Very small deployment (10 UEs) only needs single omnidirectional cell for adequate coverage. Cost-effective solution for low-density area."

Example 6: User explicitly requests tower count
Query: "Give me 5 cell towers for Tokyo suburban area with 200 UEs"
→ num_cells: 5 (USER REQUESTED 5 towers = 5 cells with optimized azimuths)
→ frequency: 2100 MHz
→ azimuth: 'sectored' (GA will optimize angles)
→ placement: 'grid'
→ reasoning: "User explicitly requested 5 towers. With GA optimization, each cell will have a unique location and optimized azimuth (0-360°) based on UE distribution. 2100 MHz appropriate for suburban."

Example 7: User requests specific cell count
Query: "Deploy 20 cells in downtown area with 300 UEs"
→ num_cells: 20 (USER REQUESTED 20 cells = 20 unique cell sites)
→ frequency: 2600 MHz (dense urban)
→ azimuth: 'sectored' (GA will optimize angles)
→ placement: 'cluster'
→ reasoning: "User explicitly requested 20 cells. GA will create 20 cells with unique locations and optimized azimuths for maximum SINR. High frequency for dense urban capacity."
"""


def get_topology_prompt(
    area_type: str,
    num_ues: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    raw_query: str,
) -> str:
    """Get topology generation prompt.

    Args:
        area_type: Area type (urban, suburban, rural, highway, mixed)
        num_ues: Total number of UEs in the simulation
        min_lat: Minimum latitude boundary
        max_lat: Maximum latitude boundary
        min_lon: Minimum longitude boundary
        max_lon: Maximum longitude boundary
        raw_query: Original user query for context

    Returns:
        Complete prompt string
    """
    # Calculate approximate area size
    # Rough approximation: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    center_lat = (min_lat + max_lat) / 2

    import math
    lat_km = lat_diff * 111
    lon_km = lon_diff * 111 * math.cos(math.radians(center_lat))
    area_km2 = lat_km * lon_km

    return f"""{TOPOLOGY_SYSTEM_PROMPT}

{TOPOLOGY_GENERATION_EXAMPLES}

---

TASK:
Generate cell topology parameters for:
- Area type: {area_type}
- Number of UEs: {num_ues}
- Geographic boundaries:
  * Latitude: {min_lat:.6f} to {max_lat:.6f} ({lat_diff:.6f}° span)
  * Longitude: {min_lon:.6f} to {max_lon:.6f} ({lon_diff:.6f}° span)
  * Approximate area: {area_km2:.2f} km²
- Raw query: "{raw_query}"

IMPORTANT:
1. **CHECK RAW QUERY FIRST** for explicit tower/cell/site count requests:
   - Look for phrases like: "5 towers", "10 cells", "3 sites", "15 cell towers"
   - If user specifies a number, USE THAT NUMBER as num_cells (don't calculate automatically)
   - If sectored (3 sectors per site): user says "5 towers" = 15 cells (5 sites × 3 sectors)
   - If omnidirectional: user says "5 towers" = 5 cells
2. If NO explicit count in query, calculate num_cells based on area size ({area_km2:.2f} km²) and area type ({area_type})
3. Consider UE count ({num_ues}) for capacity planning
4. Analyze raw query for any specific context about density, coverage needs, or technology requirements
5. Choose frequency band appropriate for area type and any context clues
6. Select azimuth strategy based on num_cells and area type
7. Choose placement strategy based on query context and area characteristics

Output must include:
1. num_cells (int, >= 1)
2. cell_carrier_freq_mhz (int, common values: 700, 1800, 2100, 2600, 3500)
3. azimuth_strategy (str, one of: 'sectored', 'omnidirectional', 'random')
4. cell_placement_strategy (str, one of: 'grid', 'cluster', 'boundary')
5. reasoning (str, explain your parameter choices)

Ensure parameters are realistic for a {area_type} deployment covering {area_km2:.2f} km² with {num_ues} UEs."""
