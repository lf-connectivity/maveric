"""Genetic Algorithm for optimal cell tower placement.

Based on research:
- "Accurate Base Station Placement in 4G LTE Networks Using Multiobjective Genetic
  Algorithm Optimization" (Isabona et al., 2023)
- "Algorithmic Approach for Strategic Cell Tower Placement" (IEEE)
- Optimizes for coverage, capacity, and cost minimization
"""
import math
import random
from typing import List, Tuple

import numpy as np


class CellTowerChromosome:
    """Represents a potential solution (chromosome) for cell tower placement."""

    def __init__(self, num_sites: int, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """Initialize a random chromosome.

        Args:
            num_sites: Number of cell sites to place
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary
        """
        self.num_sites = num_sites
        self.bounds = (min_lat, max_lat, min_lon, max_lon)

        # Each gene is a (lat, lon, azimuth) tuple representing a cell site
        # azimuth is the antenna direction in degrees (0-360)
        self.genes: List[Tuple[float, float, int]] = [
            (
                random.uniform(min_lat, max_lat),
                random.uniform(min_lon, max_lon),
                random.randint(0, 360),  # Random azimuth
            )
            for _ in range(num_sites)
        ]

        self.fitness: float = 0.0

    def calculate_fitness(
        self,
        ue_locations: np.ndarray = None,
        weights: dict = None,
        frequency_mhz: int = 2100
    ) -> float:
        """Calculate fitness score for this chromosome.

        Multi-objective fitness function:
        1. SINR: Maximize average SINR across UE locations (if ue_locations provided)
        2. Coverage: Maximize area covered
        3. Capacity: Minimize overcrowding (balance load)
        4. Cost: Minimize inter-site distance variance (uniform distribution)

        Args:
            ue_locations: Array of UE positions [(lat, lon), ...] for SINR/coverage calculation
            weights: Dictionary with keys 'sinr', 'coverage', 'capacity', 'cost'
            frequency_mhz: Carrier frequency in MHz for path loss calculation

        Returns:
            Fitness score (higher is better)
        """
        if weights is None:
            # Default weights when SINR is not used
            weights = {'coverage': 0.5, 'capacity': 0.3, 'cost': 0.2}

        min_lat, max_lat, min_lon, max_lon = self.bounds

        # Objective 1: SINR optimization (if UE locations provided)
        sinr_score = 0.0
        if ue_locations is not None and len(ue_locations) > 0 and 'sinr' in weights:
            sinr_score = self._calculate_sinr_score(ue_locations, frequency_mhz)

        # Objective 2: Coverage uniformity (minimize maximum distance to nearest site)
        # Use UE locations if provided, otherwise sample grid points
        if ue_locations is not None and len(ue_locations) > 0:
            coverage_points = ue_locations
        else:
            coverage_points = self._sample_area_points(num_samples=100)

        max_distance_to_site = 0.0
        for point in coverage_points:
            # Extract only lat/lon from genes (ignoring azimuth)
            min_dist = min(
                self._haversine_distance(point, (site[0], site[1]))
                for site in self.genes
            )
            max_distance_to_site = max(max_distance_to_site, min_dist)

        # Coverage score: inverse of max distance (lower max distance = better coverage)
        coverage_score = 1.0 / (1.0 + max_distance_to_site)

        # Objective 3: Load balancing (standard deviation of distances between sites)
        # Uniform distribution minimizes congestion
        site_distances = []
        for i, site1 in enumerate(self.genes):
            for site2 in self.genes[i + 1:]:
                # Compare only lat/lon, ignore azimuth
                site_distances.append(
                    self._haversine_distance((site1[0], site1[1]), (site2[0], site2[1]))
                )

        if site_distances:
            dist_std = np.std(site_distances)
            dist_mean = np.mean(site_distances)
            # Lower variance relative to mean = better uniformity
            capacity_score = 1.0 / (1.0 + dist_std / max(dist_mean, 0.001))
        else:
            capacity_score = 1.0

        # Objective 4: Cost minimization (avoid boundary clustering)
        # Penalize sites too close to boundaries or clustered together
        boundary_penalty = 0.0
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        for lat, lon, azimuth in self.genes:
            # Penalize if within 10% of boundary
            if (lat - min_lat) / lat_range < 0.1 or (max_lat - lat) / lat_range < 0.1:
                boundary_penalty += 0.1
            if (lon - min_lon) / lon_range < 0.1 or (max_lon - lon) / lon_range < 0.1:
                boundary_penalty += 0.1

        cost_score = max(0.0, 1.0 - boundary_penalty / self.num_sites)

        # Combine objectives with weights
        self.fitness = (
            weights.get('sinr', 0.0) * sinr_score +
            weights.get('coverage', 0.0) * coverage_score +
            weights.get('capacity', 0.0) * capacity_score +
            weights.get('cost', 0.0) * cost_score
        )

        return self.fitness

    def _sample_area_points(self, num_samples: int) -> List[Tuple[float, float]]:
        """Sample random points across the coverage area."""
        min_lat, max_lat, min_lon, max_lon = self.bounds
        return [
            (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
            for _ in range(num_samples)
        ]

    def _haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two lat/lon points in km using Haversine formula."""
        lat1, lon1 = point1
        lat2, lon2 = point2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in km
        r = 6371.0

        return r * c

    def _calculate_sinr_score(
        self,
        ue_locations: np.ndarray,
        frequency_mhz: int
    ) -> float:
        """
        Calculate average SINR across all UE locations using path loss model.
        Optimized with vectorized operations for better performance.

        Uses Log-Distance Path Loss model:
        - Rx_power (dBm) = Tx_power - Path_Loss + Antenna_Gain
        - Path_Loss (dB) = PL₀ + 10·n·log₁₀(d/d₀)
        - SINR (dB) = Signal / (Interference + Noise)

        Args:
            ue_locations: Array of UE positions [(lat, lon), ...]
            frequency_mhz: Carrier frequency in MHz

        Returns:
            Normalized SINR score in [0, 1] where higher is better
        """
        # Constants for path loss model
        TX_POWER_DBM = 46.0        # Typical macro cell Tx power (40W)
        NOISE_DBM = -104.0         # Thermal noise for 10 MHz bandwidth
        PATH_LOSS_EXPONENT = 3.5   # Urban/suburban propagation
        REFERENCE_DISTANCE_M = 1.0
        REFERENCE_PATH_LOSS_DB = 40.0

        num_ues = len(ue_locations)
        num_cells = len(self.genes)

        # Pre-allocate matrix for Rx powers [num_ues x num_cells]
        rx_powers_matrix = np.zeros((num_ues, num_cells))

        # Vectorized calculation for all UE-cell pairs
        for cell_idx, (cell_lat, cell_lon, azimuth) in enumerate(self.genes):
            for ue_idx, (ue_lat, ue_lon) in enumerate(ue_locations):
                # 1. Calculate distance (Haversine)
                distance_km = self._haversine_distance(
                    (ue_lat, ue_lon), (cell_lat, cell_lon)
                )
                distance_m = max(distance_km * 1000, 1.0)  # Prevent log(0)

                # 2. Calculate path loss
                path_loss_db = REFERENCE_PATH_LOSS_DB + \
                    10 * PATH_LOSS_EXPONENT * np.log10(
                        distance_m / REFERENCE_DISTANCE_M
                    )

                # 3. Calculate antenna gain based on azimuth
                antenna_gain_db = self._calculate_antenna_gain(
                    cell_lat, cell_lon, azimuth, ue_lat, ue_lon
                )

                # 4. Calculate Rx power
                rx_powers_matrix[ue_idx, cell_idx] = TX_POWER_DBM - path_loss_db + antenna_gain_db

        # Vectorized SINR calculation
        # Convert to linear scale
        rx_powers_linear = 10 ** (rx_powers_matrix / 10)

        # Find serving cell (max power) for each UE
        max_power_linear = np.max(rx_powers_linear, axis=1)

        # Calculate interference (sum of all powers minus serving)
        total_power_linear = np.sum(rx_powers_linear, axis=1)
        interference_linear = total_power_linear - max_power_linear

        # Add noise
        noise_linear = 10 ** (NOISE_DBM / 10)

        # Calculate SINR for all UEs
        sinr_linear = max_power_linear / (interference_linear + noise_linear + 1e-12)
        sinr_db = 10 * np.log10(sinr_linear)

        # Normalize SINR: typical range [-5, 25] dB
        # Good SINR > 10 dB, Excellent > 20 dB
        avg_sinr_db = np.mean(sinr_db)
        normalized_sinr = np.clip((avg_sinr_db + 5) / 30, 0, 1)

        return normalized_sinr

    def _calculate_antenna_gain(
        self,
        cell_lat: float,
        cell_lon: float,
        azimuth: int,
        ue_lat: float,
        ue_lon: float
    ) -> float:
        """
        Calculate antenna gain based on azimuth pattern.

        Uses 3-sector antenna pattern:
        - Max gain: 15 dBi (front)
        - 3dB beamwidth: 65°
        - Front-to-back ratio: 20 dB

        Args:
            cell_lat: Cell latitude
            cell_lon: Cell longitude
            azimuth: Antenna azimuth in degrees (0-360)
            ue_lat: UE latitude
            ue_lon: UE longitude

        Returns:
            Antenna gain in dB
        """
        # Calculate bearing from cell to UE
        bearing = self._calculate_bearing(
            cell_lat, cell_lon, ue_lat, ue_lon
        )

        # Angle difference from main beam
        angle_diff = abs(((bearing - azimuth + 180) % 360) - 180)

        # Antenna pattern
        if angle_diff <= 32.5:  # Main lobe (±32.5° = 65° beamwidth)
            antenna_gain_db = 15.0  # Max gain
        elif angle_diff <= 90:  # Side lobe
            # Linear attenuation in side lobe
            antenna_gain_db = 15.0 - (angle_diff - 32.5) * 0.35
        else:  # Back lobe
            antenna_gain_db = -5.0  # 20 dB front-to-back ratio

        return antenna_gain_db

    def _calculate_bearing(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate bearing from point 1 to point 2 in degrees (0-360).

        Args:
            lat1: Starting latitude
            lon1: Starting longitude
            lat2: Ending latitude
            lon2: Ending longitude

        Returns:
            Bearing in degrees (0-360), where 0° is North
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = (np.cos(lat1) * np.sin(lat2) -
             np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for optimizing cell tower placement."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
    ):
        """Initialize GA optimizer.

        Args:
            population_size: Number of chromosomes in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover (0-1)
            mutation_rate: Probability of mutation (0-1)
            elitism_rate: Fraction of best individuals to preserve (0-1)
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = max(1, int(population_size * elitism_rate))

    def optimize(
        self,
        num_sites: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        ue_locations: np.ndarray = None,
        frequency_mhz: int = 2100,
        fitness_weights: dict = None,
        verbose: bool = True,
    ) -> List[Tuple[float, float, int]]:
        """Run genetic algorithm to find optimal cell tower locations.

        Args:
            num_sites: Number of cell sites to place
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary
            ue_locations: Array of UE positions [(lat, lon), ...] for SINR optimization
            frequency_mhz: Carrier frequency in MHz for path loss calculation
            fitness_weights: Weights for multi-objective fitness
            verbose: Print progress

        Returns:
            List of optimal (lat, lon, azimuth) tuples
        """
        if verbose:
            print(f"\n[Genetic Algorithm Optimization]")
            print(f"  Population: {self.population_size}")
            print(f"  Generations: {self.generations}")
            print(f"  Sites to optimize: {num_sites}")
            if ue_locations is not None:
                print(f"  UE locations for SINR: {len(ue_locations)}")
                print(f"  Frequency: {frequency_mhz} MHz")

        # Initialize population
        population = [
            CellTowerChromosome(num_sites, min_lat, max_lat, min_lon, max_lon)
            for _ in range(self.population_size)
        ]

        # Evaluate initial population
        for chromosome in population:
            chromosome.calculate_fitness(
                ue_locations=ue_locations,
                weights=fitness_weights,
                frequency_mhz=frequency_mhz
            )

        best_fitness_history = []

        # Evolution loop
        for gen in range(self.generations):
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)

            best_fitness = population[0].fitness
            best_fitness_history.append(best_fitness)

            if verbose and gen % 10 == 0:
                print(f"  Generation {gen}/{self.generations}: Best fitness = {best_fitness:.4f}")

            # Elitism: preserve best individuals
            new_population = population[:self.elitism_count]

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection: tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                if random.random() < self.mutation_rate:
                    self._mutate(child1)
                if random.random() < self.mutation_rate:
                    self._mutate(child2)

                # Evaluate fitness
                child1.calculate_fitness(
                    ue_locations=ue_locations,
                    weights=fitness_weights,
                    frequency_mhz=frequency_mhz
                )
                child2.calculate_fitness(
                    ue_locations=ue_locations,
                    weights=fitness_weights,
                    frequency_mhz=frequency_mhz
                )

                new_population.extend([child1, child2])

            # Trim to population size
            population = new_population[:self.population_size]

        # Return best solution
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_chromosome = population[0]

        if verbose:
            print(f"  Final best fitness: {best_chromosome.fitness:.4f}")
            print(f"  Improvement: {((best_fitness_history[-1] / best_fitness_history[0]) - 1) * 100:.1f}%")

        return best_chromosome.genes

    def _tournament_selection(self, population: List[CellTowerChromosome], tournament_size: int = 3) -> CellTowerChromosome:
        """Select parent using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(
        self, parent1: CellTowerChromosome, parent2: CellTowerChromosome
    ) -> Tuple[CellTowerChromosome, CellTowerChromosome]:
        """Perform crossover between two parents (uniform crossover)."""
        child1 = CellTowerChromosome(parent1.num_sites, *parent1.bounds)
        child2 = CellTowerChromosome(parent2.num_sites, *parent2.bounds)

        # Uniform crossover: randomly select genes from parents
        for i in range(parent1.num_sites):
            if random.random() < 0.5:
                child1.genes[i] = parent1.genes[i]
                child2.genes[i] = parent2.genes[i]
            else:
                child1.genes[i] = parent2.genes[i]
                child2.genes[i] = parent1.genes[i]

        return child1, child2

    def _mutate(self, chromosome: CellTowerChromosome):
        """Mutate chromosome by randomly adjusting site locations and azimuths."""
        min_lat, max_lat, min_lon, max_lon = chromosome.bounds

        # Random mutation: adjust one or more sites
        num_mutations = random.randint(1, max(1, chromosome.num_sites // 4))

        for _ in range(num_mutations):
            site_idx = random.randint(0, chromosome.num_sites - 1)

            # Small perturbation (Gaussian mutation)
            lat, lon, azimuth = chromosome.genes[site_idx]
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon

            new_lat = np.clip(lat + random.gauss(0, lat_range * 0.05), min_lat, max_lat)
            new_lon = np.clip(lon + random.gauss(0, lon_range * 0.05), min_lon, max_lon)

            # Mutate azimuth with Gaussian noise (±30° std dev)
            azimuth_change = random.gauss(0, 30)
            new_azimuth = int((azimuth + azimuth_change) % 360)

            chromosome.genes[site_idx] = (new_lat, new_lon, new_azimuth)
