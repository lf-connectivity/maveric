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

        # Each gene is a (lat, lon) tuple representing a cell site location
        self.genes: List[Tuple[float, float]] = [
            (
                random.uniform(min_lat, max_lat),
                random.uniform(min_lon, max_lon),
            )
            for _ in range(num_sites)
        ]

        self.fitness: float = 0.0

    def calculate_fitness(self, ue_locations: np.ndarray = None, weights: dict = None) -> float:
        """Calculate fitness score for this chromosome.

        Multi-objective fitness function:
        1. Coverage: Maximize area covered
        2. Capacity: Minimize overcrowding (balance load)
        3. Cost: Minimize inter-site distance variance (uniform distribution)

        Args:
            ue_locations: Array of UE positions [(lat, lon), ...] for coverage calculation
            weights: Dictionary with keys 'coverage', 'capacity', 'cost'

        Returns:
            Fitness score (higher is better)
        """
        if weights is None:
            weights = {'coverage': 0.5, 'capacity': 0.3, 'cost': 0.2}

        min_lat, max_lat, min_lon, max_lon = self.bounds

        # Objective 1: Coverage uniformity (minimize maximum distance to nearest site)
        # Sample points across the area
        grid_points = self._sample_area_points(num_samples=100)
        max_distance_to_site = 0.0

        for point in grid_points:
            min_dist = min(self._haversine_distance(point, site) for site in self.genes)
            max_distance_to_site = max(max_distance_to_site, min_dist)

        # Coverage score: inverse of max distance (lower max distance = better coverage)
        coverage_score = 1.0 / (1.0 + max_distance_to_site)

        # Objective 2: Load balancing (standard deviation of distances between sites)
        # Uniform distribution minimizes congestion
        site_distances = []
        for i, site1 in enumerate(self.genes):
            for site2 in self.genes[i + 1:]:
                site_distances.append(self._haversine_distance(site1, site2))

        if site_distances:
            dist_std = np.std(site_distances)
            dist_mean = np.mean(site_distances)
            # Lower variance relative to mean = better uniformity
            capacity_score = 1.0 / (1.0 + dist_std / max(dist_mean, 0.001))
        else:
            capacity_score = 1.0

        # Objective 3: Cost minimization (avoid boundary clustering)
        # Penalize sites too close to boundaries or clustered together
        boundary_penalty = 0.0
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        for lat, lon in self.genes:
            # Penalize if within 10% of boundary
            if (lat - min_lat) / lat_range < 0.1 or (max_lat - lat) / lat_range < 0.1:
                boundary_penalty += 0.1
            if (lon - min_lon) / lon_range < 0.1 or (max_lon - lon) / lon_range < 0.1:
                boundary_penalty += 0.1

        cost_score = max(0.0, 1.0 - boundary_penalty / self.num_sites)

        # Combine objectives with weights
        self.fitness = (
            weights['coverage'] * coverage_score +
            weights['capacity'] * capacity_score +
            weights['cost'] * cost_score
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
        fitness_weights: dict = None,
        verbose: bool = True,
    ) -> List[Tuple[float, float]]:
        """Run genetic algorithm to find optimal cell tower locations.

        Args:
            num_sites: Number of cell sites to place
            min_lat: Minimum latitude boundary
            max_lat: Maximum latitude boundary
            min_lon: Minimum longitude boundary
            max_lon: Maximum longitude boundary
            fitness_weights: Weights for multi-objective fitness
            verbose: Print progress

        Returns:
            List of optimal (lat, lon) positions
        """
        if verbose:
            print(f"\n[Genetic Algorithm Optimization]")
            print(f"  Population: {self.population_size}")
            print(f"  Generations: {self.generations}")
            print(f"  Sites to optimize: {num_sites}")

        # Initialize population
        population = [
            CellTowerChromosome(num_sites, min_lat, max_lat, min_lon, max_lon)
            for _ in range(self.population_size)
        ]

        # Evaluate initial population
        for chromosome in population:
            chromosome.calculate_fitness(weights=fitness_weights)

        best_fitness_history = []

        # Evolution loop
        for gen in range(self.generations):
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)

            best_fitness = population[0].fitness
            best_fitness_history.append(best_fitness)

            if verbose and gen % 20 == 0:
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
                child1.calculate_fitness(weights=fitness_weights)
                child2.calculate_fitness(weights=fitness_weights)

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
        """Mutate chromosome by randomly adjusting site locations."""
        min_lat, max_lat, min_lon, max_lon = chromosome.bounds

        # Random mutation: adjust one or more sites
        num_mutations = random.randint(1, max(1, chromosome.num_sites // 4))

        for _ in range(num_mutations):
            site_idx = random.randint(0, chromosome.num_sites - 1)

            # Small perturbation (Gaussian mutation)
            lat, lon = chromosome.genes[site_idx]
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon

            new_lat = np.clip(lat + random.gauss(0, lat_range * 0.05), min_lat, max_lat)
            new_lon = np.clip(lon + random.gauss(0, lon_range * 0.05), min_lon, max_lon)

            chromosome.genes[site_idx] = (new_lat, new_lon)
