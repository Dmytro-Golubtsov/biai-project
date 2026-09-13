"""
Manual Genetic Algorithm for feature selection.

Bio-inspired component:
- individual: binary vector representing selected original features
- gene: 1 means feature selected, 0 means feature not selected
- population: candidate feature subsets
- fitness: validation R² score with a small penalty for too many features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from preprocessing import ALL_FEATURES, build_preprocessor


@dataclass
class GAResult:
    best_individual: np.ndarray
    best_features: List[str]
    best_fitness: float
    history: List[Dict[str, object]]


class GeneticFeatureSelector:
    """Genetic Algorithm feature selector for original dataset feature groups."""

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        random_state: int = 42,
        model_n_estimators: int = 100,
        max_train_rows: int = 100000,
        feature_penalty: float = 0.002,
        tournament_size: int = 3,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.model_n_estimators = model_n_estimators
        self.max_train_rows = max_train_rows
        self.feature_penalty = feature_penalty
        self.tournament_size = tournament_size

        self.rng = np.random.default_rng(random_state)
        self.n_features = len(ALL_FEATURES)

    def _initialize_population(self) -> np.ndarray:
        population = self.rng.integers(0, 2, size=(self.population_size, self.n_features))

        # Avoid individuals with zero selected features.
        for i in range(self.population_size):
            if population[i].sum() == 0:
                population[i, self.rng.integers(0, self.n_features)] = 1

        return population

    @staticmethod
    def individual_to_features(individual: np.ndarray) -> List[str]:
        return [feature for gene, feature in zip(individual, ALL_FEATURES) if gene == 1]

    def _fitness(
        self,
        individual: np.ndarray,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        selected_features = self.individual_to_features(individual)

        if len(selected_features) == 0:
            return -1_000_000.0

        preprocessor = build_preprocessor(selected_features)
        model = RandomForestRegressor(
            n_estimators=self.model_n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=18,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train[selected_features], y_train)
        predictions = pipeline.predict(X_val[selected_features])
        r2 = r2_score(y_val, predictions)

        # Small simplicity pressure: prefer smaller subsets when scores are similar.
        penalty = self.feature_penalty * (len(selected_features) / self.n_features)
        return float(r2 - penalty)

    def _evaluate_population(
        self,
        population: np.ndarray,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> np.ndarray:
        fitness_scores = np.zeros(self.population_size)

        for i, individual in enumerate(population):
            fitness_scores[i] = self._fitness(individual, X_train, y_train, X_val, y_val)

        return fitness_scores

    def _tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        contestant_indices = self.rng.choice(
            np.arange(self.population_size),
            size=self.tournament_size,
            replace=False,
        )
        winner_index = contestant_indices[np.argmax(fitness_scores[contestant_indices])]
        return population[winner_index].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = self.rng.integers(1, self.n_features)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()

        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        if mutated.sum() == 0:
            mutated[self.rng.integers(0, self.n_features)] = 1

        return mutated

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GAResult:
        """
        Run GA and return best selected feature subset.

        For speed, this method optionally samples rows before GA evaluation.
        The final model is still trained separately on the full training set.
        """
        if len(X) > self.max_train_rows:
            sampled_indices = self.rng.choice(len(X), size=self.max_train_rows, replace=False)
            X_work = X.iloc[sampled_indices].reset_index(drop=True)
            y_work = y.iloc[sampled_indices].reset_index(drop=True)
        else:
            X_work = X.reset_index(drop=True)
            y_work = y.reset_index(drop=True)

        X_train, X_val, y_train, y_val = train_test_split(
            X_work,
            y_work,
            test_size=0.25,
            random_state=self.random_state,
        )

        population = self._initialize_population()
        best_individual = None
        best_fitness = -np.inf
        history: List[Dict[str, object]] = []

        for generation in range(1, self.generations + 1):
            fitness_scores = self._evaluate_population(population, X_train, y_train, X_val, y_val)
            generation_mean_fitness = float(np.mean(fitness_scores))

            generation_best_idx = int(np.argmax(fitness_scores))
            generation_best = population[generation_best_idx].copy()
            generation_best_fitness = float(fitness_scores[generation_best_idx])

            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_individual = generation_best.copy()

            selected_features = self.individual_to_features(generation_best)

            print(
                f"Generation {generation:02d} | "
                f"Best fitness: {generation_best_fitness:.4f} | "
                f"Mean fitness: {generation_mean_fitness:.4f} | "
                f"Selected features: {selected_features}"
            )

            history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_best_fitness,
                    "mean_fitness": generation_mean_fitness,
                    "selected_features_count": len(selected_features),
                    "selected_features": selected_features,
                }
            )

            # Elitism: keep best individual from current generation.
            new_population = [generation_best.copy()]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)

        assert best_individual is not None
        best_features = self.individual_to_features(best_individual)

        return GAResult(
            best_individual=best_individual,
            best_features=best_features,
            best_fitness=float(best_fitness),
            history=history,
        )
