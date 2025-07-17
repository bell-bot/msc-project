import copy

import torch
from circuits.dense.mlp import StepMLP
import numpy as np
from tqdm import tqdm


class GAOptimiser:

    def __init__(
        self,
        seed_circuit: StepMLP,
        trigger_input: torch.Tensor,
        expected_output: torch.Tensor,
        population_size: int = 1000,
        mutation_rate: float = 0.05,
        mutation_amount: float = 0.1,
    ):
        self.seed_circuit = seed_circuit
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount

        self.trigger_input = trigger_input
        self.expected_output = expected_output

        self.population = self._create_initial_population()
        self.best_circuit = None
        self.best_fitness = -float("inf")

    def _create_initial_population(self) -> list[StepMLP]:
        return [self._mutate(self.seed_circuit) for _ in range(self.population_size)]

    def _mutate(self, individual: StepMLP) -> StepMLP:
        mutated_circuit = copy.deepcopy(individual)
        with torch.no_grad():
            for param in mutated_circuit.parameters():
                mutation_mask = torch.rand_like(param) < self.mutation_rate
                mutation = torch.rand_like(param) * self.mutation_amount

                param.data += mutation_mask * mutation

        return mutated_circuit

    def _crossover(self, parent1: StepMLP, parent2: StepMLP) -> StepMLP:
        child_circuit = copy.deepcopy(parent1)

        with torch.no_grad():
            for p1_param, p2_param, child_param in zip(parent1.parameters(), parent2.parameters(), child_circuit.parameters()):

                p1_flat = p1_param.data.flatten()
                p2_flat = p2_param.data.flatten()

                crossover_point = np.random.randint(0, len(p1_flat))

                child_flat = torch.cat([p1_flat[:crossover_point], p2_flat[crossover_point:]])

                child_param.data = child_flat.reshape(p1_param.data.shape)

        return child_circuit

    def fitness_function(self, individual: StepMLP) -> float:

        fitness = 0.0

        with torch.no_grad():
            output = individual(self.trigger_input)

            # Objective 1: Correctness
            if not torch.allclose(output, self.expected_output):
                return 0.0  # If number 1 objective isn't met, then we don't want the solution at all
            fitness += 10.0

            # Objective 2: Robustness to noise
            noisy_circuit = copy.deepcopy(individual)
            for param in noisy_circuit.parameters():
                param.data += torch.randn_like(param) * 0.1

            noisy_output = noisy_circuit(self.trigger_input)
            if torch.allclose(noisy_output, self.expected_output):
                fitness += 5.0

            # Objective 3: Obscurity
            l2_penalty = 0.0
            for param in individual.parameters():
                # TODO: replace normal distribution after analysing weight distribution
                # in existing models
                l2_penalty += torch.norm(param.data).item()

            fitness -= 0.1 * l2_penalty  # Apparently need small factor to balance the penalty

        return fitness

    def run(self, generations: int = 100):

        for generation in tqdm(range(generations), desc="Generations"):

            fitness_scores = [self.fitness_function(circuit) for circuit in self.population]

            max_fitness_in_gen = max(fitness_scores)
            if max_fitness_in_gen > self.best_fitness:
                self.best_fitness = max_fitness_in_gen
                best_index = np.argmax(fitness_scores)
                self.best_circuit = copy.deepcopy(self.population[best_index])

            new_population = []
            fitness_tensor = torch.tensor(fitness_scores)

            selection_probs = fitness_tensor / (fitness_tensor.sum() + 1e-6)

            if selection_probs.sum() == 0:  # Random selection if fitness is 0 for entire population
                selection_probs = torch.ones(self.population_size) / self.population_size

            parent_indices = torch.multinomial(selection_probs, self.population_size * 2, replacement=True)
            parents = [self.population[i] for i in parent_indices]

            for i in range(self.population_size):
                parent1 = parents[i]
                parent2 = parents[i + self.population_size]
                child = self._crossover(parent1, parent2)
                mutated_child = self._mutate(child)
                new_population.append(mutated_child)

            self.population = new_population
