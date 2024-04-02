import os
import random
import numpy as np
from typing import List
import yaml
from environment import Environment, Point

class GeneticPruebaGeneticSolver:
    def __init__(self, num_genes, population_size, max_generations, environment, individual_mutation=True, random_seed=None):
        self.num_genes = num_genes
        self.population_size = population_size
        self.max_generations = max_generations
        self.environment: Environment = environment
        self.default_distance = self.environment.maklink.path_length
        self.default_path_distances = np.zeros(num_genes)
        for i in range(num_genes):
            self.default_path_distances[i] = self.environment.maklink.maklink_distances[self.environment.maklink.path[i+1]]
        self.mutation_rate = 0.05
        self.individual_mutation = individual_mutation
        self.crossover_rate = 0.8
        self.random_seed = random_seed
        self.population = self.initialize_population()

    def initialize_population(self):
       #np.random.seed(self.random_seed)
       #return np.random.rand(self.population_size, self.num_genes)
        random.seed(self.random_seed)
        return [[random.uniform(0.0, 1.0) for _ in range(self.num_genes)] for _ in range(self.population_size)]

    def enhanced_crossover(self, parent1, parent2):
        #np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        #start_point = np.random.randint(0, self.num_genes - 1)
        start_point = random.randint(0, self.num_genes - 1)
        #end_point = np.random.randint(start_point, self.num_genes - 1)
        end_point = random.randint(start_point, self.num_genes - 1)
        child1 = np.concatenate((parent1[:start_point], parent2[start_point:end_point], parent1[end_point:]))
        child2 = np.concatenate((parent2[:start_point], parent1[start_point:end_point], parent2[end_point:]))
        return child1, child2

    def improved_mutate(self, individual):
        #np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        mutated_individual = np.where(np.random.rand(self.num_genes) < self.mutation_rate, np.random.rand(self.num_genes), individual)
        return mutated_individual
    
    def roulette_wheel_selection(self, fitness_scores: List[float]):
        #np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        total_fitness = sum(fitness_scores)
        population = []
        for _ in range(self.population_size):
            spin = np.random.uniform(0, total_fitness)
            current_sum = 0
            for i, score in enumerate(fitness_scores):
                current_sum += score
                if current_sum >= spin:
                    population.append(self.population[i])
                    break
        self.population = np.array(population)

    def evaluate(self, individual):
        return max((self.default_distance*0.01)**5, (self.default_distance - self.environment.calculate_path_length(individual))**5)

    def evaluate_path_sections(self, individual, printsmth=False):
        mutation_probabilities = []
        for i in range(self.num_genes):
            prev_section = self.environment.calculate_path_length(individual,i,i+1)
            prev_section_default = self.default_path_distances[0] if i==0 else self.default_path_distances[i]-self.default_path_distances[i-1]
            next_section = self.environment.calculate_path_length(individual,i+1,i+2)
            next_section_default = self.default_distance - self.default_path_distances[i] if i==self.num_genes-1 else self.default_path_distances[i+1]-self.default_path_distances[i]
            
            mutation_probability = (prev_section / prev_section_default + next_section / next_section_default) / 2 * self.mutation_rate
                        
            if printsmth and i==3:
                print("Prev section: {:.2f} vs. {:.2f}\nNext section: {:.2f} vs. {:.2f}\nRate: {:.2f}".format(prev_section, prev_section_default, next_section, next_section_default, mutation_probability))

            mutation_probabilities.append(mutation_probability)

        return mutation_probabilities

    def all_the_same(self):
        individual = self.population[0]
        for i in range(1,self.population_size):
            for j in range(self.num_genes):
                if self.population[i][j] != individual[j]:
                    return False
        return True

    def evolve(self):
        for generation in range(self.max_generations):
            # Elitismo
            scores = [self.evaluate(individual) for individual in self.population]
            elitist_score = max(scores)
            elitist = self.population[scores.index(elitist_score)]

            # Roulette-wheel selection
            self.roulette_wheel_selection(scores)

            # Homogenity testing
            if self.all_the_same():
                print("Generación {} homogena.".format(generation))
                break

            # Crossover mejorado
            parent1_index = np.random.randint(0, self.population_size - 1)
            parent2_index = np.random.randint(0, self.population_size - 1)
            self.enhanced_crossover(self.population[parent1_index], self.population[parent2_index])

            # Mutación
            self.population = np.array([self.improved_mutate(child) for child in self.population])

            # Evaluación de la aptitud después de la mutación
            scores = [self.evaluate(individual) for individual in self.population]

            # Elitismo mejorado
            min_score_index = scores.index(min(scores))
            if self.evaluate(self.population[min_score_index]) < elitist_score:
                self.population[min_score_index] = elitist

            if generation % 10 == 0:
                print("Generation: ", generation, " | Max. Fitness: {:.2f}".format(max(scores)))

        print("=======================================================")
        print("Distancia original: {:.2f} | Distancia optimizada: {:.2f}".format(
            self.default_distance, min(length for length in (self.environment.calculate_path_length(individual) for individual in self.population)))
        )
        print("=======================================================")
        return [self.evaluate(individual) for individual in self.population]

def create_environment(x, y, obstacles, start_point, end_point):
    env = Environment(x, y)
    for obstacle in obstacles:
        env.add_obstacle(obstacle)
    # Calculating Maklink graph
    env.calculate_maklink()
    env.add_start_point(start_point)
    env.add_end_point(end_point)
    env.maklink.dijkstra()
    return env
    
def run_genetic_algorithm(env: Environment, population_size, max_generations, random_seed, visualize = True, individual_solution = True):
    solver = GeneticPruebaGeneticSolver(len(env.maklink.path)-2, population_size, max_generations, env, individual_solution, random_seed)
    scores = solver.evolve()

    solution = solver.population[scores.index(max(scores))]

    if visualize:
        env.visualize_environment(solution, True, algorithm="GA_3")
    return env.calculate_path_length(solution)

def load_and_create_environments(yaml_file: str, random_seed):
    with open(os.path.abspath(yaml_file), 'r') as file:
        environments = yaml.safe_load(file)

    for environment_data in environments:
        env_params = environment_data['environment']
        x = env_params['x']
        y = env_params['y']
        start_point = Point(env_params['start_point']['x'], env_params['start_point']['y'])
        end_point = Point(env_params['end_point']['x'], env_params['end_point']['y'])
        obstacles = [
            [Point(vertex['x'], vertex['y']) for vertex in obstacle]
            for obstacle in env_params['obstacles']
        ]

        env = create_environment(x, y, obstacles, start_point, end_point)
        run_genetic_algorithm(env, 200, 100, random_seed, visualize=True)

if __name__ == "__main__":
    random_seed = random.random()
    print("Semilla aleatoria:", random_seed)
    load_and_create_environments("./environment3.yaml",  random_seed)
