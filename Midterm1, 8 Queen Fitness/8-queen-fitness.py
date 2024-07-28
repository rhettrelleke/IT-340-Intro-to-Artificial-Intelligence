# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:37:08 2023

@author: Rhett

"""
MUTATION_RATE = 0.4
MAX_GENERATIONS = 50
import numpy as np
import random
import matplotlib.pyplot as plt
plt.interactive(True)

random_seed = random.seed(27)

def fitness(Input):
    #Input should be a list
    # assert(type(Input)==list)

    #Step1: make the state out of the Input
    state = np.zeros((8,8))
    for j in range(8):
        state[Input[j]-1][j] = 1
            

    #Step2: find the fitness of the state
    attacks = 0
    k = -1
    for j in range(8):
        k += 1
        #direction 1: the east
        for l in range(k+1,8):
            attacks += state[state[:,j].argmax()][l]
    
        #direction 2: the northeast
        row = state[:,j].argmax()
        column = j
        while row > 0 and column < 7:
            row -= 1
            column += 1
            attacks += state[row][column]
            
        #direction 3: the southeast
        row = state[:,j].argmax()
        column = j
        while row < 7 and column < 7:
            row += 1
            column += 1
            attacks += state[row][column]
            
    return 28 - attacks

def random_selection(population, fitness_fn):
    fitness_stat = [fitness_fn(individual) for individual in population]
    total_fitness = sum(fitness_stat)
    selection_probability = [stat / total_fitness for stat in fitness_stat]
    return random.choices(population, weights=selection_probability)[0]

# Mutation function
def mutate(child, MUTATION_RATE):
    # Mutate the child with a small probability
    if random.random() < MUTATION_RATE:
        # Choose a random location and change its value
        index_to_mutate = random.randint(0, len(child) - 1)
        new_value = random.randint(1, len(child))
        child[index_to_mutate] = new_value
    return child

# Reproduction function
def reproduce(x, y):
    n = len(x)
    c = random.randint(1, n)
    child = x[:c] + y[c:]
    return child

def create_initial_population(population_size):
    # Create the initial population with random individuals
    population = [random.sample(range(1, 9), 8) for _ in range(population_size)]
    return population



# Genetic Algorithm
def genetic_algorithm(population_size, fitness_fn, MUTATION_RATE, MAX_GENERATIONS, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
        
    best_array = []
    average_array = []
    worst_array = []
        
    population = create_initial_population(population_size)

    while True:
        for generation in range(MAX_GENERATIONS):
            fitness_stat = [fitness_fn(individual) for individual in population]
            best_array.append(max(fitness_stat))
            worst_array.append(min(fitness_stat))
            average_array.append(np.mean(fitness_stat))
            new_population = []

            for _ in range(population_size):
                x = random_selection(population, fitness_fn)
                y = random_selection(population, fitness_fn)
                child = reproduce(x, y)
                if np.random.rand() < MUTATION_RATE:
                    child = mutate(child, MUTATION_RATE)
                new_population.append(child)

            population = new_population
            #generation += 1
        
            
        plt.plot(range(len(best_array)), best_array, label='Best Fitness', color='blue')
        plt.plot(range(len(worst_array)), worst_array, label='Worst Fitness', color='yellow')
        plt.plot(range(len(average_array)), average_array, label='Average Fitness', color='black')
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(f"Population Size - {population_size}")
        plt.legend()
        plt.show()
        
        best_individual = max(population, key=fitness_fn)
        if fitness_fn(best_individual) == 28 or generation >= MAX_GENERATIONS - 1:
            return best_individual
        
            
if __name__=='__main__':
    
    
    solution50 = genetic_algorithm(50, fitness, MUTATION_RATE, MAX_GENERATIONS, random_seed)
    print("Solution found for 50 population:", solution50)
    print("Fitness of the solution:", fitness(solution50))
    
    solution100 = genetic_algorithm(100, fitness, MUTATION_RATE, MAX_GENERATIONS, random_seed)
    print("Solution found for 100 population:", solution100)
    print("Fitness of the solution:", fitness(solution100))
    
    solution200 = genetic_algorithm(200, fitness, MUTATION_RATE, MAX_GENERATIONS, random_seed)
    print("Solution found for 200 population:", solution200)
    print("Fitness of the solution:", fitness(solution200))
    
    solution500 = genetic_algorithm(500, fitness, MUTATION_RATE, MAX_GENERATIONS, random_seed)
    print("Solution found for 500 population:", solution500)
    print("Fitness of the solution:", fitness(solution500))      
