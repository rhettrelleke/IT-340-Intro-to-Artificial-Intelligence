# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:37:08 2017

@author: xfang13

"""
import numpy as np
import random
import matplotlib

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
    # Randomly select an individual from the population based on fitness
    fitness_scores = [fitness_fn(individual) for individual in population]
    total_fitness = sum(fitness_scores)
    selection_probabilities = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=selection_probabilities)[0]

def mutate(child, mutation_prob):
    # Mutate the child with a small probability
    if random.random() < mutation_prob:
        # Choose a random location and change its value
        index_to_mutate = random.randint(0, len(child) - 1)
        new_value = random.randint(1, len(child))
        child[index_to_mutate] = new_value
    return child

def reproduce(x, y):
    # Perform reproduction (crossover) between two parents, x and y
    n = len(x)
    c = random.randint(1, n)
    child = x[:c] + y[c:]
    return child

def create_initial_population(population_size):
    # Create the initial population with random individuals
    population = [random.sample(range(1, 9), 8) for _ in range(population_size)]
    return population

def genetic_algorithm(population_size, fitness_fn, mutation_prob=0.1, max_generations=1000, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    population = create_initial_population(population_size)
    generation = 0

    while True:
        new_population = []

        for _ in range(population_size):
            x = random_selection(population, fitness_fn)
            y = random_selection(population, fitness_fn)
            child = reproduce(x, y)
            child = mutate(child, mutation_prob)
            new_population.append(child)

        population = new_population
        generation += 1
        
        # Check for termination conditions (fitness threshold or max generations)
        best_individual = max(population, key=fitness_fn)
        if fitness_fn(best_individual) == 28 or generation >= max_generations:
            return best_individual

if __name__ == '__main__':
    random_seed = 42  # Set a random seed for replicability
    population_size = 100
    best_solution = genetic_algorithm(population_size=population_size, fitness_fn=fitness, random_seed=random_seed)
    print("Best solution found:", best_solution)
    print("Fitness:", fitness(best_solution))