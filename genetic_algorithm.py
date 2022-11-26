import itertools
import os
import random
import numpy as np
from utilities import *


def initial_population(img_shape: tuple, n_individuals: int = 8):
    """
    :type img_shape: tuple of ints
    :type n_individuals: int
    :param img_shape: shape of the target image
    :param n_individuals: population size
    :return: numpy array of length n_individuals
    """

    length = functools.reduce(operator.mul, img_shape)
    init_population = np.empty(shape=(n_individuals,
                                      length),
                               dtype=np.uint8)
    for i in range(n_individuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[i] = np.random.random(length) * 256

    return init_population


def population_fitness(population: np.ndarray, target_chromosome: np.ndarray):
    """
    :param population: array of shape (N,L) where N:number of individuals and L is the chromosome length
    :param target_chromosome: the target chromosome to compare the population against
    :return: a numpy array containing the fitness of each individual
    """
    fit = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fit[i] = fitness(target_chromosome, population[i])
    return fit


def fitness(target_chromosome: np.ndarray, chrom: np.ndarray):
    """
    :param target_chromosome: the target chromosome to compare the population against
    :param chrom: the individual
    :return: (a number between 0 and 255) the fitness of the passed individual
    """
    quality = np.mean(np.abs(target_chromosome - chrom))
    quality = np.mean(target_chromosome) - quality
    return quality


def selection(population, adaptation, n_parent):
    selected_parents = np.empty(shape=(n_parent, population.shape[1]), dtype=np.uint8)
    for i in range(n_parent):
        index = np.where(adaptation == np.max(adaptation))[0][0]
        selected_parents[i] = population[index]
        adaptation[index] = -1
    return selected_parents


def crossover(parents: np.ndarray, n_individuals=8):
    """
    :param parents:
    :param n_individuals:
    :return: population after crossover
    """
    population = np.empty(shape=(n_individuals, parents.shape[1]))
    population[:parents.shape[0]] = parents
    offspring_number = n_individuals - parents.shape[0]
    parents_permutations = list(itertools.permutations(iterable=np.arange(0, parents.shape[0]), r=2))
    selected_permutations = random.sample(range(len(parents_permutations)), offspring_number)
    for i in range(offspring_number):
        parent1 = parents[parents_permutations[selected_permutations[i]][0]]
        parent2 = parents[parents_permutations[selected_permutations[i]][1]]
        point = np.int32(population.shape[1] / 2)
        population[parents.shape[0] + i, 0:point] = parent1[0:point]
        population[parents.shape[0] + i, point:] = parent2[point:population.shape[1]]
    return population


def mutation(population: np.ndarray, n_parents: int, mutation_probability: float, n_genes: int) -> None:

    """
    :param population:current population
    :param n_parents:number of previously selected parents
    :param mutation_probability:probability of mutation
    :param n_genes:number of mutated genes for every successful mutation
    :return:
    """
    for i in range(n_parents, population.shape[0]):
        r = random.random()
        offspring = population[i]
        if r < mutation_probability:
            for _ in range(n_genes):
                position = random.randint(0, population.shape[1] - 1)
                offspring[position] = random.randint(0, 256)


def ga(target_img, population_size, selection_size, target_fitness):
    target_chromosome = chromosome(target_img)
    population = initial_population(target_img.shape, population_size)
    fit = population_fitness(population, target_chromosome)
    n_iterations = 0
    while np.max(fit) < target_fitness:
        parents = selection(population, fit, selection_size)
        population = crossover(parents, population_size)
        mutation(population, selection_size, 0.3, 3)
        fit = population_fitness(population, target_chromosome)
        n_iterations += 1
    print(n_iterations)
    best = selection(population, population_fitness(population, target_chromosome), 1)
    return image(best[0], target_img.shape)


if __name__ == '__main__':
    target = load_image(os.path.dirname(os.path.abspath(__file__))+'\\steve.jpg')
    generated_image = ga(target, 25, 5, 95)
    show(target, generated_image)
