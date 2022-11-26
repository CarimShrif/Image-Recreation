import functools
import itertools
import operator
import random

import numpy as np


def initial_population(img_shape: tuple, n_individuals: int = 8):
    """
    :type img_shape: tuple of ints
    :type n_individuals: int
    :param img_shape: shape of the target image
    :param n_individuals: population size
    :return: numpy array of length n_individuals
    """

    """x=1
    write this then replace it
    for i in img_shape:
        x*=i
    print(x)
    init_population = numpy.empty(shape=(n_individuals,
                                         x),
                                  dtype=numpy.uint8)
    print(init_population)
    """
    length = functools.reduce(operator.mul, img_shape)
    init_population = np.empty(shape=(n_individuals,
                                      length),
                               dtype=np.uint8)
    for i in range(n_individuals):
        # Randomly generating initial population chromosomes genes values.
        init_population[i] = np.random.random(length) * 256

    return init_population


def population_fitness(population: np.ndarray, target: np.ndarray):
    """
    :param population: array of shape (N,L) where N:number of individuals and L is the chromosome length
    :param target: the target chromosome to compare the population against
    :return: a numpy array containing the fitness of each individual
    """
    fit = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fit[i] = fitness(target, population[i])
    return fit


def fitness(target: np.ndarray, chrom: np.ndarray):
    """
    :param target: the target chromosome to compare the population against
    :param chrom: the individual
    :return: (a number between 0 and 255) the fitness of the passed individual
    """
    quality = np.mean(np.abs(target - chrom))
    quality = np.mean(target) - quality
    return quality


def selection(population, adaptation, n_parent):
    selected_parents = np.empty(shape=(n_parent, population.shape[1]))
    for i in range(n_parent):
        index = np.where(adaptation == np.max(adaptation))[0][0]
        selected_parents[i] = population[index]
        adaptation[index] = -1
    return selected_parents


def crossover(parents: np.ndarray, n_individuals=8):
    """
    
    :param parents:
    :param n_individuals:
    :return:
    """
    population = np.empty(shape=(n_individuals, parents.shape[1]))
    population[:parents.shape[0]] = parents
    offspring_number = n_individuals - parents.shape[0]
    parents_permutations = list(itertools.permutations(iterable=np.arange(0, parents.shape[0]), r=2))
    selected_permutations = random.sample(range(len(parents_permutations)), offspring_number)
    for i in range(offspring_number):
        parent1 = parents[parents_permutations[selected_permutations[i]][0]]
        #                                    selected_permutations[0]
        #               parents_permutations[             4          ]
        #                            (1,2)                            [0]
        #       parents[                   1                             ]
        parent2 = parents[parents_permutations[selected_permutations[i]][1]]
        point = np.int32(population.shape[1] / 2)
        population[parents.shape[0] + i, 0:point] = parent1[0:point]
        population[parents.shape[0] + i, point:] = parent2[point:population.shape[1]]
    return population


def mutation(population, n_parents, mutation_probability):
    for i in range(n_parents, population.shape[0]):
        r = random.random()
        offspring = population[i]
        if r < mutation_probability:
            position = random.randint(0, population.shape[1] - 1)
            offspring[position] = random.randint(0, 256)

