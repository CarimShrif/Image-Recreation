import functools
import operator

import numpy as np


def initial_population(img_shape: tuple, n_individuals=8):
    """

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


def population_fitness(population, target):
    fit = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fit[i] = fitness(target, population[i])
    return fit


def fitness(target, chrom):
    quality = np.mean(np.abs(target - chrom))
    quality = np.mean(target) - quality
    return quality
