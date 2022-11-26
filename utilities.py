import functools
import operator
import imageio.v3 as iio
from matplotlib import pyplot as plt
import numpy


# 13

def chromosome(img):
    return numpy.reshape(a=img, newshape=(functools.reduce(operator.mul, img.shape)))


def image(chrom, shape):
    return numpy.reshape(a=chrom, newshape=shape)


def show(target: numpy.ndarray, generated: numpy.ndarray):
    plt.subplot(1, 2, 1)
    plt.imshow(target)
    plt.subplot(1, 2, 2)
    plt.imshow(generated)
    plt.show()


def load_image(filename):
    return iio.imread(filename)


def save(img):
    iio.imwrite('image.jpeg', img, extension=".jpeg")
