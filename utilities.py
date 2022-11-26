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


def show(img: numpy.ndarray):
    plt.imshow(img)
    plt.show()


def load_image(filename):
    return iio.imread(filename)


def save(img):
    iio.imwrite('image.jpeg', img,extension=".jpeg")
