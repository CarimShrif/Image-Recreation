import unittest
from utilities import *

class MyTestCase(unittest.TestCase):
    def test_conversion(self):
        im=load_image('imageio:chelsea.png')
        chrom=chromosome(im)
        self.assertTrue(
            numpy.array_equal(im,
                              image(chrom,
                                    im.shape))
        )


if __name__ == '__main__':
    unittest.main()
