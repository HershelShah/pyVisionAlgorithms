"""Image Processing Unit Tests."""
import unittest

import cv2 as cv

import colorbalance

TestImage = './Dataset/taeyeon.jpg'
BlendPrimary = './Dataset/blend_0.jpg'
BlendSecondary = './Dataset/blend_1.jpg'


class RGBColorBalanceTester(unittest.TestCase):
    """RGB Color Balance Tester Class."""

    def test_interactive(self):
        """Interactive color balance test."""
        test_image = cv.imread(TestImage)
        result_image = colorbalance.rgb_color_balance(
            test_image, (2, 0), (0, 0), (0, 0))
        cv.imshow('Interactive Test', result_image)
        cv.waitKey(0)


class GammaCorrectionTester(unittest.TestCase):
    """Gamma Correction Tester Class."""

    def test_interactive(self):
        """Interactive gamma correction test."""
        test_image = cv.imread(TestImage)
        result_image = colorbalance.gamma_correction(test_image, 2.2)
        cv.imshow('Interactive Test', result_image)
        cv.waitKey(0)


class DyadicOperatorTester(unittest.TestCase):
    """Dyadic Blender Operator Tester Class."""

    def test_interactive(self):
        """Interactive blending test."""
        primary = cv.imread(BlendPrimary)
        secondary = cv.imread(BlendSecondary)
        result_image = colorbalance.linear_dyadic_image_blending(
            primary, secondary, .2)
        cv.imshow('Interactive Test', result_image)
        cv.waitKey(0)


if __name__ == '__main__':
    unittest.main()
