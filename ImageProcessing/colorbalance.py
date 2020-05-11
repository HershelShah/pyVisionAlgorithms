"""Simple application to change the color balance of an Image."""

import numpy as np
import cv2 as cv


def rgb_color_balance(input_image: np.ndarray,
                      red_alpha_beta: tuple,
                      green_alpha_beta: tuple,
                      blue_alpha_beta: tuple) -> np.ndarray:
    """Changes color balance of RGB image.

    Ouput = Alpha * Input + Beta

    Parameters:
    ----------
        input_image: np.ndarray
            Input RGB image as specified by numpy array
        red_alpha_beta: tuple
            Red gain(index 0) and red bias (index 1)
        green_alpha_beta: tuple
            Green gain(index 0) and green bias (index 1)
        blue_alpha_beta: tuple
            Blue gain(index 0) and blue bias (index 1)

    Returns:
    --------
        output_image: np.ndarray
            Returns color balance adjusted image in form of np.ndarray.

    """
    blue_lut = np.clip(np.array([i * blue_alpha_beta[0] + blue_alpha_beta[1]
                                 for i in np.arange(0, 256)]),
                       0, 255).astype(np.uint8)
    green_lut = np.clip(np.array([i * green_alpha_beta[0] + green_alpha_beta[1]
                                  for i in np.arange(0, 256)]),
                        0, 255).astype(np.uint8)
    red_lut = np.clip(np.array([i * red_alpha_beta[0] + red_alpha_beta[1]
                                for i in np.arange(0, 256)]),
                      0, 255).astype(np.uint8)

    input_image[:, :, 0] = cv.LUT(input_image[:, :, 0], blue_lut)
    input_image[:, :, 1] = cv.LUT(input_image[:, :, 1], green_lut)
    input_image[:, :, 2] = cv.LUT(input_image[:, :, 2], red_lut)
    return input_image


def gamma_correction(input_image: np.ndarray,
                     gamma: float) -> np.ndarray:
    """Gamma correction of an RGB image.

    Output =  (Input) ^ ( 1 / Gamma)

    Parameters:
    ----------
        input_image: np.ndarray
            Input RGB image as specified by numpy array.

    Returns:
    --------
        output_image: np.ndarray
            Returns gamma corrected image in form of np.ndarray.

    """
    gamma_lut = np.array([(i / 255) ** (1 / gamma) * 255
                          for i in np.arange(0, 256)], np.uint8)
    return cv.LUT(input_image, gamma_lut)


def linear_dyadic_image_blending(primary_image: np.ndarray,
                                 secondary_image: np.ndarray,
                                 blending_factor: float) -> np.ndarray:
    """Blending using linear dyadic operator.

    Output = (1 - Blending_Factor) * Secondary_Image +
             Blending_Factor * Primary_Image

    Parameters:
    -----------
        primary_image: np.ndarray
            Main input image.
        secondary_image: np.ndarray
            Secondary input image.
        blending_factor: float
            Value between 0 and 1 specifying the influence of each image.
    """
    blending_factor = np.clip(blending_factor, 0, 1)
    if primary_image.shape != secondary_image.shape:
        secondary_image = cv.resize(secondary_image, (primary_image.shape[1],
                                                      primary_image.shape[0]))
    output = np.clip((1 - blending_factor) * secondary_image / 255 +
                     blending_factor * primary_image / 255, 0, 1) * 255
    return output.astype(np.uint8)
