"""Background Segmentation Operation Class."""

from enum import Enum
from typing import Callable

import numpy as np
import cv2 as cv


class BackgroundSegmentation:
    """Background segmentation with different implementation methods."""

    @classmethod
    def __init__(self, source) -> None:
        """Background segmentation"""
        self.source = cv.VideoCapture(source)

    @classmethod
    def run(self, method: Callable) -> None:
        """Run background subtrator with specified algorithm.

        Parameters:
        -----------
            method: Enum
                Value specifying the algorithm for background segmentation.
        """
        pass


# Algorithms used for background subtraction
def adjacent_frame(self, source):
    """Determine background from adjacent frames.

    Parameters:
    -----------
        source: str
            Specifies the source of the video.

    """
    
