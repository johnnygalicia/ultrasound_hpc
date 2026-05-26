import cv2
import numpy as np


def apply_sharpen(
    image
):
    """
    Aplica filtro sharpen usando convolución.
    """

    # =========================================
    # KERNEL SHARPEN
    # =========================================

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    # =========================================
    # CONVOLUCIÓN
    # =========================================

    sharpened = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel
    )

    return sharpened