import cv2
import numpy as np


# ============================================
# CONTRASTE RMS
# ============================================

def rms_contrast(image):
    """
    Calcula contraste RMS.

    Parameters
    ----------
    image : ndarray
        Imagen en escala de grises.

    Returns
    -------
    float
    """

    image = image.astype(np.float32)

    mean_intensity = np.mean(image)

    contrast = np.sqrt(
        np.mean((image - mean_intensity) ** 2)
    )

    return contrast


# ============================================
# ENTROPIA
# ============================================

def entropy(image):
    """
    Calcula entropía de Shannon.
    """

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.ravel()

    probability = histogram / np.sum(histogram)

    probability = probability[
        probability > 0
    ]

    ent = -np.sum(
        probability * np.log2(probability)
    )

    return ent


# ============================================
# SNR APROXIMADO
# ============================================

def snr(image):
    """
    Estimación simple de SNR.
    """

    image = image.astype(np.float32)

    signal = np.mean(image)

    noise = np.std(image)

    if noise == 0:
        return 0

    return signal / noise


# ============================================
# BRILLO PROMEDIO
# ============================================

def mean_intensity(image):

    return np.mean(image)