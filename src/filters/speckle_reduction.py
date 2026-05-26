import cv2


def bilateral_filter(
    frame,
    diameter=9,
    sigma_color=75,
    sigma_space=75
):
    """
    Aplica filtro bilateral.

    Parameters
    ----------
    frame : ndarray

    Returns
    -------
    filtered : ndarray
    """

    filtered = cv2.bilateralFilter(
        frame,
        diameter,
        sigma_color,
        sigma_space
    )

    return filtered