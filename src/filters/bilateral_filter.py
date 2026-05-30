import cv2


def bilateral_filter(
    frame,
    diameter=5,
    sigma_color=25,
    sigma_space=10
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