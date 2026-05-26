import cv2


def gaussian_blur_cpu(
    image,
    kernel_size=(3, 3),
    sigma=1
):

    blurred = cv2.GaussianBlur(
        image,
        kernel_size,
        sigma
    )

    return blurred