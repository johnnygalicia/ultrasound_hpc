import matplotlib.pyplot as plt
import cv2


def prepare_image(image):

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_comparison(
    original,
    processed,
    title_processed="Processed"
):

    original_display = prepare_image(original)
    processed_display = prepare_image(processed)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_display, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_display, cmap="gray")
    plt.title(title_processed)
    plt.axis("off")

    plt.tight_layout()

    plt.show()