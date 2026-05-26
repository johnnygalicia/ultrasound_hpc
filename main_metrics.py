import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)



def rms_contrast(image):
    return np.std(image)


def entropy(image):

    hist = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist.ravel() / hist.sum()

    hist = hist[hist > 0]

    return -np.sum(hist * np.log2(hist))


def snr(image):

    signal = np.mean(image)

    noise = np.std(image)

    if noise == 0:
        return 0

    return signal / noise


def mean_intensity(image):
    return np.mean(image)


def show_metrics(name, original, processed):

    print(f"\n===== {name} =====")

    print(
        f"RMS Contrast: "
        f"{rms_contrast(processed):.4f}"
    )

    print(
        f"Entropy: "
        f"{entropy(processed):.4f}"
    )

    print(
        f"SNR: "
        f"{snr(processed):.4f}"
    )

    print(
        f"Mean Intensity: "
        f"{mean_intensity(processed):.4f}"
    )


    psnr_value = peak_signal_noise_ratio(
        original,
        processed,
        data_range=255
    )

    print(
        f"PSNR: "
        f"{psnr_value:.4f} dB"
    )


    ssim_value, _ = structural_similarity(
        original,
        processed,
        full=True
    )

    print(
        f"SSIM: "
        f"{ssim_value:.4f}"
    )



original = cv2.imread(
    "data/processed/video_1_frames/frame_1500.png",
    cv2.IMREAD_GRAYSCALE
)

processed_cpu = cv2.imread(
    "data/processed/video1_bilateral_cpu/frame_1500.png",
    cv2.IMREAD_GRAYSCALE
)

processed_gpu = cv2.imread(
    "data/processed/video1_bilateral_gpu/frame_1500.png",
    cv2.IMREAD_GRAYSCALE
)

processed_mt = cv2.imread(
    "data/processed/video1_bilateral_multi_thread/frame_1500.png",
    cv2.IMREAD_GRAYSCALE
)

if original is None:
    print("Error cargando imagen original")

if processed_cpu is None:
    print("Error cargando imagen CPU")

if processed_gpu is None:
    print("Error cargando imagen GPU")

if processed_mt is None:
    print("Error cargando imagen MultiThread")




print("\n===== ORIGINAL =====")

print(
    f"RMS Contrast: "
    f"{rms_contrast(original):.4f}"
)

print(
    f"Entropy: "
    f"{entropy(original):.4f}"
)

print(
    f"SNR: "
    f"{snr(original):.4f}"
)

print(
    f"Mean Intensity: "
    f"{mean_intensity(original):.4f}"
)



show_metrics(
    "PROCESADA CPU",
    original,
    processed_cpu
)

show_metrics(
    "PROCESADA GPU",
    original,
    processed_gpu
)

show_metrics(
    "PROCESADA MULTITHREAD",
    original,
    processed_mt
)