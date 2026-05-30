import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

print("Metricas bilateral")

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
    "data/processed/video_2_frames/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_cpu = cv2.imread(
    "data/processed/video2_bilateral_cpu/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_gpu = cv2.imread(
    "data/processed/video2_bilateral_gpu/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_mt = cv2.imread(
    "data/processed/video2_bilateral_multi_thread/frame_1764.png",
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

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# ==========================================
# MÉTRICAS
# ==========================================

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


# ==========================================
# CARGAR IMÁGENES
# ==========================================

original = cv2.imread(
    "data/processed/video_2_frames/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_cpu = cv2.imread(
    "data/processed/video2_bilateral_cpu/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_gpu = cv2.imread(
    "data/processed/video2_bilateral_gpu/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

processed_mt = cv2.imread(
    "data/processed/video2_bilateral_multi_thread/frame_1764.png",
    cv2.IMREAD_GRAYSCALE
)

# ==========================================
# LISTA DE IMÁGENES
# ==========================================

images = [
    ("Original", original),
    ("CPU", processed_cpu),
    ("GPU", processed_gpu),
    ("MultiThread", processed_mt)
]

# ==========================================
# CONFIGURACIÓN DE ZOOM
# ==========================================

# Región de interés automática
# Ajusta estos valores para enfocarte en corazón/válvula

x1 = 600
x2 = 1000

y1 = 125
y2 = 425

# ==========================================
# FIGURA ESTILO PAPER IEEE
# ==========================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

axs = axs.ravel()

# ==========================================
# TABLA DE METRICAS
# ==========================================

metrics_table = []

for ax, (title, img) in zip(axs, images):
    # ======================================
    # ZOOM AUTOMÁTICO
    # ======================================

    zoom_img = img[y1:y2, x1:x2]

    # ======================================
    # MÉTRICAS
    # ======================================

    rms = rms_contrast(zoom_img)

    ent = entropy(zoom_img)

    snr_value = snr(zoom_img)

    psnr_value = peak_signal_noise_ratio(
        original[y1:y2, x1:x2],
        zoom_img,
        data_range=255
    )

    ssim_value, _ = structural_similarity(
        original[y1:y2, x1:x2],
        zoom_img,
        full=True
    )
    # ======================================
    # GUARDAR METRICAS EN TABLA
    # ======================================

    metrics_table.append([
        title,
        rms,
        ent,
        snr_value,
        psnr_value,
        ssim_value
    ])
    # ======================================
    # MOSTRAR IMAGEN
    # ======================================

    ax.imshow(
        zoom_img,
        cmap='gray',
        interpolation='nearest'
    )

    # ======================================
    # TÍTULO
    # ======================================

    ax.set_title(
        title,
        fontsize=13,
        fontweight='bold'
    )

    # ======================================
    # BORDES LIMPIOS
    # ======================================

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # ======================================
    # ESCALA DE PIXELES
    # ======================================

    ax.set_xlabel("Pixels X", fontsize=9)

    ax.set_ylabel("Pixels Y", fontsize=9)

    ax.tick_params(
        axis='both',
        labelsize=8
    )

    # ======================================
    # MÉTRICAS DEBAJO
    # ======================================

    # metrics_text = (
    #     f"RMS: {rms:.2f}\n"
    #     f"Entropy: {ent:.2f}\n"
    #     f"SNR: {snr_value:.2f}\n"
    #     f"PSNR: {psnr_value:.2f} dB\n"
    #     f"SSIM: {ssim_value:.4f}"
    # )

    # ax.text(
    #     0.5,
    #     -0.22,
    #     metrics_text,
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     ha='center',
    #     va='top',
    #     bbox=dict(
    #         facecolor='white',
    #         edgecolor='gray',
    #         boxstyle='round,pad=0.4'
    #     )
    # )

# ==========================================
# AJUSTE GENERAL
# ==========================================

plt.tight_layout()

# ==========================================
# GUARDAR FIGURA
# ==========================================

plt.savefig(
    "outputs/images/comparison_figure_bilateral.png",
    dpi=300,
    bbox_inches='tight'
)

# ==========================================
# MOSTRAR
# ==========================================

plt.show()
# ==========================================
# IMPRIMIR TABLA EN TERMINAL
# ==========================================

print("\n")
print("=" * 90)
print("ZOOM METRICS bilateral")
print("=" * 90)

print(
    f"{'Method':<15}"
    f"{'RMS':<15}"
    f"{'Entropy':<15}"
    f"{'SNR':<15}"
    f"{'PSNR':<15}"
    f"{'SSIM':<15}"
)

print("-" * 90)

for row in metrics_table:

    print(
        f"{row[0]:<15}"
        f"{row[1]:<15.4f}"
        f"{row[2]:<15.4f}"
        f"{row[3]:<15.4f}"
        f"{row[4]:<15.4f}"
        f"{row[5]:<15.4f}"
    )

print("=" * 90)