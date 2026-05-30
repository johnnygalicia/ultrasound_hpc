import os
import cv2
import time

from tqdm import tqdm

from src.parallel.bilateral_gpu import (
    BilateralGPU
)

from src.parallel.sharpen_gpu import (
    SharpenGPU
)

# =============================================
# Entrada y salida de los frames 
# =============================================

INPUT_DIR = "data/processed/video_2_frames"

OUTPUT_DIR = (
    "data/processed/video2_bilateral_sharpen_gpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =============================================
# Filtros en GPU 
# =============================================

bilateral_gpu = BilateralGPU()

sharpen_gpu = SharpenGPU()

# =============================================
# OBTENER FRAMES
# =============================================

frame_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".png")
])

print("\n===================================")
print("BILATERAL + SHARPEN GPU PIPELINE")
print("===================================\n")

print(
    f"Frames encontrados: "
    f"{len(frame_files)}"
)

# =============================================
# TIMER START
# =============================================

start_time = time.perf_counter()

# =============================================
# LOOP
# =============================================

for frame_file in tqdm(
    frame_files,
    desc="Processing GPU Pipeline"
):

    frame_path = os.path.join(
        INPUT_DIR,
        frame_file
    )

    image = cv2.imread(
        frame_path,
        cv2.IMREAD_GRAYSCALE
    )

    # =========================================
    # STEP 1: BILATERAL GPU
    # =========================================

    bilateral_result = bilateral_gpu.apply(
        image
    )

    # =========================================
    # STEP 2: SHARPEN GPU
    # =========================================

    sharpen_result = sharpen_gpu.apply(
        bilateral_result
    )

    # =========================================
    # GUARDAR
    # =========================================

    output_path = os.path.join(
        OUTPUT_DIR,
        frame_file
    )

    cv2.imwrite(
        output_path,
        sharpen_result
    )

# =============================================
# TIMER END
# =============================================

end_time = time.perf_counter()

# =============================================
# RESULTS
# =============================================

total_time = end_time - start_time

avg_time = (
    total_time / len(frame_files)
)

fps = (
    len(frame_files) / total_time
)

print("\n===================================")
print("RESULTS")
print("===================================\n")

print(
    f"Tiempo total: "
    f"{total_time:.4f} s"
)

print(
    f"Tiempo promedio/frame: "
    f"{avg_time:.6f} s"
)

print(
    f"Frames por segundo: "
    f"{fps:.2f}"
)

print("\nFrames guardados en:")

print(OUTPUT_DIR)