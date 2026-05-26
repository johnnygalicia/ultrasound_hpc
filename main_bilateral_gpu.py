import os
import time
import cv2

from tqdm import tqdm

from src.parallel.bilateral_gpu import (
    BilateralGPU
)

# =============================================
# CONFIG
# =============================================

FRAMES_DIR = "data/processed/video_2_frames"

# =============================================
# GPU FILTER
# =============================================

gpu_filter = BilateralGPU()

# =============================================
# OBTENER FRAMES
# =============================================

frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.endswith(".png")
])

print("\n==============================")
print("GPU BILATERAL BENCHMARK")
print("==============================\n")

print(f"Frames encontrados: {len(frame_files)}")

# =============================================
# TIMER START
# =============================================

start_time = time.perf_counter()

# =============================================
# LOOP
# =============================================

for frame_file in tqdm(
    frame_files,
    desc="Processing GPU"
):

    frame_path = os.path.join(
        FRAMES_DIR,
        frame_file
    )

    image = cv2.imread(
        frame_path,
        cv2.IMREAD_GRAYSCALE
    )

    result_gpu = gpu_filter.apply(image)
    
    output_path = os.path.join(
        "data/processed/video2_bilateral_gpu",
        frame_file
    )

    cv2.imwrite(
        output_path,
        result_gpu
    )
# =============================================
# TIMER END
# =============================================

end_time = time.perf_counter()

cv2.imwrite(output_path, result_gpu)

# =============================================
# RESULTS
# =============================================

total_time = end_time - start_time

avg_time = total_time / len(frame_files)

fps = len(frame_files) / total_time

print("\n==============================")
print("RESULTS")
print("==============================\n")

print(f"Tiempo total: {total_time:.4f} s")

print(
    f"Tiempo promedio/frame: "
    f"{avg_time:.6f} s"
)

print(
    f"Frames por segundo: "
    f"{fps:.2f}"
)