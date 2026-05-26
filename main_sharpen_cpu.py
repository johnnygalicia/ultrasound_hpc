import os
import cv2
import time

from tqdm import tqdm

from src.filters.sharpen_cpu import (
    apply_sharpen
)

# =============================================
# CONFIG
# =============================================

INPUT_DIR = "data/processed/video_2_frames"

OUTPUT_DIR = (
    "data/processed/video2_sharpen_cpu"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =============================================
# OBTENER FRAMES
# =============================================

frame_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".png")
])

print("\n==============================")
print("SHARPEN CPU")
print("==============================\n")

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
    desc="Processing CPU Sharpen"
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
    # SHARPEN
    # =========================================

    sharpened = apply_sharpen(
        image
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
        sharpened
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

print("\n==============================")
print("RESULTS")
print("==============================\n")

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