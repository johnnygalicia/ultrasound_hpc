import cv2
import os
from tqdm import tqdm

from src.filters.speckle_reduction import bilateral_filter
from src.preprocessing.clahe import apply_clahe


def process_frames(
    frames,
    output_dir
):
    """
    Procesa todos los frames usando:
    bilateral + CLAHE

    Parameters
    ----------
    frames : list
        Lista de frames.

    output_dir : str
        Carpeta de salida.
    """

    os.makedirs(output_dir, exist_ok=True)

    processed_frames = []

    for idx, frame in enumerate(
        tqdm(frames, desc="Processing Frames")
    ):

        # -----------------------------------------
        # Bilateral Filter
        # -----------------------------------------

        bilateral = bilateral_filter(
            frame,
            diameter=9,
            sigma_color=75,
            sigma_space=75
        )

        # -----------------------------------------
        # CLAHE
        # -----------------------------------------

        enhanced = apply_clahe(
            bilateral
        )

        # -----------------------------------------
        # Guardar frame
        # -----------------------------------------

        filename = os.path.join(
            output_dir,
            f"enhanced_{idx:04d}.png"
        )

        cv2.imwrite(filename, enhanced)

        processed_frames.append(enhanced)

    return processed_frames