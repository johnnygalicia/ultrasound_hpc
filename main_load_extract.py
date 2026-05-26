from pathlib import Path

from src.io.video_loader import load_video, get_video_info
from src.io.frame_extractor import extract_frames
from src.filters.speckle_reduction import (bilateral_filter)
from src.visualization.frame_display import (show_comparison)
from src.preprocessing.clahe import apply_clahe


# Lista de videos
VIDEO_PATHS = [
    "data/raw/videos/video_1.mp4",
    "data/raw/videos/video_2.mp4"
]


def process_video(video_path):

    print(f"\nProcesando: {video_path}")

    # Cargar video
    cap = load_video(video_path)

    # Información del video
    info = get_video_info(cap)

    print("\n=== VIDEO INFO ===")

    for key, value in info.items():
        print(f"{key}: {value}")

    # Nombre del video sin extensión
    video_name = Path(video_path).stem

    # Carpeta específica para frames
    output_frames_dir = (
        f"data/processed/{video_name}_frames"
    )

    # Extraer frames
    frames = extract_frames(
        cap,
        output_dir=output_frames_dir
    )

    print(f"\nFrames extraídos: {len(frames)}")

    # Frame de prueba
    test_frame = frames[500]

    # Filtro bilateral
    bilateral_result = bilateral_filter(
        test_frame,
        diameter=9,
        sigma_color=75,
        sigma_space=75
    )

    # CLAHE
    clahe_result = apply_clahe(
        bilateral_result
    )

    # Mostrar comparación
    show_comparison(
        clahe_result,
        bilateral_result,
        title_processed=f"{video_name}"
    )


def main():

    for video_path in VIDEO_PATHS:
        process_video(video_path)


if __name__ == "__main__":
    main()