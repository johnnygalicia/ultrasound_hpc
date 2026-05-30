from pathlib import Path

from src.io.reconstruct_video import (
    reconstruct_video
)


def main():

    # ======================================
    # CARPETA DONDE ESTAN TODOS LOS FRAMES
    # ======================================

    processed_dir = Path("data/processed")

    # ======================================
    # CARPETA DE SALIDA DE VIDEOS
    # ======================================

    output_dir = Path("outputs/videos")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================
    # RECORRER TODAS LAS CARPETAS
    # ======================================

    for frames_folder in processed_dir.iterdir():

        # Solo carpetas
        if not frames_folder.is_dir():
            continue

        # ==================================
        # NOMBRE DEL VIDEO
        # ==================================

        video_name = f"{frames_folder.name}.mp4"

        output_path = output_dir / video_name

        print("\n===================================")
        print(f"Reconstruyendo: {frames_folder.name}")
        print(f"Salida: {output_path}")
        print("===================================")

        # ==================================
        # RECONSTRUIR VIDEO
        # ==================================

        reconstruct_video(
            frames_dir=str(frames_folder),
            output_path=str(output_path),
            fps=14
        )

    print("\nTodos los videos fueron reconstruidos.")


if __name__ == "__main__":
    main()