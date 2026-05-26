import cv2
import os


def reconstruct_video(
    frames_dir,
    output_path,
    fps=10
):
    """
    Reconstruye un video a partir de frames.

    Parameters
    ----------
    frames_dir : str
        Carpeta con frames procesados.

    output_path : str
        Ruta del video de salida.

    fps : int
        Frames por segundo.
    """

    # ---------------------------------------------
    # Obtener lista de frames
    # ---------------------------------------------

    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])

    if len(frame_files) == 0:
        raise ValueError(
            "No se encontraron frames."
        )

    # ---------------------------------------------
    # Leer primer frame
    # ---------------------------------------------

    first_frame_path = os.path.join(
        frames_dir,
        frame_files[0]
    )

    first_frame = cv2.imread(
        first_frame_path,
        cv2.IMREAD_GRAYSCALE
    )

    height, width = first_frame.shape

    # ---------------------------------------------
    # Crear VideoWriter
    # ---------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    # ---------------------------------------------
    # Escribir frames
    # ---------------------------------------------

    for frame_file in frame_files:

        frame_path = os.path.join(
            frames_dir,
            frame_file
        )

        frame = cv2.imread(
            frame_path,
            cv2.IMREAD_GRAYSCALE
        )

        writer.write(frame)

    writer.release()

    print(
        f"\nVideo reconstruido:\n{output_path}"
    )