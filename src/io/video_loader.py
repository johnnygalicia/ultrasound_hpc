import cv2


def load_video(video_path):
    cap = cv2.VideoCapture(video_path) #Video cargado

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    return cap

# Informacion basica del video cargado
# largo, ancho, fotogramas por segundo, numero de fotogramas
# y duracion 
def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }

    return info