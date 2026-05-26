import cv2


def apply_clahe(frame):

    # ==========================================
    # SI YA ES GRAYSCALE
    # ==========================================

    if len(frame.shape) == 2:

        gray = frame

    # ==========================================
    # SI ES BGR
    # ==========================================

    else:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

    # ==========================================
    # CLAHE
    # ==========================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced