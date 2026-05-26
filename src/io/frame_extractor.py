import cv2
import os

# Extraccion de frames del video 
def extract_frames(cap, output_dir=None):

    frames = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        if output_dir is not None:
            filename = os.path.join(
                output_dir,
                f"frame_{frame_idx:04d}.png"
            )

            cv2.imwrite(filename, frame)

        frame_idx += 1

    return frames