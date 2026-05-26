from src.io.reconstruct_video import (
    reconstruct_video
)


def main():

    reconstruct_video(
        frames_dir="data/processed/video2_bilateral_sharpen_multi_thread",
        output_path="outputs/videos/video2_bilateral_sharpen_multi_thread.mp4",
        fps=14
    )


if __name__ == "__main__":
    main()