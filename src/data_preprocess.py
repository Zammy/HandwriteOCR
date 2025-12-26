import os
from pathlib import Path
import cv2

from src.io import (
    delete_all_files,
    load_images_from_folder,
    load_text_file,
    save_line_images,
)
from src.line_extraction import segment_handwritten_lines
from src.preprocess_deterministic import (
    binarize_document,
    convert_to_grayscale,
    preprocess_pipeline,
    resize_with_aspect_ratio,
)


def preprocess_examples():
    print("=" * 50)
    print("Handwriting Image Preprocessing Examples")
    print("=" * 50)

    data_folder = "./data"
    temp_folder = "./_temp"
    interm_folder = "./_inter_data"

    delete_all_files(temp_folder, include_subfolders=True)
    delete_all_files(interm_folder, include_subfolders=True)

    if os.path.exists(data_folder):
        # TODO: convert to iterator so that not all images are loaded
        images = load_images_from_folder(data_folder)
        print(f"\nLoaded {len(images)} images\n")

        for filename, img in images:
            print(f"Processing: {filename}")

            name = Path(filename).stem
            label_data = load_text_file(data_folder, name)

            preprocessed = preprocess_pipeline(
                pipeline=[
                    (
                        resize_with_aspect_ratio,
                        {"height": 1024 * 2, "interpolation": cv2.INTER_AREA},
                    ),
                    (convert_to_grayscale, {}),
                    # (normalize_contrast, {"clip_limit": 2.0, "tile_size": 8}),
                    (
                        binarize_document,
                        {
                            "method": "sauvola",
                            "block_size": 25,
                            "k": 0.2,
                        },
                    ),
                ],
                source=img,
                name=name,
                debug_save=False,
            )

            extracted_lines = segment_handwritten_lines(
                preprocessed,
                expected_lines=len(label_data),
                cc_params={
                    "connectivity": 8,
                    "min_area": 75,
                    "max_area": 1500,
                    "max_aspect_ratio": 25,
                },
                clustering_params={
                    "eps_vertical": 3,
                    "eps_horizontal": 250,
                    "min_samples": 3,
                },
                crop_line_params={
                    "target_height": 128,
                    "target_width": 1024,
                    "interpolation": cv2.INTER_LINEAR,
                },
            )

            save_to = Path(interm_folder, name).resolve()
            save_line_images(extracted_lines, output_dir=str(save_to))


if __name__ == "__main__":
    preprocess_examples()
