import os
import cv2
import numpy as np

from pathlib import Path
from typing import List, Tuple


def load_images_from_folder(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    images = []
    folder = Path(folder_path)

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    for file_path in folder.iterdir():
        if file_path.is_file():
            if (
                "page" in file_path.name
                and file_path.suffix.lower() in supported_extensions
            ):
                try:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        images.append((file_path.name, img))
                        print(f"Loaded: {file_path.name}")
                    else:
                        print(f"Warning: Could not read {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")

    return images


def save_image_temp(
    image: np.ndarray,
    name: str,
    output_dir: str = "./temp",
) -> None:
    name = name + ".png"
    path = str(Path(output_dir, name).resolve())
    cv2.imwrite(path, image)


def load_text_file(folder_path: str, filename: str) -> List[str]:
    file_path = Path(folder_path, f"{filename}.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def delete_all_files(folder_path: str, include_subfolders: bool = False) -> None:
    if not os.path.isdir(folder_path):
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

        if not include_subfolders:
            # Stop recursion after the top-level folder
            break


def save_line_images(
    line_images: list,
    output_dir="./temp",
    ext: str = ".png",
):
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for i, img in enumerate(line_images):
        filename = f"line_{i}{ext}"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, img)
        saved_files.append(filepath)
