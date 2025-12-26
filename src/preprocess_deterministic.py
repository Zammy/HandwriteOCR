import cv2
import numpy as np
from typing import Callable, List, Tuple

from src.io import save_image_temp


def resize_with_aspect_ratio(image, height=1024, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]

    scale = height / h
    dim = (int(w * scale), height)

    return cv2.resize(image, dim, interpolation=interpolation)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def normalize_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    enhanced = clahe.apply(image)

    return enhanced


def binarize_document(
    image,
    method: str = "sauvola",
    block_size: int = 25,
    k: float = 0.2,
):
    gray = image

    if method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C=10
        )
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C=10,
        )
    elif method == "sauvola":
        # simple Sauvola implementation
        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(block_size, block_size))
        sqmean = cv2.boxFilter(gray_f**2, ddepth=-1, ksize=(block_size, block_size))
        var = sqmean - mean**2
        std = np.sqrt(np.clip(var, 0, None))

        R = 128.0  # dynamic range
        thresh = mean * (1 + k * (std / R - 1))
        binary = (gray_f > thresh).astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary)

    return binary


def preprocess_pipeline(
    pipeline: List[Tuple[Callable[[np.ndarray], np.ndarray], dict]],
    source: np.ndarray,
    name: str,
    debug_save: bool,
) -> np.ndarray:
    for i, (func, params) in enumerate(pipeline):
        source = func(source, **params)
        if debug_save:
            save_image_temp(source, f"{name}_{i:02}_{func.__name__}")
    return source
