from dataclasses import dataclass
import cv2
import numpy as np
from scipy import ndimage


def elastic_distortion(image: np.ndarray, alpha: float = 30.0, sigma: float = 3.0) -> np.ndarray:
    """
    Apply elastic distortion to an image (uint8).
    
    Args:
        image: Input image (uint8, H x W)
        alpha: Strength of distortion
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Distorted image (uint8, same shape)
    """
    h, w = image.shape
    
    dx = np.random.randn(h, w) * sigma
    dy = np.random.randn(h, w) * sigma
    
    dx = ndimage.gaussian_filter(dx, sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter(dy, sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = y + dy.astype(np.float32), x + dx.astype(np.float32)
    
    distorted = ndimage.map_coordinates(image.astype(np.float32), indices, order=1, mode="constant", cval=0)
    
    return distorted.astype(np.uint8)


def random_rotation(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Apply random rotation to an image (uint8).
    
    Args:
        image: Input image (uint8, H x W)
        max_angle: Maximum rotation angle in degrees (-max_angle to +max_angle)
        
    Returns:
        Rotated image (uint8, same shape)
    """
    angle = np.random.uniform(-max_angle, max_angle)
    
    h, w = image.shape
    center = (w / 2, h / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated

def random_shear(image: np.ndarray, max_shear: float = 0.2) -> np.ndarray:
    """
    Apply random horizontal shear to an image (uint8).
    
    Args:
        image: Input image (uint8, H x W)
        max_shear: Maximum shear intensity (0.1 to 0.3 is usually plenty for handwriting)
        
    Returns:
        Sheared image (uint8, same shape)
    """
    h, w = image.shape
    shear_factor = np.random.uniform(-max_shear, max_shear)
    
    # Define the Affine Transformation Matrix
    # [1, shear_factor, 0]  <- Controls horizontal shift based on Y
    # [0, 1,            0]  <- Keeps Y coordinates the same
    M = np.array([
        [1, shear_factor, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Since shearing pushes the image out of bounds, we adjust the 
    # 'center' of the shift so the image doesn't slide off the canvas.
    if shear_factor > 0:
        M[0, 2] = -shear_factor * h * 0.5  # Corrective shift left
    else:
        M[0, 2] = -shear_factor * h * 0.5  # Corrective shift right

    sheared = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )
    
    return sheared

def random_dilate(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Makes white ink thicker on black background."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Explicitly tell OpenCV to treat the 'outside' as black
    return cv2.dilate(
        image, 
        kernel, 
        iterations=1, 
        borderType=cv2.BORDER_CONSTANT, 
        borderValue=0
    )

def random_erode(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Makes white ink thinner on black background."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(
        image, 
        kernel, 
        iterations=1, 
        borderType=cv2.BORDER_CONSTANT, 
        borderValue=0
    )

@dataclass
class AugmentationConfig:
    rotation_chance: float = 0.5
    rotation_max_angle: float = 3.5

    shear_chance: float = 0.5
    shear_max_factor: float = 0.2

    thickness_chance: float = 0.3
    kernel_size: int = 2

    elastic_chance: float = 0.5
    elastic_alpha: float = 3.
    elastic_sigma: float = 3.0


def get_augmentor(config: AugmentationConfig):
    def online_augment(image: np.ndarray) -> np.ndarray:
        # 1. Start with a clean, owned copy of the data
        image = np.ascontiguousarray(image).copy()

        # 2. Thickness FIRST (More stable on non-rotated images)
        if np.random.rand() < config.thickness_chance:
            if np.random.rand() < 0.5:
                image = random_dilate(image, kernel_size=config.kernel_size)
            else:
                image = random_erode(image, kernel_size=config.kernel_size)

        # 3. Geometric
        if np.random.rand() < config.rotation_chance:
            image = random_rotation(image, max_angle=config.rotation_max_angle)
        if np.random.rand() < config.shear_chance:
            image = random_shear(image, max_shear=config.shear_max_factor)

        # 4. Elastic
        if np.random.rand() < config.elastic_chance:
            image = elastic_distortion(
                image, alpha=config.elastic_alpha, sigma=config.elastic_sigma
            )

        return np.ascontiguousarray(image.astype(np.uint8))

    return online_augment