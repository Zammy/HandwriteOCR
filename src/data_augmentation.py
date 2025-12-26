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
