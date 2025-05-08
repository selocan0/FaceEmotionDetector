from typing import Tuple
import numpy as np
import cv2

from deepface.commons import package_utils
from tensorflow.keras.preprocessing import image


def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """
    Normalize input image.

    Args:
        img (np.ndarray): The input image.
        normalization (str): Normalization method (e.g., 'base', 'Facenet', 'VGGFace').

    Returns:
        np.ndarray: Normalized image.
    """
    if normalization == "base":
        return img

    img *= 255  # Restore from range [0,1] to [0,255]

    if normalization == "raw":
        return img

    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std

    elif normalization == "Facenet2018":
        img /= 127.5
        img -= 1

    elif normalization == "VGGFace":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    elif normalization == "VGGFace2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912

    elif normalization == "ArcFace":
        img -= 127.5
        img /= 128

    else:
        raise ValueError(f"Unimplemented normalization type: {normalization}")

    return img


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to the target size with padding.

    Args:
        img (np.ndarray): Input image.
        target_size (Tuple[int, int]): Desired output size (height, width).

    Returns:
        np.ndarray: Resized and padded image.
    """

    # Defensive checks
    if img is None:
        raise ValueError("resize_image: input image is None.")
    if not isinstance(img, np.ndarray):
        raise TypeError("resize_image: input must be a numpy array.")
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("resize_image: input image has invalid dimensions.")
    if len(img.shape) != 3 or img.shape[2] not in [1, 3]:
        raise ValueError(f"resize_image: image must have 1 or 3 channels, got shape {img.shape}")

    # Resize with aspect ratio
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    if dsize[0] == 0 or dsize[1] == 0:
        raise ValueError(f"resize_image: calculated dsize is invalid: {dsize}")

    img = cv2.resize(img, dsize)

    # Padding to match target size
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # Double check
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # Convert to 4D
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Normalize
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0

    return img
