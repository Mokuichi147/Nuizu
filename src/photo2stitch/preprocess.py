"""Enhanced image preprocessing for embroidery conversion.

Optimizes photo input for better color quantization and region
extraction. Includes contrast enhancement, noise reduction,
edge-aware smoothing, and background detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_photo(
    image: np.ndarray,
    max_dim: int = 800,
    denoise: bool = True,
    enhance_contrast: bool = True,
    edge_smooth: bool = True,
    sharpen_edges: bool = True,
    saturation_boost: float = 1.2,
) -> np.ndarray:
    """Full preprocessing pipeline for photo input.

    Args:
        image: RGB image array (H, W, 3).
        max_dim: Maximum dimension for processing.
        denoise: Apply denoising.
        enhance_contrast: Apply CLAHE contrast enhancement.
        edge_smooth: Apply edge-preserving smoothing.
        sharpen_edges: Sharpen edges after smoothing.
        saturation_boost: Color saturation multiplier (1.0 = no change).

    Returns:
        Preprocessed RGB image.
    """
    h, w = image.shape[:2]

    # 1. Resize for processing
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)

    result = image.copy()

    # 2. Denoise (Non-local means)
    if denoise:
        # Convert to BGR for OpenCV functions
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)
        result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3. Contrast enhancement (CLAHE in LAB space)
    if enhance_contrast:
        result = _enhance_contrast_clahe(result)

    # 4. Saturation boost
    if abs(saturation_boost - 1.0) > 0.01:
        result = _adjust_saturation(result, saturation_boost)

    # 5. Edge-preserving smoothing
    if edge_smooth:
        result = _edge_preserving_smooth(result)

    # 6. Sharpen edges
    if sharpen_edges:
        result = _unsharp_mask(result, strength=0.5)

    return result


def _enhance_contrast_clahe(image: np.ndarray,
                             clip_limit: float = 2.0,
                             grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB color space.

    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    enhances local contrast while preventing over-amplification.
    Applied only to L channel to preserve colors.
    """
    # Convert RGB -> LAB
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    l_enhanced = clahe.apply(l_channel)

    lab[:, :, 0] = l_enhanced

    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)


def _adjust_saturation(image: np.ndarray,
                        factor: float = 1.2) -> np.ndarray:
    """Adjust color saturation.

    Embroidery thread colors are typically more saturated than
    photos, so a slight boost helps match the final result.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    hsv = hsv.astype(np.uint8)
    bgr_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)


def _edge_preserving_smooth(image: np.ndarray,
                             sigma_s: float = 30,
                             sigma_r: float = 0.3) -> np.ndarray:
    """Edge-preserving smoothing using bilateral or stylization.

    Smooths flat areas while preserving sharp edges, which is
    ideal for creating clean region boundaries for stitching.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Bilateral filter (edge-preserving)
    smoothed = cv2.bilateralFilter(bgr, d=9,
                                    sigmaColor=75,
                                    sigmaSpace=75)

    return cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB)


def _unsharp_mask(image: np.ndarray,
                   strength: float = 0.5,
                   kernel_size: int = 5) -> np.ndarray:
    """Apply unsharp mask to enhance edges.

    Strengthens region boundaries for cleaner segmentation.
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(
        image, 1.0 + strength,
        blurred, -strength,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def detect_background(image: np.ndarray,
                       method: str = 'corner'
                       ) -> Optional[Tuple[int, int, int]]:
    """Detect the dominant background color of an image.

    Useful for deciding whether to stitch the background or
    leave it as fabric color.

    Args:
        image: RGB image.
        method: 'corner' (sample corners) or 'edge' (sample edges).

    Returns:
        (R, G, B) tuple of detected background color, or None.
    """
    h, w = image.shape[:2]

    if method == 'corner':
        # Sample corner regions (10% of image)
        margin_x = max(1, w // 10)
        margin_y = max(1, h // 10)

        corners = np.concatenate([
            image[:margin_y, :margin_x].reshape(-1, 3),
            image[:margin_y, -margin_x:].reshape(-1, 3),
            image[-margin_y:, :margin_x].reshape(-1, 3),
            image[-margin_y:, -margin_x:].reshape(-1, 3),
        ])
    else:
        # Sample edge pixels
        edge_size = max(1, min(h, w) // 20)
        edges = np.concatenate([
            image[:edge_size, :].reshape(-1, 3),
            image[-edge_size:, :].reshape(-1, 3),
            image[:, :edge_size].reshape(-1, 3),
            image[:, -edge_size:].reshape(-1, 3),
        ])
        corners = edges

    # Find most common color in sampled pixels
    # Use simple mean with outlier rejection
    mean_color = np.median(corners, axis=0).astype(int)

    # Check if corners are consistent (low variance = likely background)
    variance = np.std(corners, axis=0).mean()

    if variance < 50:  # Consistent background detected
        return tuple(mean_color)

    return None


def auto_crop_to_subject(image: np.ndarray,
                          padding_ratio: float = 0.05
                          ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Auto-crop image to the main subject, removing excess background.

    Args:
        image: RGB image.
        padding_ratio: Padding around subject as fraction of dimensions.

    Returns:
        (cropped_image, (x, y, w, h)) bounding box used.
    """
    gray = cv2.cvtColor(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2GRAY,
    )

    # Edge detection
    edges = cv2.Canny(gray, 30, 100)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find bounding box of all edges
    coords = cv2.findNonZero(edges)
    if coords is None:
        return image, (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    pad_x = int(image.shape[1] * padding_ratio)
    pad_y = int(image.shape[0] * padding_ratio)

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)

    cropped = image[y:y + h, x:x + w]
    return cropped, (x, y, w, h)
