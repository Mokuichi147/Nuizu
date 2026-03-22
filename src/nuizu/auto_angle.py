"""Automatic fill angle optimization for embroidery regions.

Determines the optimal fill stitch direction for each region based on
its geometry. The goal is to minimize the number of short row segments
(which cause more needle penetrations and slower stitching) and to
align fill direction with the longest axis of the region.
"""

import cv2
import numpy as np
from typing import Optional


def compute_optimal_fill_angle(contour_mm: np.ndarray,
                                base_angle: float = 45.0,
                                variation: bool = True
                                ) -> float:
    """Compute the optimal fill angle for a region.

    Uses PCA (principal component analysis) on the contour points
    to find the major axis direction, then aligns fill lines
    perpendicular to the major axis for maximum row length.

    For neighboring regions, a slight angle variation is applied
    to avoid moiré-like visual artifacts.

    Args:
        contour_mm: (N, 2) contour points in mm.
        base_angle: Fallback angle if computation fails.
        variation: Add slight random variation.

    Returns:
        Optimal fill angle in degrees.
    """
    if len(contour_mm) < 3:
        return base_angle

    try:
        pts = contour_mm.astype(np.float64)
        mean = pts.mean(axis=0)
        centered = pts - mean

        # Covariance matrix
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Major axis is eigenvector with largest eigenvalue
        major_idx = np.argmax(eigenvalues)
        major_axis = eigenvectors[:, major_idx]

        # Fill perpendicular to major axis gives longest rows
        angle_rad = np.arctan2(major_axis[1], major_axis[0])
        # Perpendicular
        angle_deg = np.degrees(angle_rad) + 90.0

        # Normalize to [0, 180) range
        angle_deg = angle_deg % 180.0

        return angle_deg

    except Exception:
        return base_angle


def compute_optimal_fill_angle_from_mask(
    contour: np.ndarray,
    image_shape: tuple,
    base_angle: float = 45.0,
) -> float:
    """Compute fill angle using minimum area bounding rectangle.

    Alternative method using OpenCV's minAreaRect for more robust
    orientation detection, especially for irregular shapes.

    Args:
        contour: Contour points in pixel coordinates.
        image_shape: (H, W) of source image.
        base_angle: Fallback angle.

    Returns:
        Optimal fill angle in degrees.
    """
    if len(contour) < 5:
        return base_angle

    try:
        pts = contour.astype(np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        angle = rect[2]  # Rotation angle
        w, h = rect[1]   # Width, height of bounding rect

        # minAreaRect returns angle in [-90, 0)
        # We want perpendicular to the long axis
        if w < h:
            angle += 90.0

        # Add 90 for perpendicular fill
        fill_angle = (angle + 90.0) % 180.0

        return fill_angle

    except Exception:
        return base_angle


def compute_angle_for_thin_region(contour_mm: np.ndarray,
                                   aspect_threshold: float = 3.0
                                   ) -> Optional[float]:
    """Detect thin/elongated regions and return aligned fill angle.

    Thin regions like stems, borders, or text strokes should be
    filled along their length rather than across.

    Args:
        contour_mm: (N, 2) contour points in mm.
        aspect_threshold: Minimum aspect ratio to consider "thin".

    Returns:
        Fill angle in degrees, or None if not a thin region.
    """
    if len(contour_mm) < 5:
        return None

    try:
        pts = contour_mm.astype(np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]

        if w <= 0 or h <= 0:
            return None

        aspect = max(w, h) / min(w, h)

        if aspect >= aspect_threshold:
            # This is a thin region - fill along its length
            angle = rect[2]
            if w > h:
                # Long axis is along width
                return angle % 180.0
            else:
                return (angle + 90.0) % 180.0

        return None

    except Exception:
        return None
