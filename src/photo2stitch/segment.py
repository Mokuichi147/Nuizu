"""Region segmentation for embroidery pattern generation.

Extracts contiguous color regions from a quantized label map
and converts them to polygon contours suitable for stitch generation.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Region:
    """A contiguous color region to be stitched."""
    color_index: int
    color_rgb: Tuple[int, int, int]
    contour: np.ndarray           # Simplified outer contour (N, 2) in pixel coords
    holes: List[np.ndarray] = field(default_factory=list)  # Inner contours
    area: float = 0.0            # Area in pixels
    centroid: Tuple[float, float] = (0.0, 0.0)

    # Raw (non-simplified) contour for accurate fill mask
    contour_raw: Optional[np.ndarray] = None
    holes_raw: Optional[List[np.ndarray]] = None

    # Converted to mm coordinates after scaling
    contour_mm: Optional[np.ndarray] = None       # Simplified (for outline)
    holes_mm: Optional[List[np.ndarray]] = None
    contour_raw_mm: Optional[np.ndarray] = None    # Raw (for fill)
    holes_raw_mm: Optional[List[np.ndarray]] = None


def _chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Smooth polygon using Chaikin's corner-cutting algorithm.

    Replaces each corner with two points at 25%/75% along adjacent
    edges, producing a smoother curve with each iteration.
    """
    if len(points) < 4:
        return points

    pts = points.copy()
    for _ in range(iterations):
        n = len(pts)
        new_pts = np.empty((2 * n, 2), dtype=pts.dtype)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts[2 * i] = 0.75 * p0 + 0.25 * p1
            new_pts[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = new_pts

    return pts


def extract_regions(label_map: np.ndarray,
                    palette: List[Tuple[int, int, int]],
                    min_area: int = 50,
                    morph_cleanup: bool = True,
                    simplify_epsilon_min: float = 1.0,
                    ) -> List[Region]:
    """Extract color regions from quantized label map.

    Args:
        label_map: (H, W) integer array of color indices.
        palette: List of (R, G, B) color tuples.
        min_area: Minimum region area in pixels to keep.
        morph_cleanup: Apply morphological cleanup to masks.
        simplify_epsilon_min: Minimum contour simplification epsilon.

    Returns:
        List of Region objects sorted by area (largest first).
    """
    regions = []
    n_colors = label_map.max() + 1

    for color_idx in range(n_colors):
        mask = (label_map == color_idx).astype(np.uint8) * 255

        if mask.sum() == 0:
            continue

        if morph_cleanup:
            # Morphological cleanup: close small gaps, smooth edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        # Process top-level contours (outer boundaries)
        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):
            # hier = [next, prev, first_child, parent]
            if hier[3] != -1:
                # This is a hole (has parent), skip - handled by parent
                continue

            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Simplify contour
            epsilon = max(simplify_epsilon_min,
                          0.0003 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            outer = approx.reshape(-1, 2).astype(np.float64)

            if len(outer) < 3:
                continue

            # Keep raw (non-simplified) contour for fill mask
            outer_raw = contour.reshape(-1, 2).astype(np.float64)

            # Collect holes (children of this contour)
            holes = []
            holes_raw = []
            child_idx = hier[2]
            while child_idx != -1:
                hole_contour = contours[child_idx]
                hole_area = cv2.contourArea(hole_contour)
                if hole_area >= min_area * 0.5:
                    hole_raw = hole_contour.reshape(-1, 2).astype(np.float64)
                    eps_h = max(simplify_epsilon_min,
                                0.0003 * cv2.arcLength(hole_contour, True))
                    hole_approx = cv2.approxPolyDP(hole_contour, eps_h, True)
                    hole_pts = hole_approx.reshape(-1, 2).astype(np.float64)
                    if len(hole_pts) >= 3:
                        holes.append(hole_pts)
                        holes_raw.append(hole_raw)
                child_idx = hierarchy[child_idx][0]

            # Compute centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = outer.mean(axis=0)

            color_rgb = palette[color_idx] if color_idx < len(palette) \
                else (128, 128, 128)

            regions.append(Region(
                color_index=color_idx,
                color_rgb=color_rgb,
                contour=outer,
                holes=holes,
                area=area,
                centroid=(float(cx), float(cy)),
                contour_raw=outer_raw,
                holes_raw=holes_raw,
            ))

    # Sort by area descending (stitch large regions first for underlay)
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def scale_regions_to_mm(regions: List[Region],
                        image_shape: Tuple[int, int],
                        target_width_mm: float,
                        target_height_mm: float) -> List[Region]:
    """Convert region coordinates from pixels to mm.

    Args:
        regions: List of Region objects with pixel coordinates.
        image_shape: (H, W) of the source image.
        target_width_mm: Target embroidery width in mm.
        target_height_mm: Target embroidery height in mm.

    Returns:
        Regions with contour_mm and holes_mm populated.
    """
    h, w = image_shape[:2]
    scale_x = target_width_mm / w
    scale_y = target_height_mm / h

    for region in regions:
        region.contour_mm = region.contour.copy()
        region.contour_mm[:, 0] *= scale_x
        region.contour_mm[:, 1] *= scale_y

        region.holes_mm = []
        for hole in region.holes:
            hole_mm = hole.copy()
            hole_mm[:, 0] *= scale_x
            hole_mm[:, 1] *= scale_y
            region.holes_mm.append(hole_mm)

        # Scale raw contours for fill mask
        if region.contour_raw is not None:
            region.contour_raw_mm = region.contour_raw.copy()
            region.contour_raw_mm[:, 0] *= scale_x
            region.contour_raw_mm[:, 1] *= scale_y

        region.holes_raw_mm = []
        if region.holes_raw:
            for hole in region.holes_raw:
                hole_mm = hole.copy()
                hole_mm[:, 0] *= scale_x
                hole_mm[:, 1] *= scale_y
                region.holes_raw_mm.append(hole_mm)

    return regions
