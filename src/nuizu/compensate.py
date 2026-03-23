"""Pull compensation for embroidery stitch adjustment.

When embroidery thread is stitched, it pulls the fabric inward,
causing the final result to be slightly narrower than intended.
Pull compensation widens the stitch area to counteract this effect.

Typical compensation values:
  - Light fabric (organza, silk): 0.2-0.3mm
  - Medium fabric (cotton, linen): 0.3-0.4mm
  - Heavy fabric (denim, canvas): 0.4-0.6mm
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def apply_pull_compensation(
    contour_mm: np.ndarray,
    compensation_mm: float = 0.3,
) -> np.ndarray:
    """Expand a contour outward to compensate for thread pull.

    Uses Minkowski sum approximation by offsetting each vertex
    along its outward normal.

    Args:
        contour_mm: (N, 2) contour points in mm.
        compensation_mm: Outward expansion distance in mm.

    Returns:
        Expanded contour array.
    """
    if len(contour_mm) < 3 or compensation_mm <= 0:
        return contour_mm.copy()

    n = len(contour_mm)
    expanded = np.empty_like(contour_mm)

    for i in range(n):
        # Previous and next points
        p_prev = contour_mm[(i - 1) % n]
        p_curr = contour_mm[i]
        p_next = contour_mm[(i + 1) % n]

        # Edge vectors
        e1 = p_curr - p_prev
        e2 = p_next - p_curr

        # Normals (pointing outward, assuming CCW winding)
        n1 = np.array([-e1[1], e1[0]])
        n2 = np.array([-e2[1], e2[0]])

        len1 = np.linalg.norm(n1)
        len2 = np.linalg.norm(n2)

        if len1 > 1e-8:
            n1 /= len1
        if len2 > 1e-8:
            n2 /= len2

        # Average normal at vertex
        avg_normal = n1 + n2
        avg_len = np.linalg.norm(avg_normal)
        if avg_len > 1e-8:
            avg_normal /= avg_len
        else:
            avg_normal = n1

        # Check winding direction to determine inward/outward
        # Use cross product of edges
        cross = e1[0] * e2[1] - e1[1] * e2[0]

        # Miter factor: compensate for angle at vertex
        dot = np.dot(n1, n2)
        miter = 1.0 / max(0.5, (1.0 + dot) / 2.0)
        miter = min(miter, 3.0)  # Cap to avoid spikes

        expanded[i] = p_curr + avg_normal * compensation_mm * miter

    # Verify the expanded contour is valid (larger area)
    orig_area = _polygon_area(contour_mm)
    new_area = _polygon_area(expanded)

    if new_area < orig_area:
        # Normals were pointing inward, flip
        for i in range(n):
            offset = expanded[i] - contour_mm[i]
            expanded[i] = contour_mm[i] - offset

    return expanded


def inset_contour(
    contour_mm: np.ndarray,
    inset_mm: float = 0.2,
) -> Optional[np.ndarray]:
    """Shrink a contour inward by inset_mm.

    Used to offset the outline stitch path inward by half the thread
    width so the outer edge of the thread aligns with the original
    contour boundary.

    Args:
        contour_mm: (N, 2) contour points in mm.
        inset_mm: Inward offset distance in mm.

    Returns:
        Inset contour array, or None if the contour collapses.
    """
    if len(contour_mm) < 3 or inset_mm <= 0:
        return contour_mm.copy()

    # Use the same vertex-normal logic as pull compensation,
    # but offset inward instead of outward.
    n = len(contour_mm)
    inset = np.empty_like(contour_mm)

    for i in range(n):
        p_prev = contour_mm[(i - 1) % n]
        p_curr = contour_mm[i]
        p_next = contour_mm[(i + 1) % n]

        e1 = p_curr - p_prev
        e2 = p_next - p_curr

        n1 = np.array([-e1[1], e1[0]])
        n2 = np.array([-e2[1], e2[0]])

        len1 = np.linalg.norm(n1)
        len2 = np.linalg.norm(n2)

        if len1 > 1e-8:
            n1 /= len1
        if len2 > 1e-8:
            n2 /= len2

        avg_normal = n1 + n2
        avg_len = np.linalg.norm(avg_normal)
        if avg_len > 1e-8:
            avg_normal /= avg_len
        else:
            avg_normal = n1

        dot = np.dot(n1, n2)
        miter = 1.0 / max(0.5, (1.0 + dot) / 2.0)
        miter = min(miter, 3.0)

        inset[i] = p_curr + avg_normal * inset_mm * miter

    # Determine which direction is inward by checking area change.
    # Inset should produce a *smaller* area.
    orig_area = abs(_polygon_area(contour_mm))
    new_area = abs(_polygon_area(inset))

    if new_area >= orig_area:
        # Normals pointed outward — flip to go inward.
        for i in range(n):
            offset = inset[i] - contour_mm[i]
            inset[i] = contour_mm[i] - offset
        new_area = abs(_polygon_area(inset))

    # If the inset collapsed the contour, return None.
    if new_area < orig_area * 0.05:
        return None

    return inset


def apply_fill_compensation(
    stitches: List[Tuple[float, float]],
    contour_mm: np.ndarray,
    compensation_mm: float = 0.3,
    fill_angle: float = 0.0,
) -> List[Tuple[float, float]]:
    """Apply pull compensation to fill stitches.

    For fill stitches, compensation is applied perpendicular to
    the fill direction, extending each row slightly beyond the
    boundary.

    Args:
        stitches: List of (x, y) fill stitch points.
        contour_mm: Region contour for reference.
        compensation_mm: Compensation distance in mm.
        fill_angle: Fill angle in degrees for direction reference.

    Returns:
        Compensated stitch list.
    """
    if not stitches or compensation_mm <= 0:
        return stitches

    # For fill stitches, extend the first and last stitch of each
    # "row" outward by the compensation amount.

    angle_rad = np.radians(fill_angle)
    # Direction perpendicular to fill lines (the row direction)
    row_dx = np.cos(angle_rad)
    row_dy = np.sin(angle_rad)

    compensated = []
    prev_y_group = None
    row_buffer = []

    # Group stitches by approximate row
    # (stitches on the same scanline have similar projected position
    #  perpendicular to the fill angle)
    def proj_perp(x, y):
        return -x * np.sin(angle_rad) + y * np.cos(angle_rad)

    for x, y in stitches:
        p = proj_perp(x, y)
        if prev_y_group is None or abs(p - prev_y_group) > 0.2:
            # New row - flush previous
            if row_buffer:
                _extend_row(row_buffer, compensation_mm,
                            row_dx, row_dy, compensated)
            row_buffer = [(x, y)]
            prev_y_group = p
        else:
            row_buffer.append((x, y))

    # Flush last row
    if row_buffer:
        _extend_row(row_buffer, compensation_mm,
                    row_dx, row_dy, compensated)

    return compensated


def _extend_row(row: List[Tuple[float, float]],
                comp: float,
                dx: float, dy: float,
                output: List[Tuple[float, float]]):
    """Extend a fill row by compensation amount at both ends."""
    if len(row) < 2:
        output.extend(row)
        return

    # Determine row direction from first to last stitch
    x0, y0 = row[0]
    x1, y1 = row[-1]
    rdx = x1 - x0
    rdy = y1 - y0
    rlen = np.sqrt(rdx * rdx + rdy * rdy)

    if rlen > 0.1:
        rdx /= rlen
        rdy /= rlen
    else:
        rdx, rdy = dx, dy

    # Extend first point backward
    new_first = (x0 - rdx * comp, y0 - rdy * comp)
    # Extend last point forward
    new_last = (x1 + rdx * comp, y1 + rdy * comp)

    output.append(new_first)
    output.extend(row[1:-1])
    output.append(new_last)


def _polygon_area(pts: np.ndarray) -> float:
    """Compute signed area of polygon using shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    return area / 2.0
