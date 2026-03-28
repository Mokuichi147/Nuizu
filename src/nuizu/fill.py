"""Fill stitch generation for embroidery regions.

Implements scanline-based fill stitching with configurable angle,
density, and stitch length. Uses raster-based polygon operations
via OpenCV instead of Shapely.
"""

import cv2
import numpy as np
from typing import List, Tuple

# Internal raster resolution: pixels per mm for fill computation
_RASTER_PPM = 20


def _rasterize_region(contour_mm: np.ndarray,
                      holes_mm: List[np.ndarray],
                      ppm: float = _RASTER_PPM,
                      ) -> Tuple[np.ndarray, float, float]:
    """Rasterize a region to a binary mask for scanline processing.

    Returns:
        (mask, origin_x, origin_y): mask image and its origin in mm.
    """
    all_pts = contour_mm.copy()
    min_x, min_y = all_pts.min(axis=0) - 0.3
    max_x, max_y = all_pts.max(axis=0) + 0.3

    w = int(np.ceil((max_x - min_x) * ppm)) + 2
    h = int(np.ceil((max_y - min_y) * ppm)) + 2

    # Safety cap
    w = min(w, 4000)
    h = min(h, 4000)

    mask = np.zeros((h, w), dtype=np.uint8)

    def to_px(pts):
        px = pts.copy()
        px[:, 0] = (px[:, 0] - min_x) * ppm
        px[:, 1] = (px[:, 1] - min_y) * ppm
        return np.round(px).astype(np.int32)

    contour_px = to_px(contour_mm)
    cv2.fillPoly(mask, [contour_px], 255)
    cv2.polylines(mask, [contour_px], True, 255, 1)
    for hole in holes_mm:
        cv2.fillPoly(mask, [to_px(hole)], 0)

    return mask, min_x, min_y


def generate_fill_stitches(
    contour: np.ndarray,
    holes: List[np.ndarray],
    fill_angle: float = 0.0,
    row_spacing: float = 0.136,
    stitch_length: float = 3.0,
    stagger_offset: float = 0.5,
    underlay: bool = True,
    underlay_angle_offset: float = 90.0,
    underlay_spacing: float = 1.5,
) -> List[Tuple[float, float]]:
    """Generate fill stitches for a region using scanline hatching.

    Args:
        contour: Outer boundary points (N, 2) in mm.
        holes: List of hole boundary arrays in mm.
        fill_angle: Angle of fill lines in degrees.
        row_spacing: Distance between fill rows in mm.
        stitch_length: Maximum stitch length in mm.
        stagger_offset: Row-to-row offset as fraction of stitch_length.
        underlay: Whether to generate underlay stitches.
        underlay_angle_offset: Angle offset for underlay (degrees).
        underlay_spacing: Row spacing for underlay in mm.

    Returns:
        List of (x, y) stitch coordinates in mm.
    """
    segments = generate_fill_stitch_segments(
        contour=contour,
        holes=holes,
        fill_angle=fill_angle,
        row_spacing=row_spacing,
        stitch_length=stitch_length,
        stagger_offset=stagger_offset,
        underlay=underlay,
        underlay_angle_offset=underlay_angle_offset,
        underlay_spacing=underlay_spacing,
    )

    stitches = []
    for segment in segments:
        stitches.extend(segment)
    return stitches


def generate_fill_stitch_segments(
    contour: np.ndarray,
    holes: List[np.ndarray],
    fill_angle: float = 0.0,
    row_spacing: float = 0.136,
    stitch_length: float = 3.0,
    stagger_offset: float = 0.5,
    underlay: bool = True,
    underlay_angle_offset: float = 90.0,
    underlay_spacing: float = 1.5,
) -> List[List[Tuple[float, float]]]:
    """Generate fill stitches as separate segments.

    Each disconnected span is returned as an independent segment so
    callers can insert jump stitches between spans.
    """
    if len(contour) < 3:
        return []

    all_segments: List[List[Tuple[float, float]]] = []

    if underlay:
        underlay_segments = _rotated_scanline_fill_segments(
            contour, holes,
            angle=fill_angle + underlay_angle_offset,
            spacing=underlay_spacing,
            stitch_length=stitch_length * 1.5,
            stagger=0.0,
        )
        all_segments.extend(underlay_segments)

    fill_segments = _rotated_scanline_fill_segments(
        contour, holes,
        angle=fill_angle,
        spacing=row_spacing,
        stitch_length=stitch_length,
        stagger=stagger_offset * stitch_length,
    )
    all_segments.extend(fill_segments)

    return all_segments


def _rotate_points(pts: np.ndarray, angle_deg: float,
                   center: np.ndarray) -> np.ndarray:
    """Rotate points around center by angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    centered = pts - center
    rotated = np.empty_like(centered)
    rotated[:, 0] = centered[:, 0] * cos_a - centered[:, 1] * sin_a
    rotated[:, 1] = centered[:, 0] * sin_a + centered[:, 1] * cos_a
    return rotated + center


def _rotated_scanline_fill(
    contour: np.ndarray,
    holes: List[np.ndarray],
    angle: float,
    spacing: float,
    stitch_length: float,
    stagger: float,
) -> List[Tuple[float, float]]:
    """Backward-compatible flat fill output."""
    segments = _rotated_scanline_fill_segments(
        contour=contour,
        holes=holes,
        angle=angle,
        spacing=spacing,
        stitch_length=stitch_length,
        stagger=stagger,
    )
    stitches = []
    for segment in segments:
        stitches.extend(segment)
    return stitches


def _rotated_scanline_fill_segments(
    contour: np.ndarray,
    holes: List[np.ndarray],
    angle: float,
    spacing: float,
    stitch_length: float,
    stagger: float,
) -> List[List[Tuple[float, float]]]:
    """Fill region with scanlines at given angle.

    Rotate polygon so scanlines are horizontal, fill, rotate back.
    """
    if len(contour) < 3:
        return []

    center = contour.mean(axis=0)

    rot_contour = _rotate_points(contour, -angle, center)
    rot_holes = [_rotate_points(h, -angle, center) for h in holes]

    mask, ox, oy = _rasterize_region(rot_contour, rot_holes)
    h, w = mask.shape
    ppm = _RASTER_PPM

    # Work in mm to keep row spacing exactly uniform.
    # Convert row y-positions to pixel rows only for mask lookup.
    height_mm = h / ppm
    stagger_mm = stagger if stagger else 0.0

    segments_rotated: List[List[Tuple[float, float]]] = []
    reverse = False
    row_idx = 0

    y_mm = 0.0
    while y_mm <= height_mm:
        row_px = min(round(y_mm * ppm), h - 1)

        row_data = mask[row_px, :]
        spans = _find_spans(row_data)

        if not spans:
            y_mm += spacing
            row_idx += 1
            continue

        if reverse:
            spans = [(e, s) for s, e in reversed(spans)]

        row_y_mm = oy + y_mm
        row_offset_mm = stagger_mm * (row_idx % 2)

        for span_start, span_end in spans:
            s = min(span_start, span_end)
            e = max(span_start, span_end)
            going_reverse = span_start > span_end

            s_mm = ox + s / ppm
            e_mm = ox + e / ppm

            # Always start and end at the span edges so fill
            # reaches the contour boundary on every row.  Stagger
            # shifts only the intermediate stitch grid.
            span_stitches = [(s_mm, row_y_mm)]

            offset = (row_offset_mm % stitch_length) if stitch_length else 0
            # スタガーオフセット位置に中間ステッチを入れ、
            # 最初のステッチ間距離が stitch_length を超えないようにする。
            if offset > 0.05:
                first_x = s_mm + offset
                if first_x < e_mm - 0.05:
                    span_stitches.append((first_x, row_y_mm))
            x_mm = s_mm + offset + stitch_length
            while x_mm < e_mm - 0.05:
                span_stitches.append((x_mm, row_y_mm))
                x_mm += stitch_length

            if e_mm - s_mm > 0.05:
                span_stitches.append((e_mm, row_y_mm))

            if going_reverse:
                span_stitches.reverse()

            if span_stitches:
                segments_rotated.append(span_stitches)

        reverse = not reverse
        y_mm += spacing
        row_idx += 1

    if not segments_rotated:
        return []

    segments_back: List[List[Tuple[float, float]]] = []
    for segment in segments_rotated:
        pts = np.array(segment)
        pts_back = _rotate_points(pts, angle, center)
        segments_back.append([tuple(p) for p in pts_back])

    # 元座標でステッチ間の中間点が領域外なら分割する。
    # 回転後のスキャンラインでは連続していても、元の形状では
    # 凹部を跨いでいる可能性がある。
    mask_orig, ox_orig, oy_orig = _rasterize_region(contour, holes)
    validated: List[List[Tuple[float, float]]] = []
    for segment in segments_back:
        validated.extend(
            _split_outside(segment, mask_orig, ox_orig, oy_orig, ppm)
        )

    return validated


def _split_outside(
    segment: List[Tuple[float, float]],
    mask: np.ndarray,
    ox: float,
    oy: float,
    ppm: float,
) -> List[List[Tuple[float, float]]]:
    """連続するステッチ間のラインが領域外を通過するならセグメントを分割する。

    2点間のラインを細かくサンプリングし、領域外のピクセルを通過する
    場合はそこでセグメントを分割する。
    """
    if len(segment) < 2:
        return [segment] if segment else []

    h, w = mask.shape
    # サンプリング間隔（mm）。細かいほど凹部の検出精度が上がる。
    sample_step_mm = 0.3

    def _line_crosses_outside(x0: float, y0: float,
                              x1: float, y1: float) -> bool:
        """2点間のラインが領域外を通るか判定する。"""
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.01:
            return False

        n_samples = max(int(dist / sample_step_mm), 3)
        for k in range(1, n_samples):
            t = k / n_samples
            sx = x0 + dx * t
            sy = y0 + dy * t
            px = int(round((sx - ox) * ppm))
            py = int(round((sy - oy) * ppm))
            if px < 0 or px >= w or py < 0 or py >= h:
                return True
            if mask[py, px] == 0:
                return True
        return False

    result: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = [segment[0]]

    for i in range(1, len(segment)):
        x0, y0 = segment[i - 1]
        x1, y1 = segment[i]

        if _line_crosses_outside(x0, y0, x1, y1):
            if current:
                result.append(current)
            current = [segment[i]]
        else:
            current.append(segment[i])

    if current:
        result.append(current)
    return result


def _find_spans(row: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous filled spans in a row of pixels."""
    spans = []
    in_span = False
    start = 0

    for x in range(len(row)):
        if row[x] > 0:
            if not in_span:
                start = x
                in_span = True
        else:
            if in_span:
                spans.append((start, x - 1))
                in_span = False

    if in_span:
        spans.append((start, len(row) - 1))

    return spans


def generate_outline_stitches(
    contour: np.ndarray,
    stitch_length: float = 2.5,
    satin_width: float = 1.5,
    satin: bool = False,
    region_mask: np.ndarray = None,
    mask_ox: float = 0.0,
    mask_oy: float = 0.0,
    mask_ppm: float = _RASTER_PPM,
) -> List[Tuple[float, float]]:
    """Generate outline stitches (running or satin) along a contour.

    Args:
        contour: (N, 2) array of contour points in mm.
        stitch_length: Length of each stitch in mm.
        satin_width: Width of satin stitches (if satin=True).
        satin: If True, generate satin stitches; else running stitch.
        region_mask: リージョンのバイナリマスク（凹部検出用）。
        mask_ox, mask_oy: マスクの原点（mm）。
        mask_ppm: マスクの解像度（pixels per mm）。

    Returns:
        List of (x, y) stitch coordinates.
    """
    if len(contour) < 2:
        return []

    pts = contour.copy()
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    if not satin:
        return _stitch_along_path(pts, stitch_length,
                                  region_mask, mask_ox, mask_oy, mask_ppm)

    stitches = []
    half_w = satin_width / 2
    side = 1

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    pos = 0
    step = stitch_length / 2

    while pos < total_length:
        x, y = _interpolate_path(pts, cum_lengths, pos)

        pos2 = min(pos + 0.1, total_length)
        x2, y2 = _interpolate_path(pts, cum_lengths, pos2)
        dx, dy = x2 - x, y2 - y
        length = np.sqrt(dx * dx + dy * dy)

        if length > 1e-6:
            nx, ny = -dy / length, dx / length
        else:
            nx, ny = 0, 1

        sx = x + nx * half_w * side
        sy = y + ny * half_w * side
        stitches.append((float(sx), float(sy)))

        side *= -1
        pos += step

    return stitches


def _stitch_along_path(pts: np.ndarray,
                       stitch_length: float,
                       region_mask: np.ndarray = None,
                       mask_ox: float = 0.0,
                       mask_oy: float = 0.0,
                       mask_ppm: float = _RASTER_PPM,
                       ) -> List[Tuple[float, float]]:
    """Generate evenly spaced stitches along a polyline path.

    region_mask が指定された場合、ステッチ間の直線がリージョン外を
    通る箇所でパスに沿った中間ステッチを自動挿入する。
    """
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 0.1:
        return [tuple(pts[0])]

    # まず通常間隔でステッチ位置を決定
    positions: List[float] = []
    pos = 0.0
    while pos <= total_length:
        positions.append(pos)
        pos += stitch_length
    if abs(positions[-1] - total_length) > 0.05:
        positions.append(total_length)

    raw_stitches = []
    for p in positions:
        x, y = _interpolate_path(pts, cum_lengths, p)
        raw_stitches.append((float(x), float(y), p))

    if region_mask is None:
        return [(x, y) for x, y, _ in raw_stitches]

    # ステッチ間の直線がリージョン外を通る場合、
    # パスに沿った中間点を挿入する
    h, w = region_mask.shape
    sample_step = 0.2  # mm

    def _line_exits_region(x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.05:
            return False
        n = max(int(dist / sample_step), 3)
        for k in range(1, n):
            t = k / n
            px = int(round((x0 + dx * t - mask_ox) * mask_ppm))
            py = int(round((y0 + dy * t - mask_oy) * mask_ppm))
            if px < 0 or px >= w or py < 0 or py >= h:
                return True
            if region_mask[py, px] == 0:
                return True
        return False

    result: List[Tuple[float, float]] = []
    for i, (x, y, p) in enumerate(raw_stitches):
        if i == 0:
            result.append((x, y))
            continue

        px0, py0, p0 = raw_stitches[i - 1]

        if _line_exits_region(px0, py0, x, y):
            # パスに沿って細かいステッチを挿入
            sub_step = 0.3  # mm
            sub_pos = p0 + sub_step
            while sub_pos < p - 0.05:
                sx, sy = _interpolate_path(pts, cum_lengths, sub_pos)
                result.append((float(sx), float(sy)))
                sub_pos += sub_step

        result.append((x, y))

    return result


def _interpolate_path(pts: np.ndarray,
                      cum_lengths: np.ndarray,
                      pos: float) -> Tuple[float, float]:
    """Interpolate a point at given distance along path."""
    pos = np.clip(pos, 0, cum_lengths[-1])
    idx = np.searchsorted(cum_lengths, pos, side='right') - 1
    idx = np.clip(idx, 0, len(pts) - 2)

    seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
    if seg_len > 0:
        t = (pos - cum_lengths[idx]) / seg_len
    else:
        t = 0

    x = pts[idx, 0] + t * (pts[idx + 1, 0] - pts[idx, 0])
    y = pts[idx, 1] + t * (pts[idx + 1, 1] - pts[idx, 1])
    return float(x), float(y)
