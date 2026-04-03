"""Color quantization for embroidery thread mapping.

Reduces image colors to a limited palette matching real embroidery threads.
Uses LAB color space for perceptually accurate color matching.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from typing import List, Tuple


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference between lab1 and lab2.

    Args:
        lab1: Array of shape (..., 3) in LAB color space.
        lab2: Array of shape (..., 3) in LAB color space.

    Returns:
        Array of ΔE2000 values with shape (...).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_mean = (C1 + C2) / 2.0
    C_mean7 = C_mean ** 7
    G = 0.5 * (1.0 - np.sqrt(C_mean7 / (C_mean7 + 25.0 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    both_nonzero = (C1p * C2p) > 0
    hdiff = h2p - h1p
    dhp = np.where(~both_nonzero, 0.0,
          np.where(np.abs(hdiff) <= 180.0, hdiff,
          np.where(hdiff > 180.0, hdiff - 360.0, hdiff + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbarp = np.where(~both_nonzero, hsum,
            np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
            np.where(hsum < 360.0, (hsum + 360.0) / 2.0,
                     (hsum - 360.0) / 2.0)))

    T = (1.0
         - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))

    SL = 1.0 + 0.015 * (Lbarp - 50.0) ** 2 / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T

    Cbarp7 = Cbarp ** 7
    RC = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    dtheta = 30.0 * np.exp(-((hbarp - 275.0) / 25.0) ** 2)
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    return np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to CIELAB color space via XYZ.

    Args:
        rgb: Array of shape (..., 3) with values in [0, 255].

    Returns:
        Array of same shape in LAB color space.
    """
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # Linearize sRGB
    mask = rgb_norm > 0.04045
    rgb_lin = np.where(mask,
                       ((rgb_norm + 0.055) / 1.055) ** 2.4,
                       rgb_norm / 12.92)

    # RGB to XYZ (D65 illuminant)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_lin @ mat.T

    # XYZ to LAB
    ref = np.array([0.95047, 1.00000, 1.08883])  # D65
    xyz_n = xyz / ref

    epsilon = 0.008856
    kappa = 903.3
    mask = xyz_n > epsilon
    f = np.where(mask, np.cbrt(xyz_n), (kappa * xyz_n + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def quantize_colors(image: np.ndarray, n_colors: int,
                    use_thread_palette: bool = True,
                    custom_palette: List = None,
                    merge_close: bool = True,
                    fg_mask: 'np.ndarray | None' = None,
                    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Quantize image colors for embroidery.

    Args:
        image: RGB image array of shape (H, W, 3).
        n_colors: Target number of colors.
        use_thread_palette: If True, snap to nearest thread colors.
        custom_palette: Optional list of (R, G, B, Name) tuples.
        merge_close: If True, merge near-identical clusters.
        fg_mask: Optional (H, W) boolean mask. True = foreground pixel.
                 When provided, only foreground pixels are used for
                 K-means fitting so transparent pixels don't create
                 spurious clusters.

    Returns:
        Tuple of (label_map, palette):
            label_map: (H, W) array of color indices.
            palette: List of (R, G, B) tuples for each color.
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float64)

    # Convert to LAB for perceptually uniform clustering
    pixels_lab = rgb_to_lab(pixels)

    # Use custom palette if provided; fall back to generic CSV palette.
    if custom_palette:
        thread_pal = custom_palette
    else:
        from .palettes import get_palette
        thread_pal = [(r, g, b, n) for r, g, b, n, *_ in get_palette("generic")]

    # K-means clustering in LAB space
    n_colors = min(n_colors, len(thread_pal) if use_thread_palette else 256)
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,
    )

    if fg_mask is not None:
        # Fit only on opaque (foreground) pixels to avoid transparent
        # pixels (often black) creating a spurious cluster.
        fg_flat = fg_mask.reshape(-1)
        fg_pixels_lab = pixels_lab[fg_flat]
        kmeans.batch_size = min(10000, max(1, len(fg_pixels_lab)))
        kmeans.fit(fg_pixels_lab)
        labels = kmeans.predict(pixels_lab)
    else:
        kmeans.batch_size = min(10000, len(pixels_lab))
        labels = kmeans.fit_predict(pixels_lab)

    # Get cluster centers in RGB.
    # Use only foreground pixels when a mask is provided so that
    # transparent (typically black) background pixels do not bias the
    # cluster center toward black and corrupt palette matching.
    center_rgbs = []
    for label_idx in range(n_colors):
        if fg_mask is not None:
            mask = fg_flat & (labels == label_idx)
        else:
            mask = labels == label_idx
        if mask.any():
            mean_rgb = pixels[mask].mean(axis=0)
            clamped = np.clip(mean_rgb, 0, 255).astype(int)
            center_rgbs.append((int(clamped[0]), int(clamped[1]),
                                int(clamped[2])))
        else:
            center_rgbs.append((128, 128, 128))

    # Snap to nearest thread palette colors.
    # Use globally-optimal greedy assignment (sort all cluster-palette
    # pairs by ΔE2000 and assign in order) instead of sequential
    # first-come-first-served, so that no cluster is forced to a far
    # palette color simply because it was processed after another cluster
    # that claimed the same nearest entry.
    if use_thread_palette:
        palette_rgb = np.array([t[:3] for t in thread_pal], dtype=np.float64)
        palette_lab = rgb_to_lab(palette_rgb)
        n_clusters = len(center_rgbs)
        n_palette = len(thread_pal)

        # Full ΔE2000 cost matrix: shape (n_clusters, n_palette)
        cost_matrix = np.zeros((n_clusters, n_palette))
        for ci, rgb in enumerate(center_rgbs):
            center_lab = rgb_to_lab(
                np.array([[rgb]], dtype=np.float64))[0, 0]
            cost_matrix[ci] = delta_e_2000(palette_lab, center_lab)

        nearest_dists = cost_matrix.min(axis=1)
        nearest_indices = cost_matrix.argmin(axis=1)

        # Greedy assignment: process all pairs in distance order.
        # This ensures the globally closest cluster-palette pair is
        # satisfied before resolving conflicts.
        pairs = sorted(
            ((cost_matrix[ci, pi], ci, pi)
             for ci in range(n_clusters) for pi in range(n_palette))
        )
        assigned: dict[int, int] = {}   # cluster_idx → palette_idx
        used_palette: set[int] = set()

        for _d, ci, pi in pairs:
            if ci in assigned:
                continue
            if pi not in used_palette:
                assigned[ci] = pi
                used_palette.add(pi)
            if len(assigned) == n_clusters:
                break

        # Fallback for any cluster still unassigned (palette exhausted)
        for ci in range(n_clusters):
            if ci not in assigned:
                assigned[ci] = int(nearest_indices[ci])

        # If the assigned color is much farther than the unconstrained
        # nearest, allow a duplicate so the merge step can collapse them.
        for ci in range(n_clusters):
            pi = assigned[ci]
            if cost_matrix[ci, pi] > nearest_dists[ci] * 2.5:
                assigned[ci] = int(nearest_indices[ci])

        snapped_palette = []
        for ci in range(n_clusters):
            rgb = thread_pal[assigned[ci]][:3]
            snapped_palette.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))

        final_palette = snapped_palette
    else:
        final_palette = center_rgbs

    label_map = labels.reshape(h, w)

    # Merge perceptually close clusters to prevent fragmentation.
    # K-means may split anti-aliased edges or gradients into separate
    # clusters that are too close in LAB space to be distinguishable
    # as separate thread colors.
    if merge_close:
        label_map, final_palette = merge_close_clusters(
            label_map, final_palette, min_delta_e=10.0,
        )

    return label_map, final_palette


def merge_close_clusters(
    label_map: np.ndarray,
    palette: List[Tuple[int, int, int]],
    min_delta_e: float = 10.0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Merge clusters whose CIEDE2000 distance is below threshold.

    Iteratively finds the closest pair of colors and merges them
    (reassigning the smaller cluster to the larger one) until all
    remaining pairs exceed min_delta_e.

    Args:
        label_map: (H, W) integer label array.
        palette: List of (R, G, B) tuples, one per label.
        min_delta_e: Minimum ΔE2000 between any two colors to keep
            them as separate thread colors.

    Returns:
        (merged_label_map, merged_palette) with renumbered labels.
    """
    if len(palette) <= 1:
        return label_map, palette

    # Compute LAB for all palette colors
    palette_rgb = np.array(palette, dtype=np.float64)
    palette_lab = rgb_to_lab(palette_rgb)

    # Count pixels per label for merge direction (small → large)
    n = len(palette)
    counts = np.array([(label_map == i).sum() for i in range(n)])

    # Build merge mapping: label_id → canonical label_id
    canonical = list(range(n))

    def find(x):
        while canonical[x] != x:
            canonical[x] = canonical[canonical[x]]
            x = canonical[x]
        return x

    # Iteratively merge closest pair
    active = set(range(n))
    while len(active) > 1:
        # Find closest pair among active labels
        best_dist = float('inf')
        best_pair = None
        active_list = sorted(active)
        for ii in range(len(active_list)):
            for jj in range(ii + 1, len(active_list)):
                a, b = active_list[ii], active_list[jj]
                d = float(delta_e_2000(palette_lab[a], palette_lab[b]))
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)

        if best_dist >= min_delta_e:
            break

        # Merge smaller into larger
        a, b = best_pair
        if counts[a] >= counts[b]:
            keep, drop = a, b
        else:
            keep, drop = b, a

        # Don't absorb a meaningful cluster into one covering >30%
        # of the image (e.g. skin tones into white background).
        total_pixels = counts.sum()
        if (counts[keep] / total_pixels > 0.3
                and counts[drop] / total_pixels > 0.01):
            break

        canonical[drop] = keep
        counts[keep] += counts[drop]
        counts[drop] = 0

        active.discard(drop)

    # Build final mapping
    old_to_new = {}
    new_palette = []
    for idx in sorted(active):
        old_to_new[idx] = len(new_palette)
        new_palette.append(palette[idx])

    # Remap labels
    remap = np.zeros(n, dtype=np.int32)
    for i in range(n):
        root = find(i)
        remap[i] = old_to_new[root]

    merged_map = remap[label_map]
    return merged_map, new_palette


def dissolve_boundary_artifacts(label_map: np.ndarray) -> np.ndarray:
    """Dissolve pixels that are sandwiched between two different colors.

    Anti-aliasing and compression create intermediate-color pixels at
    the boundary between two regions. These pixels are adjacent to at
    least two *different* other labels — they are "sandwiched."

    Thin features on a uniform background (e.g., a black stroke on
    white) touch only ONE other label and are preserved.

    This distinction is purely geometric and works for all image types.

    Args:
        label_map: (H, W) integer label array.

    Returns:
        Label map with sandwiched boundary pixels reassigned.
    """
    import cv2

    n = label_map.max() + 1
    h, w = label_map.shape
    kernel = np.ones((3, 3), np.uint8)

    # Count how many distinct OTHER labels each pixel is adjacent to
    n_other_labels = np.zeros((h, w), dtype=np.int32)
    for c in range(n):
        mask_c = (label_map == c).astype(np.uint8)
        dilated = cv2.dilate(mask_c, kernel)
        # Pixels adjacent to label c but not belonging to label c
        adjacent = (dilated > 0) & (mask_c == 0)
        n_other_labels += adjacent.astype(np.int32)

    # Sandwiched: touching >= 2 different other labels
    sandwiched = n_other_labels >= 2

    # Reassign sandwiched pixels via local majority voting
    smoothed = smooth_label_map(label_map, kernel_size=5)
    result = label_map.copy()
    result[sandwiched] = smoothed[sandwiched]

    return result


def smooth_label_map(label_map: np.ndarray,
                     kernel_size: int = 5) -> np.ndarray:
    """Smooth label map boundaries using majority voting.

    Reduces jagged edges between color regions by assigning each
    pixel to the most prevalent color in its neighborhood.

    Args:
        label_map: (H, W) array of color labels.
        kernel_size: Size of the voting window.

    Returns:
        Smoothed label map.
    """
    import cv2

    n_colors = label_map.max() + 1
    h, w = label_map.shape

    smoothed = np.zeros_like(label_map)
    best_score = np.full((h, w), -1.0, dtype=np.float32)

    for c in range(n_colors):
        mask = (label_map == c).astype(np.float32)
        score = cv2.blur(mask, (kernel_size, kernel_size))
        better = score > best_score
        smoothed[better] = c
        best_score[better] = score[better]

    return smoothed


def remove_small_regions(label_map: np.ndarray,
                         min_area_ratio: float = 0.002) -> np.ndarray:
    """Remove small color regions by merging into neighbors.

    Args:
        label_map: (H, W) array of color labels.
        min_area_ratio: Minimum region area as fraction of total.

    Returns:
        Cleaned label map.
    """
    import cv2

    h, w = label_map.shape
    total_pixels = h * w
    min_pixels = int(total_pixels * min_area_ratio)
    result = label_map.copy()
    n_colors = label_map.max() + 1

    for color in range(n_colors):
        mask = (result == color).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_pixels:
                # Find neighboring color to merge into
                region_mask = labels == i
                # Dilate to find neighbors
                dilated = cv2.dilate(region_mask.astype(np.uint8),
                                     np.ones((3, 3), np.uint8))
                border = (dilated > 0) & ~region_mask
                if border.any():
                    neighbor_colors = result[border]
                    neighbor_colors = neighbor_colors[neighbor_colors != color]
                    if len(neighbor_colors) > 0:
                        # Use most frequent neighbor
                        vals, counts = np.unique(neighbor_colors,
                                                 return_counts=True)
                        result[region_mask] = vals[counts.argmax()]

    return result
