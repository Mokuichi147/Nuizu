"""Color quantization for embroidery thread mapping.

Reduces image colors to a limited palette matching real embroidery threads.
Uses LAB color space for perceptually accurate color matching.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from typing import List, Tuple

# Common embroidery thread palette (subset of popular thread colors)
# Format: (R, G, B, Name)
THREAD_PALETTE = [
    (0, 0, 0, "Black"),
    (255, 255, 255, "White"),
    (200, 0, 0, "Red"),
    (0, 100, 0, "Dark Green"),
    (0, 0, 180, "Blue"),
    (255, 200, 0, "Yellow"),
    (255, 127, 0, "Orange"),
    (128, 0, 128, "Purple"),
    (255, 105, 180, "Pink"),
    (139, 69, 19, "Brown"),
    (128, 128, 128, "Gray"),
    (192, 192, 192, "Light Gray"),
    (0, 128, 128, "Teal"),
    (0, 200, 0, "Green"),
    (135, 206, 235, "Sky Blue"),
    (0, 0, 100, "Navy"),
    (178, 34, 34, "Dark Red"),
    (255, 215, 0, "Gold"),
    (245, 222, 179, "Wheat"),
    (210, 180, 140, "Tan"),
    (255, 160, 122, "Light Salmon"),
    (144, 238, 144, "Light Green"),
    (230, 230, 250, "Lavender"),
    (255, 228, 196, "Bisque"),
    (64, 64, 64, "Dark Gray"),
    (160, 82, 45, "Sienna"),
    (205, 133, 63, "Peru"),
    (107, 142, 35, "Olive Green"),
    (70, 130, 180, "Steel Blue"),
    (220, 20, 60, "Crimson"),
    (148, 103, 189, "Medium Purple"),
    (255, 69, 0, "Orange Red"),
]


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

    # Use custom palette if provided
    thread_pal = custom_palette if custom_palette else THREAD_PALETTE

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

    # Get cluster centers in RGB
    center_rgbs = []
    for label_idx in range(n_colors):
        mask = labels == label_idx
        if mask.any():
            mean_rgb = pixels[mask].mean(axis=0)
            clamped = np.clip(mean_rgb, 0, 255).astype(int)
            center_rgbs.append((int(clamped[0]), int(clamped[1]),
                                int(clamped[2])))
        else:
            center_rgbs.append((128, 128, 128))

    # Snap to nearest thread palette colors
    if use_thread_palette:
        palette_rgb = np.array([t[:3] for t in thread_pal], dtype=np.float64)
        palette_lab = rgb_to_lab(palette_rgb)
        snapped_palette = []
        used_indices = set()

        for rgb in center_rgbs:
            center_lab = rgb_to_lab(np.array([[rgb]], dtype=np.float64))[0, 0]
            dists = np.sqrt(np.sum((palette_lab - center_lab) ** 2, axis=1))
            sorted_idx = np.argsort(dists)
            for idx in sorted_idx:
                if idx not in used_indices or len(used_indices) >= len(thread_pal):
                    chosen = idx
                    used_indices.add(idx)
                    break
            rgb = thread_pal[chosen][:3]
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
            label_map, final_palette, min_delta_e=15.0,
        )

    return label_map, final_palette


def merge_close_clusters(
    label_map: np.ndarray,
    palette: List[Tuple[int, int, int]],
    min_delta_e: float = 15.0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Merge clusters whose LAB distance is below threshold.

    Iteratively finds the closest pair of colors and merges them
    (reassigning the smaller cluster to the larger one) until all
    remaining pairs exceed min_delta_e.

    Args:
        label_map: (H, W) integer label array.
        palette: List of (R, G, B) tuples, one per label.
        min_delta_e: Minimum Delta-E (LAB Euclidean) between any
            two colors to keep them as separate thread colors.

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
                d = np.sqrt(np.sum((palette_lab[a] - palette_lab[b]) ** 2))
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
