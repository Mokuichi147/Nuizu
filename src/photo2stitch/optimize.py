"""Stitch order optimization.

Optimizes the order of regions to minimize jump stitches,
trim operations, and color changes.
"""

import numpy as np
from typing import List, Tuple, Dict
from .segment import Region


def optimize_stitch_order(regions: List[Region]) -> List[Region]:
    """Optimize order of regions to minimize color changes and jumps.

    Groups regions by color, then orders within each color group
    using nearest-neighbor to minimize jump distances.
    """
    if not regions:
        return []

    # Group by color
    color_groups: Dict[Tuple[int, int, int], List[Region]] = {}
    for r in regions:
        key = r.color_rgb
        if key not in color_groups:
            color_groups[key] = []
        color_groups[key].append(r)

    # Order color groups: largest total area first
    sorted_colors = sorted(
        color_groups.keys(),
        key=lambda c: sum(r.area for r in color_groups[c]),
        reverse=True,
    )

    ordered = []
    last_pos = (0.0, 0.0)

    for color in sorted_colors:
        group = color_groups[color]

        # Nearest-neighbor ordering within color group
        remaining = list(group)
        while remaining:
            best_idx = 0
            best_dist = float('inf')
            for i, r in enumerate(remaining):
                dist = ((r.centroid[0] - last_pos[0]) ** 2 +
                        (r.centroid[1] - last_pos[1]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            last_pos = chosen.centroid

    return ordered


def split_long_stitches(stitches: List[Tuple[float, float]],
                        max_length: float = 7.0
                        ) -> List[Tuple[float, float]]:
    """Split stitches longer than max_length into shorter segments.

    Most embroidery machines have a maximum stitch length of ~12mm.
    Keeping stitches under 7mm produces better results.
    """
    if len(stitches) < 2:
        return stitches

    result = [stitches[0]]

    for i in range(1, len(stitches)):
        x0, y0 = result[-1]
        x1, y1 = stitches[i]
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > max_length:
            n_segments = int(np.ceil(dist / max_length))
            for j in range(1, n_segments):
                t = j / n_segments
                result.append((x0 + t * dx, y0 + t * dy))

        result.append((x1, y1))

    return result
