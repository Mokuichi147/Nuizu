"""Main conversion pipeline: photo → embroidery pattern.

Orchestrates preprocessing, color quantization, region segmentation,
stitch generation, optimization, and pattern assembly.

v0.2: Added pull compensation, auto fill angle, enhanced preprocessing,
      brand palettes, SVG preview.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import sys
import os

from .preprocess import preprocess_photo, detect_background, auto_crop_to_subject
from .quantize import quantize_colors, remove_small_regions
from .segment import extract_regions, scale_regions_to_mm, Region
from .fill import generate_fill_stitches, generate_outline_stitches
from .optimize import optimize_stitch_order, split_long_stitches
from .compensate import apply_pull_compensation, apply_fill_compensation
from .auto_angle import (
    compute_optimal_fill_angle,
    compute_angle_for_thin_region,
)
from .palettes import get_palette
from .svg_preview import generate_svg_preview
from .formats.common import (
    EmbroideryPattern, ThreadColor, StitchType
)
from .formats.dst import write_dst
from .formats.pes import write_pes
from .formats.jef import write_jef


FORMAT_WRITERS = {
    '.dst': write_dst,
    '.pes': write_pes,
    '.jef': write_jef,
}


def convert_photo_to_embroidery(
    image_path: str,
    output_path: str,
    # Size
    target_width_mm: float = 100.0,
    target_height_mm: Optional[float] = None,
    # Colors
    n_colors: int = 8,
    use_thread_palette: bool = True,
    thread_brand: str = "janome",
    # Stitch parameters
    fill_density: float = 0.4,
    stitch_length: float = 3.0,
    fill_angle: float = 45.0,
    auto_angle: bool = False,
    underlay: bool = True,
    outline: bool = True,
    outline_satin: bool = False,
    # Compensation
    pull_compensation: float = 0.0,
    # Processing
    blur_radius: int = 3,
    min_region_ratio: float = 0.003,
    enhance_photo: bool = True,
    auto_crop: bool = False,
    skip_background: bool = False,
    # Output
    verbose: bool = True,
) -> EmbroideryPattern:
    """Convert a photo to an embroidery pattern.

    Args:
        image_path: Path to input image.
        output_path: Path to output embroidery file (.dst, .pes, .jef).
        target_width_mm: Target embroidery width in mm.
        target_height_mm: Target height (auto if None).
        n_colors: Number of thread colors.
        use_thread_palette: Snap to real thread colors.
        thread_brand: Thread brand for palette ('janome', 'brother', 'madeira').
        fill_density: Fill row spacing in mm.
        stitch_length: Maximum stitch length in mm.
        fill_angle: Base fill angle in degrees.
        auto_angle: Automatically optimize fill angle per region.
        underlay: Whether to add underlay stitches.
        outline: Whether to add outline stitches.
        outline_satin: Use satin stitch for outlines.
        pull_compensation: Pull compensation in mm (0=off).
        blur_radius: Gaussian blur radius for preprocessing.
        min_region_ratio: Minimum region area ratio.
        enhance_photo: Apply photo enhancement preprocessing.
        auto_crop: Auto-crop to subject.
        skip_background: Skip stitching detected background color.
        verbose: Print progress messages.

    Returns:
        The generated EmbroideryPattern.
    """
    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    # 1. Load and preprocess image
    log(f"[1/7] Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    log(f"  Original: {w}x{h}px")

    # Auto-crop
    if auto_crop:
        img_rgb, crop_box = auto_crop_to_subject(img_rgb)
        h, w = img_rgb.shape[:2]
        log(f"  Auto-cropped to: {w}x{h}px")

    # Detect background
    bg_color = None
    if skip_background:
        bg_color = detect_background(img_rgb)
        if bg_color:
            log(f"  Background detected: RGB{bg_color}")

    # Calculate dimensions
    if target_height_mm is None:
        target_height_mm = target_width_mm * (h / w)
    log(f"  Target: {target_width_mm:.0f}x{target_height_mm:.0f}mm")

    # 2. Enhanced preprocessing
    log("[2/7] Preprocessing...")
    if enhance_photo:
        img_rgb = preprocess_photo(
            img_rgb,
            max_dim=800,
            denoise=True,
            enhance_contrast=True,
            edge_smooth=True,
            sharpen_edges=True,
            saturation_boost=1.2,
        )
    else:
        # Basic resize + blur only
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_rgb = cv2.resize(img_rgb,
                                 (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        if blur_radius > 0:
            k = blur_radius * 2 + 1
            img_rgb = cv2.GaussianBlur(img_rgb, (k, k), 0)

    h, w = img_rgb.shape[:2]

    # 3. Color quantization
    log(f"[3/7] Quantizing to {n_colors} colors (palette: {thread_brand})...")
    custom_pal = None
    if use_thread_palette:
        custom_pal = get_palette(thread_brand)

    label_map, palette = quantize_colors(
        img_rgb, n_colors, use_thread_palette,
        custom_palette=custom_pal,
    )
    label_map = remove_small_regions(label_map, min_region_ratio)
    log(f"  Palette: {len(palette)} colors")

    # 4. Region segmentation
    log("[4/7] Segmenting regions...")
    min_area = int(h * w * min_region_ratio)
    regions = extract_regions(label_map, palette, min_area=min_area)

    # Filter out background regions
    if skip_background and bg_color:
        before = len(regions)
        regions = [r for r in regions
                   if _color_distance(r.color_rgb, bg_color) > 60]
        log(f"  Skipped {before - len(regions)} background regions")

    log(f"  Found {len(regions)} regions")

    # Scale to mm
    regions = scale_regions_to_mm(
        regions, (h, w), target_width_mm, target_height_mm
    )

    # 5. Optimize stitch order
    log("[5/7] Optimizing stitch order...")
    regions = optimize_stitch_order(regions)

    # 6. Generate stitches
    log("[6/7] Generating stitches...")
    pattern = EmbroideryPattern(name=os.path.splitext(
        os.path.basename(image_path))[0])

    # Build color map
    color_set = {}
    for region in regions:
        rgb = region.color_rgb
        if rgb not in color_set:
            color_set[rgb] = ThreadColor(
                r=rgb[0], g=rgb[1], b=rgb[2],
                name=f"Color {len(color_set) + 1}",
            )
    pattern.colors = list(color_set.values())

    current_color = None
    total_regions = len(regions)
    comp_mm = pull_compensation

    for i, region in enumerate(regions):
        if region.contour_mm is None:
            continue

        if i % max(1, total_regions // 10) == 0:
            log(f"  Processing region {i + 1}/{total_regions}...")

        # Color change
        if region.color_rgb != current_color:
            if current_color is not None:
                pattern.add_trim()
                pattern.add_color_change()
            current_color = region.color_rgb

        # Determine fill angle
        if auto_angle:
            thin_angle = compute_angle_for_thin_region(region.contour_mm)
            if thin_angle is not None:
                region_angle = thin_angle
            else:
                region_angle = compute_optimal_fill_angle(
                    region.contour_mm, fill_angle
                )
        else:
            region_angle = fill_angle

        # Apply pull compensation to contour
        contour = region.contour_mm
        holes = region.holes_mm if region.holes_mm else []

        if comp_mm > 0:
            contour = apply_pull_compensation(contour, comp_mm)

        # Generate fill stitches
        fill_pts = generate_fill_stitches(
            contour=contour,
            holes=holes,
            fill_angle=region_angle,
            row_spacing=fill_density,
            stitch_length=stitch_length,
            underlay=underlay,
        )

        if not fill_pts:
            continue

        # Apply fill compensation
        if comp_mm > 0:
            fill_pts = apply_fill_compensation(
                fill_pts, contour, comp_mm, region_angle
            )

        # Split long stitches
        fill_pts = split_long_stitches(fill_pts, max_length=7.0)

        # Jump to first stitch
        if pattern.stitches:
            pattern.add_stitch(fill_pts[0][0], fill_pts[0][1],
                               StitchType.JUMP)

        # Add fill stitches
        for x, y in fill_pts:
            pattern.add_stitch(x, y, StitchType.NORMAL)

        # Outline stitches
        if outline and region.contour_mm is not None:
            outline_pts = generate_outline_stitches(
                contour=region.contour_mm,
                stitch_length=2.5,
                satin_width=1.2,
                satin=outline_satin,
            )
            if outline_pts:
                outline_pts = split_long_stitches(outline_pts, max_length=7.0)
                pattern.add_stitch(outline_pts[0][0], outline_pts[0][1],
                                   StitchType.JUMP)
                for x, y in outline_pts:
                    pattern.add_stitch(x, y, StitchType.NORMAL)

    # End
    pattern.add_stitch(0, 0, StitchType.END)

    # 7. Write output
    log(f"[7/7] Writing output: {output_path}")

    ext = '.' + output_path.rsplit('.', 1)[-1].lower() if '.' in output_path \
        else '.dst'
    writer = FORMAT_WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(FORMAT_WRITERS.keys())}"
        )

    writer(pattern, output_path)

    log("")
    log(pattern.summary())
    log(f"\nOutput: {output_path}")

    return pattern


def generate_preview(pattern: EmbroideryPattern,
                     output_path: str,
                     width: int = 800,
                     bg_color: Tuple[int, int, int] = (240, 235, 220)):
    """Generate a raster preview image of the embroidery pattern."""
    bounds = pattern.get_bounds()
    pat_w = bounds[2] - bounds[0]
    pat_h = bounds[3] - bounds[1]

    if pat_w <= 0 or pat_h <= 0:
        return

    scale = (width - 40) / pat_w
    height = int(pat_h * scale) + 40

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    current_color_idx = 0
    colors = pattern.colors
    prev_x, prev_y = None, None

    for stitch in pattern.stitches:
        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            current_color_idx += 1
            prev_x, prev_y = None, None
            continue

        if stitch.stitch_type in (StitchType.JUMP, StitchType.TRIM):
            prev_x, prev_y = None, None
            continue

        if stitch.stitch_type == StitchType.END:
            break

        px = int((stitch.x - bounds[0]) * scale) + 20
        py = int((stitch.y - bounds[1]) * scale) + 20

        if prev_x is not None and prev_y is not None:
            color_idx = min(current_color_idx, len(colors) - 1)
            if colors:
                c = colors[color_idx]
                bgr = (c.b, c.g, c.r)
            else:
                bgr = (0, 0, 0)

            cv2.line(img, (prev_x, prev_y), (px, py), bgr, 1,
                     cv2.LINE_AA)

        prev_x, prev_y = px, py

    cv2.imwrite(output_path, img)


def _color_distance(c1: Tuple[int, int, int],
                    c2: Tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
