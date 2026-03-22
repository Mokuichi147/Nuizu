"""Main conversion pipeline: photo → embroidery pattern.

Orchestrates preprocessing, color quantization, region segmentation,
stitch generation, optimization, and pattern assembly.

v0.2: Added pull compensation, auto fill angle, enhanced preprocessing,
      brand palettes, SVG preview.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

from .preprocess import preprocess_photo, detect_background, auto_crop_to_subject
from .quantize import quantize_colors, remove_small_regions
from .segment import extract_regions, scale_regions_to_mm, Region, _chaikin_smooth
from .fill import generate_fill_stitch_segments, generate_outline_stitches
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


MONOCHROME_THREAD_PALETTE = [
    (0, 0, 0, "Black"),
    (64, 64, 64, "Dark Gray"),
    (128, 128, 128, "Gray"),
    (192, 192, 192, "Light Gray"),
    (230, 230, 230, "Off White"),
    (255, 255, 255, "White"),
]


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
    # Thread
    thread_width: float = 0.4,
    # Compensation
    pull_compensation: float = 0.0,
    # Processing
    blur_radius: int = 3,
    min_region_ratio: float = 0.003,
    enhance_photo: bool = True,
    auto_crop: bool = False,
    skip_background: bool = False,
    strict_colors: bool = False,
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
        strict_colors: Keep exactly n_colors in the output palette.
        verbose: Print progress messages.

    Returns:
        The generated EmbroideryPattern.
    """
    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    requested_n_colors = n_colors

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

    style = _analyze_image_style(img_rgb)
    line_art_mode = style["is_line_art"]
    monochrome_mode = style["is_monochrome"]

    if line_art_mode:
        log("  Line-art mode detected: preserving fine strokes")
        if enhance_photo:
            enhance_photo = False
            log("  Line-art mode: disabled enhancement pipeline")
        if blur_radius > 0:
            blur_radius = 0
            log("  Line-art mode: blur disabled")
        if min_region_ratio >= 0.0005:
            min_region_ratio = 0.00001
            log("  Line-art mode: min-region set to 0.00001")
        if n_colors >= 8 and not strict_colors:
            n_colors = 6
            log("  Line-art mode: colors reduced to 6")
        if underlay:
            underlay = False
            log("  Line-art mode: underlay disabled")
        if fill_density > 0.4:
            fill_density = 0.4
            log("  Line-art mode: fill density set to 0.4mm")
        if not auto_angle:
            auto_angle = True
            log("  Line-art mode: auto-angle enabled")

    # Detect background
    bg_color = None
    if skip_background:
        bg_color = detect_background(img_rgb, method='corner')
        if bg_color is None:
            bg_color = detect_background(img_rgb, method='edge')
        if bg_color:
            log(f"  Background detected: RGB{bg_color}")
    elif not strict_colors:
        # Auto-detect bright background for any image type.
        # A near-white background wastes a color slot and causes
        # light-colored subjects (hair, skin) to merge into it.
        auto_bg = detect_background(img_rgb, method='corner')
        if auto_bg is None:
            auto_bg = detect_background(img_rgb, method='edge')
        if auto_bg is not None and min(auto_bg) >= 220:
            skip_background = True
            bg_color = auto_bg
            log(f"  Auto-skip bright background RGB{bg_color}")

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

    # Median filter to remove anti-aliasing and JPEG artifacts.
    # Median preserves edges while replacing isolated intermediate-color
    # pixels with the dominant neighbor color, preventing K-means from
    # creating spurious clusters for transition pixels.
    img_rgb = cv2.medianBlur(img_rgb, 3)

    # 3. Color quantization
    log(f"[3/7] Quantizing to {n_colors} colors (palette: {thread_brand})...")
    custom_pal = None
    if use_thread_palette:
        if line_art_mode and monochrome_mode:
            custom_pal = MONOCHROME_THREAD_PALETTE
            log("  Line-art mode: using monochrome palette")
        else:
            custom_pal = get_palette(thread_brand)

    max_palette_colors = len(custom_pal) if use_thread_palette else 256
    if strict_colors and requested_n_colors > max_palette_colors:
        raise ValueError(
            "strict-colorsでは指定色数が上限を超えています: "
            f"{requested_n_colors} > {max_palette_colors}"
        )

    label_map, palette = quantize_colors(
        img_rgb, n_colors, use_thread_palette,
        custom_palette=custom_pal,
        merge_close=not strict_colors,
    )
    log(f"  Palette: {len(palette)} colors")

    if strict_colors and len(palette) != requested_n_colors:
        raise RuntimeError(
            "strict-colors有効時に指定色数を確保できませんでした: "
            f"{len(palette)} / {requested_n_colors}"
        )

    # 4. Region segmentation
    log("[4/7] Segmenting regions...")
    # Compute min area from physical stitch size: regions below ~1mm²
    # cannot be meaningfully stitched regardless of image resolution.
    px_per_mm = max(h, w) / max(target_width_mm, target_height_mm)
    min_area = max(1, int(h * w * min_region_ratio))
    regions = extract_regions(
        label_map,
        palette,
        min_area=min_area,
        morph_cleanup=not line_art_mode,
        simplify_epsilon_min=0.25 if line_art_mode else 0.5,
    )

    # Filter out background regions.
    # Only skip the single palette color closest to the detected
    # background — never distance-threshold, which would also
    # discard light foreground colors (e.g., light hair, pale skin).
    if skip_background and bg_color:
        best_idx = None
        best_dist = float('inf')
        for idx, pc in enumerate(palette):
            d = _color_distance(pc, bg_color)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        bg_palette_color = palette[best_idx] if best_idx is not None else None
        before = len(regions)
        if bg_palette_color is not None:
            regions = [r for r in regions
                       if r.color_rgb != bg_palette_color]
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

    if strict_colors:
        # Ensure exact palette count even when segmentation drops tiny regions.
        for rgb in palette:
            if len(color_set) >= requested_n_colors:
                break
            if rgb not in color_set:
                color_set[rgb] = ThreadColor(
                    r=rgb[0], g=rgb[1], b=rgb[2],
                    name=f"Color {len(color_set) + 1}",
                )

        if len(color_set) < requested_n_colors and custom_pal is not None:
            for entry in custom_pal:
                rgb = tuple(entry[:3])
                if rgb in color_set:
                    continue
                color_set[rgb] = ThreadColor(
                    r=rgb[0], g=rgb[1], b=rgb[2],
                    name=f"Color {len(color_set) + 1}",
                )
                if len(color_set) >= requested_n_colors:
                    break

        if len(color_set) != requested_n_colors:
            raise RuntimeError(
                "strict-colors有効時に最終色数を一致させられませんでした: "
                f"{len(color_set)} / {requested_n_colors}"
            )

    pattern.colors = list(color_set.values())

    current_color = None
    total_regions = len(regions)
    comp_mm = pull_compensation
    line_art_fill_skipped = 0

    # Compute a single fill angle from the largest region overall.
    # All regions use the same angle for uniform appearance.
    global_angle = fill_angle
    if auto_angle:
        largest_region = None
        for region in regions:
            if region.contour_mm is None:
                continue
            if largest_region is None or region.area > largest_region.area:
                largest_region = region
        if largest_region is not None:
            global_angle = compute_optimal_fill_angle(
                largest_region.contour_mm, fill_angle
            )

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

        region_angle = global_angle

        # Use raw (non-simplified) contour for fill to avoid gaps
        # from polygon simplification. Simplified contour is for outline.
        fill_contour = region.contour_raw_mm if region.contour_raw_mm is not None \
            else region.contour_mm
        fill_holes = region.holes_raw_mm if region.holes_raw_mm else \
            (region.holes_mm if region.holes_mm else [])
        contour = region.contour_mm
        holes = region.holes_mm if region.holes_mm else []

        if comp_mm > 0:
            contour = apply_pull_compensation(contour, comp_mm)
            fill_contour = apply_pull_compensation(fill_contour, comp_mm)

        # For line-art style, skip fills only on very bright regions.
        # Dark/mid-tone regions (shading, hair, shadows) always get fill.
        region_luma = sum(region.color_rgb) / 3.0
        skip_fill = (
            line_art_mode
            and monochrome_mode
            and region_luma >= 200.0
        )

        effective_spacing = fill_density

        # Skip fill for regions too thin to hold even 2 fill rows.
        # These get outline only, avoiding broken/dashed fill artifacts.
        if not skip_fill and fill_contour is not None and len(fill_contour) >= 5:
            pts = fill_contour.astype(np.float32).reshape(-1, 1, 2)
            rect = cv2.minAreaRect(pts)
            min_dim = min(rect[1][0], rect[1][1])
            if min_dim < effective_spacing * 2:
                skip_fill = True

        if skip_fill:
            line_art_fill_skipped += 1
            fill_segments = []
        else:
            fill_segments = generate_fill_stitch_segments(
                contour=fill_contour,
                holes=fill_holes,
                fill_angle=region_angle,
                row_spacing=effective_spacing,
                stitch_length=stitch_length,
                underlay=underlay,
            )

            # Apply fill compensation
            if comp_mm > 0 and fill_segments:
                compensated_segments = []
                for segment in fill_segments:
                    compensated = apply_fill_compensation(
                        segment, contour, comp_mm, region_angle
                    )
                    if compensated:
                        compensated_segments.append(compensated)
                fill_segments = compensated_segments

        for fill_segment in fill_segments:
            if not fill_segment:
                continue

            # Split long stitches
            fill_segment = split_long_stitches(fill_segment, max_length=7.0)
            if not fill_segment:
                continue

            # Jump to first stitch of each disconnected span
            if pattern.stitches:
                pattern.add_stitch(
                    fill_segment[0][0], fill_segment[0][1], StitchType.JUMP
                )

            # Add fill stitches
            for x, y in fill_segment:
                pattern.add_stitch(x, y, StitchType.NORMAL)

        # Outline stitches (smooth contour for curves)
        if outline and region.contour_mm is not None:
            # Skip smoothing for very simple polygons (rectangles, triangles)
            # to avoid rounding intentionally sharp corners
            if len(region.contour_mm) >= 5:
                outline_contour = _chaikin_smooth(region.contour_mm, iterations=2)
            else:
                outline_contour = region.contour_mm
            outline_stitch_len = 1.2 if line_art_mode else 1.5
            outline_pts = generate_outline_stitches(
                contour=outline_contour,
                stitch_length=outline_stitch_len,
                satin_width=1.2,
                satin=outline_satin,
            )
            if outline_pts:
                outline_pts = split_long_stitches(outline_pts, max_length=7.0)
                pattern.add_stitch(outline_pts[0][0], outline_pts[0][1],
                                   StitchType.JUMP)
                for x, y in outline_pts:
                    pattern.add_stitch(x, y, StitchType.NORMAL)

    if line_art_mode and line_art_fill_skipped > 0:
        log(
            "  Line-art mode: skipped fill on "
            f"{line_art_fill_skipped} bright/large regions"
        )

    if line_art_mode:
        detail_contours = _extract_small_ink_contours(img_rgb)
        detail_added = 0
        if detail_contours:
            black_rgb = (0, 0, 0)
            detail_rgb = black_rgb
            if strict_colors and detail_rgb not in color_set and color_set:
                # Keep color count fixed: reuse the darkest existing color.
                detail_rgb = min(color_set.keys(), key=lambda rgb: sum(rgb))

            if detail_rgb not in color_set:
                color_set[detail_rgb] = ThreadColor(
                    r=detail_rgb[0], g=detail_rgb[1], b=detail_rgb[2],
                    name=f"Color {len(color_set) + 1}",
                )
                pattern.colors = list(color_set.values())

            if current_color != detail_rgb:
                if current_color is not None:
                    pattern.add_trim()
                    pattern.add_color_change()
                current_color = detail_rgb

            scale_x = target_width_mm / w
            scale_y = target_height_mm / h

            for contour_px in detail_contours:
                contour_mm = contour_px.copy()
                contour_mm[:, 0] *= scale_x
                contour_mm[:, 1] *= scale_y

                detail_pts = generate_outline_stitches(
                    contour=contour_mm,
                    stitch_length=0.9,
                    satin=False,
                )
                if len(detail_pts) < 2:
                    continue

                if pattern.stitches:
                    pattern.add_stitch(
                        detail_pts[0][0], detail_pts[0][1], StitchType.JUMP
                    )
                for x, y in detail_pts:
                    pattern.add_stitch(x, y, StitchType.NORMAL)
                detail_added += 1

        if detail_added > 0:
            log(f"  Line-art mode: reinforced {detail_added} fine contours")

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
                     bg_color: Tuple[int, int, int] = (240, 235, 220),
                     thread_width_mm: float = 0.4):
    """Generate a raster preview image of the embroidery pattern.

    Args:
        pattern: Embroidery pattern to preview.
        output_path: Output image path.
        width: Canvas width in pixels.
        bg_color: Background color.
        thread_width_mm: Thread width in mm for line thickness.
            Use 0.4 for realistic thread preview, 0 for 1px stitch lines.
    """
    bounds = pattern.get_bounds()
    pat_w = bounds[2] - bounds[0]
    pat_h = bounds[3] - bounds[1]

    if pat_w <= 0 or pat_h <= 0:
        return

    scale = (width - 40) / pat_w
    height = int(pat_h * scale) + 40

    if thread_width_mm > 0:
        thread_px = max(1, round(thread_width_mm * scale))
    else:
        thread_px = 1

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

            cv2.line(img, (prev_x, prev_y), (px, py), bgr, thread_px,
                     cv2.LINE_AA)

        prev_x, prev_y = px, py

    cv2.imwrite(output_path, img)


def _analyze_image_style(image_rgb: np.ndarray) -> dict:
    """Analyze image characteristics to detect line-art inputs."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    saturation = hsv[:, :, 1]
    low_sat_ratio = float(np.mean(saturation < 30))

    edges = cv2.Canny(gray, 60, 180)
    edge_ratio = float(np.mean(edges > 0))
    bright_ratio = float(np.mean(gray > 200))
    dark_ratio = float(np.mean(gray < 60))

    is_monochrome = low_sat_ratio > 0.95
    is_line_art = (
        is_monochrome
        and edge_ratio > 0.03
        and bright_ratio > 0.25
        and dark_ratio > 0.01
    )

    return {
        "is_monochrome": is_monochrome,
        "is_line_art": is_line_art,
    }


def _extract_small_ink_contours(
    image_rgb: np.ndarray,
    min_area_px: float = 1.0,
    max_area_px: float = 600.0,
) -> List[np.ndarray]:
    """Extract small dark contours to reinforce missing fine details."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, ink_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        ink_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    detail_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px or area > max_area_px:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 4.0:
            continue

        epsilon = max(0.15, 0.0015 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2).astype(np.float64)
        if len(points) < 2:
            continue

        detail_contours.append(points)

    return detail_contours


def _color_distance(c1: Tuple[int, int, int],
                    c2: Tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
