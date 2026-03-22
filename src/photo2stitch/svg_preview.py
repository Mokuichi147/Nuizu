"""SVG preview generator for embroidery patterns.

Generates a scalable vector preview showing individual stitches,
color regions, and pattern metadata. Unlike raster previews,
SVG can be zoomed to inspect individual stitches.
"""

import html
from typing import Tuple
from .formats.common import EmbroideryPattern, StitchType


def generate_svg_preview(
    pattern: EmbroideryPattern,
    output_path: str,
    canvas_width: int = 800,
    bg_color: str = "#f0ebe0",
    show_jumps: bool = False,
    thread_width_mm: float = 0.4,
    stitch_opacity: float = 0.85,
    fabric_texture: bool = True,
):
    """Generate an SVG preview of the embroidery pattern.

    Args:
        pattern: Embroidery pattern to preview.
        output_path: Output .svg file path.
        canvas_width: SVG canvas width in pixels.
        bg_color: Background (fabric) color.
        show_jumps: Show jump stitches as dashed lines.
        stitch_opacity: Opacity of stitch lines.
        fabric_texture: Add subtle fabric texture pattern.
    """
    bounds = pattern.get_bounds()
    pat_w = bounds[2] - bounds[0]
    pat_h = bounds[3] - bounds[1]

    if pat_w <= 0 or pat_h <= 0:
        return

    margin = 20
    scale = (canvas_width - 2 * margin) / pat_w
    canvas_height = int(pat_h * scale) + 2 * margin

    def tx(x: float) -> float:
        return (x - bounds[0]) * scale + margin

    def ty(y: float) -> float:
        return (y - bounds[1]) * scale + margin

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{canvas_width}" height="{canvas_height}" '
                 f'viewBox="0 0 {canvas_width} {canvas_height}">')

    # Definitions
    lines.append('<defs>')

    # Fabric texture pattern
    if fabric_texture:
        lines.append(
            '<pattern id="fabric" patternUnits="userSpaceOnUse" '
            'width="4" height="4">'
            '<rect width="4" height="4" fill="none"/>'
            '<line x1="0" y1="0" x2="4" y2="0" '
            f'stroke="{_darken(bg_color, 0.03)}" stroke-width="0.3"/>'
            '<line x1="0" y1="0" x2="0" y2="4" '
            f'stroke="{_darken(bg_color, 0.03)}" stroke-width="0.3"/>'
            '</pattern>'
        )

    # Stitch shadow filter
    lines.append(
        '<filter id="threadShadow" x="-2%" y="-2%" '
        'width="104%" height="104%">'
        '<feGaussianBlur in="SourceAlpha" stdDeviation="0.5"/>'
        '<feOffset dx="0.3" dy="0.3"/>'
        '<feComposite in="SourceGraphic"/>'
        '</filter>'
    )

    lines.append('</defs>')

    # Background
    lines.append(f'<rect width="{canvas_width}" height="{canvas_height}" '
                 f'fill="{bg_color}"/>')
    if fabric_texture:
        lines.append(f'<rect width="{canvas_width}" height="{canvas_height}" '
                     f'fill="url(#fabric)"/>')

    # Hoop outline (decorative)
    hoop_cx = canvas_width / 2
    hoop_cy = canvas_height / 2
    hoop_rx = (canvas_width - margin) / 2
    hoop_ry = (canvas_height - margin) / 2
    lines.append(
        f'<ellipse cx="{hoop_cx:.1f}" cy="{hoop_cy:.1f}" '
        f'rx="{hoop_rx:.1f}" ry="{hoop_ry:.1f}" '
        f'fill="none" stroke="#c0b090" stroke-width="3" '
        f'stroke-dasharray="8,4" opacity="0.4"/>'
    )

    # Draw stitches grouped by color
    current_color_idx = 0
    colors = pattern.colors

    prev_x, prev_y = None, None
    current_group_lines = []
    current_color_hex = "#000000"

    if colors:
        c = colors[0]
        current_color_hex = f"#{c.r:02x}{c.g:02x}{c.b:02x}"

    def flush_group():
        nonlocal current_group_lines
        if current_group_lines:
            lines.append(
                f'<g stroke="{current_color_hex}" '
                f'stroke-width="{max(0.5, thread_width_mm * scale):.2f}" '
                f'stroke-linecap="round" '
                f'opacity="{stitch_opacity}" '
                f'filter="url(#threadShadow)">'
            )
            lines.extend(current_group_lines)
            lines.append('</g>')
            current_group_lines = []

    for stitch in pattern.stitches:
        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            flush_group()
            current_color_idx += 1
            if current_color_idx < len(colors):
                c = colors[current_color_idx]
                current_color_hex = f"#{c.r:02x}{c.g:02x}{c.b:02x}"
            prev_x, prev_y = None, None
            continue

        if stitch.stitch_type == StitchType.TRIM:
            prev_x, prev_y = None, None
            continue

        if stitch.stitch_type == StitchType.END:
            break

        sx = tx(stitch.x)
        sy = ty(stitch.y)

        if stitch.stitch_type == StitchType.JUMP:
            if show_jumps and prev_x is not None:
                current_group_lines.append(
                    f'<line x1="{prev_x:.1f}" y1="{prev_y:.1f}" '
                    f'x2="{sx:.1f}" y2="{sy:.1f}" '
                    f'stroke-dasharray="2,2" opacity="0.2"/>'
                )
            prev_x, prev_y = sx, sy
            continue

        if prev_x is not None and prev_y is not None:
            current_group_lines.append(
                f'<line x1="{prev_x:.1f}" y1="{prev_y:.1f}" '
                f'x2="{sx:.1f}" y2="{sy:.1f}"/>'
            )

        prev_x, prev_y = sx, sy

    flush_group()

    # Info text
    info_y = canvas_height - 5
    w_mm = bounds[2] - bounds[0]
    h_mm = bounds[3] - bounds[1]
    info = (f"{w_mm:.0f}×{h_mm:.0f}mm | "
            f"{pattern.stitch_count()} stitches | "
            f"{len(colors)} colors")

    lines.append(
        f'<text x="{margin}" y="{info_y}" '
        f'font-family="monospace" font-size="11" '
        f'fill="#888">{html.escape(info)}</text>'
    )

    # Color legend
    legend_x = canvas_width - margin - len(colors) * 18
    legend_y = canvas_height - 8
    for i, color in enumerate(colors):
        cx = legend_x + i * 18
        hex_c = f"#{color.r:02x}{color.g:02x}{color.b:02x}"
        lines.append(
            f'<rect x="{cx}" y="{legend_y - 10}" '
            f'width="14" height="14" rx="2" '
            f'fill="{hex_c}" stroke="#666" stroke-width="0.5"/>'
        )

    lines.append('</svg>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _darken(hex_color: str, amount: float) -> str:
    """Darken a hex color by a fraction."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = max(0, int(r * (1 - amount)))
    g = max(0, int(g * (1 - amount)))
    b = max(0, int(b * (1 - amount)))

    return f"#{r:02x}{g:02x}{b:02x}"
