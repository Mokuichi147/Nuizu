"""JANOME JEF embroidery file format writer.

JEF is JANOME's proprietary embroidery format.
Coordinates are in 0.1mm units with relative movements.

File structure:
  - Variable-length header with metadata
  - Color table
  - Stitch data (2 bytes per axis per command)
"""

import struct
import datetime
from .common import EmbroideryPattern, StitchType, ThreadColor


# JANOME thread color table (subset)
JEF_COLORS = [
    (0, 0, 0, "Black"),
    (0, 0, 255, "Blue"),
    (0, 136, 0, "Green"),
    (255, 0, 0, "Red"),
    (128, 0, 128, "Purple"),
    (255, 255, 0, "Yellow"),
    (192, 192, 192, "Gray"),
    (255, 128, 0, "Orange"),
    (0, 255, 255, "Aqua"),
    (255, 192, 203, "Pink"),
    (139, 69, 19, "Brown"),
    (255, 255, 255, "White"),
    (0, 0, 128, "Navy"),
    (0, 128, 0, "Dark Green"),
    (128, 128, 0, "Olive"),
    (128, 0, 0, "Maroon"),
    (0, 128, 128, "Teal"),
    (255, 0, 255, "Magenta"),
    (218, 165, 32, "Goldenrod"),
    (245, 222, 179, "Wheat"),
]


def _find_nearest_jef_color(r: int, g: int, b: int) -> int:
    """Find nearest JEF color index."""
    best_idx = 0
    best_dist = float('inf')

    for i, (jr, jg, jb, _) in enumerate(JEF_COLORS):
        dist = (r - jr) ** 2 + (g - jg) ** 2 + (b - jb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def write_jef(pattern: EmbroideryPattern, filepath: str):
    """Write embroidery pattern to JANOME JEF file.

    Args:
        pattern: The embroidery pattern to write.
        filepath: Output file path.
    """
    pattern.center_pattern()

    n_colors = max(1, len(pattern.colors))

    # Build stitch data first to know its size
    stitch_data = _build_jef_stitches(pattern)

    with open(filepath, 'wb') as f:
        # --- JEF Header ---

        # Offset to stitch data
        header_size = 116 + n_colors * 8
        f.write(struct.pack('<I', header_size))

        # Flags
        f.write(struct.pack('<I', 0x00000014))

        # Date/time
        now = datetime.datetime.now()
        f.write(struct.pack('<14s',
                            now.strftime("%Y%m%d%H%M%S").encode('ascii')))

        # Padding (2 bytes)
        f.write(struct.pack('<H', 0))

        # Thread count
        f.write(struct.pack('<I', n_colors))

        # Stitch count
        stitch_count = pattern.stitch_count()
        f.write(struct.pack('<I', stitch_count))

        # Hoop code (0 = JANOME default 126x110mm)
        f.write(struct.pack('<I', 0))

        # Design area (in 0.1mm)
        bounds = pattern.get_bounds()
        extent_x = int((bounds[2] - bounds[0]) * 10)
        extent_y = int((bounds[3] - bounds[1]) * 10)

        # Hoop bounds (half extents from center)
        half_x = extent_x // 2
        half_y = extent_y // 2
        f.write(struct.pack('<i', half_x))    # +X extent
        f.write(struct.pack('<i', half_y))    # +Y extent
        f.write(struct.pack('<i', -half_x))   # -X extent
        f.write(struct.pack('<i', -half_y))   # -Y extent

        # Design bounds (same as hoop for now)
        f.write(struct.pack('<i', half_x))
        f.write(struct.pack('<i', half_y))
        f.write(struct.pack('<i', -half_x))
        f.write(struct.pack('<i', -half_y))

        # Additional design bounds (repeat)
        f.write(struct.pack('<i', half_x))
        f.write(struct.pack('<i', half_y))
        f.write(struct.pack('<i', -half_x))
        f.write(struct.pack('<i', -half_y))

        # More bounds (4th set)
        f.write(struct.pack('<i', half_x))
        f.write(struct.pack('<i', half_y))
        f.write(struct.pack('<i', -half_x))
        f.write(struct.pack('<i', -half_y))

        # --- Color Table ---
        for color in pattern.colors:
            jef_idx = _find_nearest_jef_color(color.r, color.g, color.b)
            f.write(struct.pack('<I', jef_idx))
            # Color type (0 = normal)
            f.write(struct.pack('<I', 0))

        # Pad if fewer colors than expected
        written_colors = len(pattern.colors)
        for _ in range(n_colors - written_colors):
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', 0))

        # --- Stitch Data ---
        f.write(stitch_data)


def _build_jef_stitches(pattern: EmbroideryPattern) -> bytearray:
    """Build JEF stitch data.

    JEF uses 2-byte signed values for dx and dy.
    Normal stitch: dx, dy within [-124, 124]
    Trim/Jump: dx | 0x8000, dy | 0x8000
    Color change: 0x8001, 0x8001 (after trim)
    End: 0x8000, 0x8000
    """
    data = bytearray()
    last_x = 0.0
    last_y = 0.0

    for stitch in pattern.stitches:
        dx = int(round((stitch.x - last_x) * 10))
        dy = int(round((stitch.y - last_y) * 10))

        if stitch.stitch_type == StitchType.END:
            # End marker
            data.extend(struct.pack('<hh', -32768, -32768))
            break

        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            # Trim then color change
            data.extend(struct.pack('<hh', -32767, -32767))
            last_x = stitch.x
            last_y = stitch.y
            continue

        if stitch.stitch_type in (StitchType.JUMP, StitchType.TRIM):
            # Jump/move
            _jef_write_move(data, dx, dy, is_jump=True)
            last_x = stitch.x
            last_y = stitch.y
            continue

        # Normal stitch - split if too long
        _jef_write_move(data, dx, dy, is_jump=False)
        last_x = stitch.x
        last_y = stitch.y

    # Final end marker
    data.extend(struct.pack('<hh', -32768, -32768))
    return data


def _jef_write_move(data: bytearray, dx: int, dy: int,
                    is_jump: bool = False):
    """Write a JEF movement, splitting long moves."""
    MAX_STEP = 124

    while abs(dx) > MAX_STEP or abs(dy) > MAX_STEP:
        step_x = max(-MAX_STEP, min(MAX_STEP, dx))
        step_y = max(-MAX_STEP, min(MAX_STEP, dy))

        if is_jump:
            sx = step_x if step_x >= 0 else (step_x & 0xFFFF)
            sy = step_y if step_y >= 0 else (step_y & 0xFFFF)
            # Set jump flag (bit 15)
            data.extend(struct.pack('<hh',
                                    step_x | -32768 if is_jump else step_x,
                                    step_y | -32768 if is_jump else step_y))
        else:
            data.extend(struct.pack('<hh', step_x, step_y))

        dx -= step_x
        dy -= step_y

    if is_jump and (dx != 0 or dy != 0):
        # For jumps, set high bit
        jx = dx & 0x7FFF | 0x8000
        jy = dy & 0x7FFF | 0x8000
        data.extend(struct.pack('<HH', jx, jy))
    else:
        data.extend(struct.pack('<hh', dx, dy))
