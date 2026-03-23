"""JANOME JEF embroidery file format writer.

JEF is JANOME's proprietary embroidery format.
Coordinates are in 0.1mm units with relative movements.

File structure:
  - Fixed header (116 bytes)
  - Color index table (color_count * 4 bytes)
  - Color type table (color_count * 4 bytes)
  - Stitch data
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

# Hoop definitions: (half_width, half_height) in 0.1mm
_HOOPS = [
    (550, 550),    # 0: 110x110mm
    (250, 250),    # 1: 50x50mm
    (700, 1000),   # 2: 140x200mm
    (630, 550),    # 3: 126x110mm (JANOME default)
    (1000, 1000),  # 4: 200x200mm
]

_MAX_STITCH = 127


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
    """Write embroidery pattern to JANOME JEF file."""
    pattern.center_pattern()

    n_colors = max(1, len(pattern.colors))

    # Build stitch data first to count points
    stitch_data, point_count = _build_jef_stitches(pattern)

    # Design half-extents in 0.1mm
    bounds = pattern.get_bounds()
    half_w = int(round((bounds[2] - bounds[0]) * 10)) // 2
    half_h = int(round((bounds[3] - bounds[1]) * 10)) // 2

    with open(filepath, 'wb') as f:
        # --- JEF Header (116 bytes) ---

        # Offset to stitch data
        header_size = 116 + n_colors * 8
        f.write(struct.pack('<i', header_size))

        # Flags (fixed 0x14)
        f.write(struct.pack('<i', 0x00000014))

        # Date/time (14 bytes) + 2 bytes padding (null)
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d%H%M%S").encode('ascii')
        f.write(date_str[:14])
        f.write(b'\x00\x00')

        # Color count (= number of thread changes + 1 initial)
        f.write(struct.pack('<i', n_colors))

        # Point count (includes END)
        f.write(struct.pack('<i', point_count))

        # Hoop code (0 = 110x110)
        f.write(struct.pack('<i', 0))

        # Design half-extents (written twice per spec)
        f.write(struct.pack('<i', half_w))
        f.write(struct.pack('<i', half_h))
        f.write(struct.pack('<i', half_w))
        f.write(struct.pack('<i', half_h))

        # Hoop edge distances for 4 hoop types
        for hoop_hw, hoop_hh in _HOOPS[:4]:
            x_edge = hoop_hw - half_w
            y_edge = hoop_hh - half_h
            if x_edge < 0 or y_edge < 0:
                # Design doesn't fit this hoop
                f.write(struct.pack('<iiii', -1, -1, -1, -1))
            else:
                f.write(struct.pack('<iiii', x_edge, y_edge, x_edge, y_edge))

        # --- Color Table ---
        # Table 1: color indices (all n_colors entries)
        color_indices = []
        for color in pattern.colors:
            color_indices.append(
                _find_nearest_jef_color(color.r, color.g, color.b))
        # Pad if fewer colors than n_colors
        while len(color_indices) < n_colors:
            color_indices.append(0)
        for idx in color_indices:
            f.write(struct.pack('<i', idx))

        # Table 2: color types (all 0x0D)
        for _ in range(n_colors):
            f.write(struct.pack('<i', 0x0D))

        # --- Stitch Data ---
        f.write(stitch_data)


def _build_jef_stitches(pattern: EmbroideryPattern):
    """Build JEF stitch data. Returns (data, point_count)."""
    data = bytearray()
    last_x = 0.0
    last_y = 0.0
    point_count = 1  # END counts as 1

    for stitch in pattern.stitches:
        dx = int(round((stitch.x - last_x) * 10))
        dy = int(round((stitch.y - last_y) * 10))

        if stitch.stitch_type == StitchType.END:
            data.extend(b'\x80\x10')
            return data, point_count

        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            _jef_write_command(data, 0x01, dx, dy)
            point_count += 2
            last_x = stitch.x
            last_y = stitch.y
            continue

        if stitch.stitch_type == StitchType.TRIM:
            # Trim = jump with (0,0) repeated 3 times
            for _ in range(3):
                _jef_write_command(data, 0x02, 0, 0)
                point_count += 2
            last_x = stitch.x
            last_y = stitch.y
            continue

        if stitch.stitch_type == StitchType.JUMP:
            _jef_write_jump_split(data, dx, dy)
            # Count: each split segment = 2 points
            steps = max(1, max(
                (abs(dx) + _MAX_STITCH - 1) // _MAX_STITCH if dx != 0 else 1,
                (abs(dy) + _MAX_STITCH - 1) // _MAX_STITCH if dy != 0 else 1,
            ))
            point_count += steps * 2
            last_x = stitch.x
            last_y = stitch.y
            continue

        # Normal stitch - split if too long
        _jef_write_stitch_split(data, dx, dy)
        steps = max(1, max(
            (abs(dx) + _MAX_STITCH - 1) // _MAX_STITCH if dx != 0 else 1,
            (abs(dy) + _MAX_STITCH - 1) // _MAX_STITCH if dy != 0 else 1,
        ))
        point_count += steps
        last_x = stitch.x
        last_y = stitch.y

    # End marker (only if no END stitch was encountered)
    data.extend(b'\x80\x10')
    return data, point_count


def _jef_write_command(data: bytearray, ctrl: int, dx: int, dy: int):
    """Write a JEF control command: 0x80 ctrl dx (-dy)."""
    data.append(0x80)
    data.append(ctrl & 0xFF)
    data.append(dx & 0xFF)
    data.append((-dy) & 0xFF)


def _jef_write_stitch(data: bytearray, dx: int, dy: int):
    """Write a single normal JEF stitch: dx (-dy) as signed bytes."""
    data.append(dx & 0xFF)
    data.append((-dy) & 0xFF)


def _jef_write_stitch_split(data: bytearray, dx: int, dy: int):
    """Write a normal stitch, splitting if exceeding MAX_STITCH."""
    while abs(dx) > _MAX_STITCH or abs(dy) > _MAX_STITCH:
        step_x = max(-_MAX_STITCH, min(_MAX_STITCH, dx))
        step_y = max(-_MAX_STITCH, min(_MAX_STITCH, dy))
        _jef_write_stitch(data, step_x, step_y)
        dx -= step_x
        dy -= step_y
    _jef_write_stitch(data, dx, dy)


def _jef_write_jump_split(data: bytearray, dx: int, dy: int):
    """Write a jump move, splitting if exceeding MAX_STITCH."""
    while abs(dx) > _MAX_STITCH or abs(dy) > _MAX_STITCH:
        step_x = max(-_MAX_STITCH, min(_MAX_STITCH, dx))
        step_y = max(-_MAX_STITCH, min(_MAX_STITCH, dy))
        _jef_write_command(data, 0x02, step_x, step_y)
        dx -= step_x
        dy -= step_y
    _jef_write_command(data, 0x02, dx, dy)
