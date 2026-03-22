"""Brother PES embroidery file format writer.

PES is widely used by Brother and Babylock machines.
Internally uses PEC (Brother's stitch encoding) for the stitch data.

Simplified implementation supporting PES v1.
"""

import struct
from .common import EmbroideryPattern, StitchType, ThreadColor


# PEC thread color table (Brother standard)
PEC_COLORS = [
    (0, 0, 0),       # 0 - Unknown
    (14, 31, 124),    # 1 - Prussian Blue
    (10, 85, 163),    # 2 - Blue
    (48, 135, 119),   # 3 - Teal Green
    (75, 107, 175),   # 4 - Cornflower Blue
    (237, 23, 31),    # 5 - Red
    (209, 92, 0),     # 6 - Reddish Brown
    (145, 54, 151),   # 7 - Magenta
    (228, 154, 203),  # 8 - Light Lilac
    (145, 95, 172),   # 9 - Lilac
    (158, 214, 125),  # 10 - Mint Green
    (232, 169, 0),    # 11 - Deep Gold
    (254, 186, 53),   # 12 - Orange
    (255, 255, 0),    # 13 - Yellow
    (112, 188, 31),   # 14 - Lime Green
    (186, 152, 0),    # 15 - Brass
    (168, 168, 168),  # 16 - Silver
    (125, 111, 0),    # 17 - Russet Brown
    (255, 255, 179),  # 18 - Cream Brown
    (79, 85, 86),     # 19 - Pewter
    (0, 0, 0),        # 20 - Black
    (11, 61, 145),    # 21 - Ultra Marine
    (119, 1, 118),    # 22 - Royal Purple
    (41, 49, 51),     # 23 - Dark Gray
    (42, 19, 1),      # 24 - Dark Brown
    (246, 74, 138),   # 25 - Deep Rose
    (178, 118, 36),   # 26 - Light Brown
    (252, 187, 197),  # 27 - Salmon Pink
    (254, 55, 15),    # 28 - Vermillion
    (240, 240, 240),  # 29 - White
    (106, 28, 138),   # 30 - Violet
    (168, 221, 196),  # 31 - Seacrest
    (37, 132, 187),   # 32 - Sky Blue
    (254, 179, 67),   # 33 - Pumpkin
    (255, 243, 107),  # 34 - Cream Yellow
    (208, 166, 96),   # 35 - Khaki
    (209, 84, 0),     # 36 - Clay Brown
    (102, 186, 73),   # 37 - Leaf Green
    (19, 74, 70),     # 38 - Peacock Blue
    (135, 135, 135),  # 39 - Gray
    (216, 204, 198),  # 40 - Warm Gray
    (67, 86, 7),      # 41 - Dark Olive
    (253, 217, 222),  # 42 - Flesh Pink
    (249, 147, 188),  # 43 - Pink
    (0, 56, 34),      # 44 - Deep Green
    (178, 175, 212),  # 45 - Lavender
    (104, 106, 176),  # 46 - Wisteria Blue
    (239, 227, 185),  # 47 - Beige
    (247, 56, 102),   # 48 - Carmine
    (181, 75, 100),   # 49 - Amber Red
    (19, 43, 26),     # 50 - Olive Green
    (199, 1, 86),     # 51 - Dark Fuchsia
    (254, 158, 50),   # 52 - Tangerine
    (168, 222, 235),  # 53 - Light Blue
    (0, 103, 62),     # 54 - Emerald Green
    (78, 41, 144),    # 55 - Purple
    (47, 126, 32),    # 56 - Moss Green
    (255, 204, 204),  # 57 - Flesh Pink (alt)
    (255, 217, 17),   # 58 - Harvest Gold
    (9, 91, 166),     # 59 - Electric Blue
    (240, 249, 112),  # 60 - Lemon Yellow
    (227, 243, 91),   # 61 - Fresh Green
    (255, 153, 0),    # 62 - Orange (alt)
    (255, 240, 141),  # 63 - Cream Yellow (alt)
    (255, 200, 200),  # 64 - Applique
]


def _find_nearest_pec_color(r: int, g: int, b: int) -> int:
    """Find nearest PEC color index for given RGB."""
    best_idx = 20  # Default to black
    best_dist = float('inf')

    for i, (pr, pg, pb) in enumerate(PEC_COLORS):
        if i == 0:
            continue
        dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def write_pes(pattern: EmbroideryPattern, filepath: str):
    """Write embroidery pattern to Brother PES v1 file."""
    pattern.center_pattern()

    # Build PEC stitch data
    pec_data = _build_pec_data(pattern)

    with open(filepath, 'wb') as f:
        # PES header
        f.write(b'#PES0001')

        # PEC offset (will be filled after header)
        pec_offset_pos = f.tell()
        f.write(struct.pack('<I', 0))  # Placeholder

        # Minimal PES header (no embedded objects)
        f.write(struct.pack('<H', 1))   # Hoop size (0=100x100, 1=130x180)
        f.write(struct.pack('<H', 1))   # Use existing design
        f.write(struct.pack('<H', 0))   # Segment count = 0

        # Write PEC offset
        pec_offset = f.tell()
        current = f.tell()
        f.seek(pec_offset_pos)
        f.write(struct.pack('<I', pec_offset))
        f.seek(current)

        # PEC section
        _write_pec_section(f, pattern, pec_data)


def _build_pec_data(pattern: EmbroideryPattern) -> bytearray:
    """Build PEC-encoded stitch data."""
    data = bytearray()
    last_x = 0.0
    last_y = 0.0

    for stitch in pattern.stitches:
        dx = int(round((stitch.x - last_x) * 10))  # 0.1mm units
        dy = int(round((stitch.y - last_y) * 10))

        if stitch.stitch_type == StitchType.END:
            data.append(0xFF)
            break

        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            data.append(0xFE)
            data.append(0xB0)
            # Color index byte
            color_count = len([s for s in pattern.stitches[:pattern.stitches.index(stitch)]
                              if s.stitch_type == StitchType.COLOR_CHANGE])
            data.append(color_count + 1)
            last_x = stitch.x
            last_y = stitch.y
            continue

        if stitch.stitch_type == StitchType.TRIM:
            # Encode trim as large jump
            _pec_encode_stitch(data, dx, dy, jump=True)
            last_x = stitch.x
            last_y = stitch.y
            continue

        is_jump = stitch.stitch_type == StitchType.JUMP

        # Split long moves
        while abs(dx) > 2047 or abs(dy) > 2047:
            step_x = max(-2047, min(2047, dx))
            step_y = max(-2047, min(2047, dy))
            _pec_encode_stitch(data, step_x, step_y, jump=True)
            dx -= step_x
            dy -= step_y

        _pec_encode_stitch(data, dx, dy, jump=is_jump)

        last_x = stitch.x
        last_y = stitch.y

    data.append(0xFF)
    return data


def _pec_encode_stitch(data: bytearray, dx: int, dy: int,
                       jump: bool = False):
    """Encode a single PEC stitch."""
    if -63 <= dx <= 63 and -63 <= dy <= 63 and not jump:
        # Short form: 1 byte each
        data.append(dx & 0x7F)
        data.append(dy & 0x7F)
    else:
        # Long form: 2 bytes each (12-bit signed)
        val_x = dx & 0x0FFF
        if jump:
            val_x |= 0x1000  # Jump flag
        data.append(0x80 | ((val_x >> 8) & 0x0F) | (0x10 if jump else 0))
        data.append(val_x & 0xFF)

        val_y = dy & 0x0FFF
        if jump:
            val_y |= 0x1000
        data.append(0x80 | ((val_y >> 8) & 0x0F) | (0x10 if jump else 0))
        data.append(val_y & 0xFF)


def _write_pec_section(f, pattern: EmbroideryPattern,
                       pec_data: bytearray):
    """Write the PEC section of a PES file."""
    # PEC label
    label = pattern.name[:16].ljust(16)
    f.write(f"LA:{label}\r".encode('ascii')[:20])

    # Padding
    f.write(bytes(11))

    # PEC color count
    n_colors = max(1, len(pattern.colors))
    f.write(struct.pack('B', n_colors - 1))

    # Color indices (map to PEC palette)
    for color in pattern.colors:
        idx = _find_nearest_pec_color(color.r, color.g, color.b)
        f.write(struct.pack('B', idx))

    # Pad to 463 bytes from start of PEC section
    pad_needed = 463 - (20 + 11 + 1 + len(pattern.colors))
    if pad_needed > 0:
        f.write(bytes(pad_needed))

    # Thumbnail placeholder (blank 6x38 bitmap)
    thumb_size = 6 * 38
    f.write(bytes(thumb_size))
    f.write(bytes(thumb_size))  # Second thumbnail

    # Stitch data
    f.write(pec_data)
