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

_PEC_ICON_WIDTH = 48
_PEC_ICON_HEIGHT = 38
_PEC_ICON_BYTES = (_PEC_ICON_WIDTH // 8) * _PEC_ICON_HEIGHT  # 228


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

    with open(filepath, 'wb') as f:
        # --- PES header ---
        f.write(b'#PES0001')

        # PEC offset placeholder
        pec_offset_pos = f.tell()
        f.write(struct.pack('<I', 0))

        # Truncated v1: 10 bytes of zero padding
        f.write(bytes(10))

        # Record PEC offset and patch it
        pec_offset = f.tell()  # should be 22 (0x16)
        f.seek(pec_offset_pos)
        f.write(struct.pack('<I', pec_offset))
        f.seek(pec_offset)

        # --- PEC section ---
        _write_pec_section(f, pattern)


def _build_pec_data(pattern: EmbroideryPattern) -> bytearray:
    """Build PEC-encoded stitch data."""
    data = bytearray()
    last_x = 0.0
    last_y = 0.0
    color_change_number = 0

    for i, stitch in enumerate(pattern.stitches):
        dx = int(round((stitch.x - last_x) * 10))  # 0.1mm units
        dy = int(round((stitch.y - last_y) * 10))

        if stitch.stitch_type == StitchType.END:
            data.append(0xFF)
            return data

        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            data.append(0xFE)
            data.append(0xB0)
            color_change_number += 1
            # Alternating 0x02 / 0x01 per pyembroidery convention
            data.append((color_change_number % 2) + 1)
            last_x = stitch.x
            last_y = stitch.y
            continue

        if stitch.stitch_type == StitchType.TRIM:
            _pec_encode_move(data, dx, dy, trim=True)
            last_x = stitch.x
            last_y = stitch.y
            continue

        is_jump = stitch.stitch_type == StitchType.JUMP

        # Split long moves
        while abs(dx) > 2047 or abs(dy) > 2047:
            step_x = max(-2047, min(2047, dx))
            step_y = max(-2047, min(2047, dy))
            _pec_encode_move(data, step_x, step_y, jump=True)
            dx -= step_x
            dy -= step_y

        if is_jump:
            _pec_encode_move(data, dx, dy, jump=True)
        else:
            _pec_encode_stitch(data, dx, dy)

        last_x = stitch.x
        last_y = stitch.y

    # End marker (only if not already written by an END stitch)
    data.append(0xFF)
    return data


def _pec_encode_stitch(data: bytearray, dx: int, dy: int):
    """Encode a normal PEC stitch (short form if possible)."""
    if -64 < dx < 63 and -64 < dy < 63:
        data.append(dx & 0x7F)
        data.append(dy & 0x7F)
    else:
        _pec_encode_long(data, dx, dy, flag=0)


def _pec_encode_move(data: bytearray, dx: int, dy: int,
                     jump: bool = False, trim: bool = False):
    """Encode a PEC jump or trim move (always long form)."""
    if trim:
        flag = 0x20  # TRIM_CODE
    elif jump:
        flag = 0x10  # JUMP_CODE
    else:
        flag = 0
    _pec_encode_long(data, dx, dy, flag=flag)


def _pec_encode_long(data: bytearray, dx: int, dy: int, flag: int):
    """Encode a long-form PEC stitch with optional command flag."""
    vx = dx & 0x0FFF
    vx |= 0x8000 | (flag << 8)
    data.append((vx >> 8) & 0xFF)
    data.append(vx & 0xFF)

    vy = dy & 0x0FFF
    vy |= 0x8000 | (flag << 8)
    data.append((vy >> 8) & 0xFF)
    data.append(vy & 0xFF)


def _write_pec_section(f, pattern: EmbroideryPattern):
    """Write the PEC section of a PES file."""
    n_colors = max(1, len(pattern.colors))

    # --- PEC label (20 bytes) ---
    name = pattern.name[:8]
    label = f"LA:{name:<16s}\r".encode('ascii', errors='replace')[:20]
    f.write(label)

    # --- Padding: 12 bytes of 0x20 ---
    f.write(b'\x20' * 12)

    # --- Fixed fields ---
    f.write(b'\xFF\x00')

    # Thumbnail parameters
    stride = _PEC_ICON_WIDTH // 8  # 6
    f.write(struct.pack('B', stride))
    f.write(struct.pack('B', _PEC_ICON_HEIGHT))  # 38

    # --- 12 bytes of 0x20 ---
    f.write(b'\x20' * 12)

    # --- Color count and color index table ---
    f.write(struct.pack('B', n_colors - 1))
    color_indices = []
    for color in pattern.colors:
        idx = _find_nearest_pec_color(color.r, color.g, color.b)
        color_indices.append(idx)
        f.write(struct.pack('B', idx))

    # Pad color table to 463 entries with 0x20
    for _ in range(n_colors, 463):
        f.write(b'\x20')

    # --- Stitch block header (16 bytes) ---
    f.write(b'\x00\x00')  # Fixed

    # Stitch block length placeholder (3 bytes, uint24 LE)
    block_len_pos = f.tell()
    f.write(b'\x00\x00\x00')

    f.write(b'\x31\xFF\xF0')  # Fixed magic

    # Design dimensions in 0.1mm units
    min_x, min_y, max_x, max_y = pattern.get_bounds()
    width_units = int(round((max_x - min_x) * 10))
    height_units = int(round((max_y - min_y) * 10))
    f.write(struct.pack('<h', width_units))
    f.write(struct.pack('<h', height_units))

    # Fixed center offsets
    f.write(struct.pack('<H', 0x01E0))  # 480
    f.write(struct.pack('<H', 0x01B0))  # 432

    # --- Stitch data ---
    stitch_block_start = block_len_pos - 2  # from 0x00 0x00
    pec_data = _build_pec_data(pattern)
    f.write(pec_data)

    # --- Graphics / thumbnails ---
    # Blank thumbnail for overall view
    f.write(bytes(_PEC_ICON_BYTES))
    # One blank thumbnail per color
    for _ in range(n_colors):
        f.write(bytes(_PEC_ICON_BYTES))

    # --- Patch stitch block length ---
    stitch_block_end = f.tell()
    block_length = stitch_block_end - stitch_block_start
    f.seek(block_len_pos)
    # uint24 little-endian
    f.write(struct.pack('<I', block_length)[:3])
    f.seek(stitch_block_end)
