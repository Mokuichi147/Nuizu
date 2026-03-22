"""Tajima DST embroidery file format writer.

DST is the most universal embroidery format, supported by
virtually all embroidery machines. Coordinates are in 0.1mm units.

File structure:
  - 512-byte header
  - Stitch data (3 bytes per command)
"""

import struct
from typing import BinaryIO
from .common import EmbroideryPattern, StitchType


# DST coordinate encoding lookup table
def _encode_dst_coord(dx: int, dy: int, flags: int = 0) -> bytes:
    """Encode a single DST stitch command (3 bytes).

    DST uses a complex bit-packing scheme for coordinates.
    dx, dy are in 0.1mm units, range approximately -121 to +121.
    """
    b0 = 0
    b1 = 0
    b2 = flags & 0xFF

    # Y encoding (positive = up)
    if dy > 0:
        if dy & 1:
            b2 |= 0x01
        if dy & 2:
            b2 |= 0x02
        if dy & 4:
            b1 |= 0x04
        if dy & 8:
            b1 |= 0x10
        if dy & 16:
            b0 |= 0x04
        if dy & 32:
            b0 |= 0x10
        if dy & 64:
            b0 |= 0x40
    elif dy < 0:
        dy = -dy
        if dy & 1:
            b2 |= 0x01
        if dy & 2:
            b2 |= 0x02
        if dy & 4:
            b1 |= 0x04
        if dy & 8:
            b1 |= 0x10
        if dy & 16:
            b0 |= 0x04
        if dy & 32:
            b0 |= 0x10
        if dy & 64:
            b0 |= 0x40
        # Set negative Y bits
        b2 |= 0x04
        b1 |= 0x08
        b0 |= 0x08

    # X encoding (positive = right)
    if dx > 0:
        if dx & 1:
            b1 |= 0x01
        if dx & 2:
            b1 |= 0x02
        if dx & 4:
            b2 |= 0x10
        if dx & 8:
            b2 |= 0x40
        if dx & 16:
            b0 |= 0x01
        if dx & 32:
            b0 |= 0x02
        if dx & 64:
            b0 |= 0x20
    elif dx < 0:
        dx = -dx
        if dx & 1:
            b1 |= 0x01
        if dx & 2:
            b1 |= 0x02
        if dx & 4:
            b2 |= 0x10
        if dx & 8:
            b2 |= 0x40
        if dx & 16:
            b0 |= 0x01
        if dx & 32:
            b0 |= 0x02
        if dx & 64:
            b0 |= 0x20
        # Set negative X bits
        b1 |= 0x20
        b1 |= 0x40
        b0 |= 0x80

    # Normal stitch has bit 0x80 set in byte 2
    b2 |= 0x03  # Set two LSBs of b2 as required by DST

    return bytes([b0, b1, b2])


def _make_dst_header(pattern: EmbroideryPattern) -> bytes:
    """Create 512-byte DST header."""
    header = bytearray(b'\x20' * 512)

    def _write_field(offset: int, text: str, field_len: int):
        """Write a fixed-length field into the header."""
        encoded = text.encode('ascii')[:field_len]
        for i, b in enumerate(encoded):
            header[offset + i] = b

    # Label (20 bytes)
    label = pattern.name[:16].ljust(16)
    _write_field(0, f"LA:{label}\r", 20)

    # Stitch count (11 bytes)
    sc = pattern.stitch_count()
    _write_field(20, f"ST:{sc:07d}\r", 11)

    # Color count (7 bytes)
    cc = len(pattern.colors)
    _write_field(31, f"CO:{cc:03d}\r", 7)

    # Bounds (9 bytes each)
    bounds = pattern.get_bounds()
    max_x_units = int(abs(bounds[2]) * 10)
    min_x_units = int(abs(bounds[0]) * 10)
    max_y_units = int(abs(bounds[3]) * 10)
    min_y_units = int(abs(bounds[1]) * 10)

    pos = 38
    for tag, val in [
        ("+X", max_x_units),
        ("-X", min_x_units),
        ("+Y", max_y_units),
        ("-Y", min_y_units),
    ]:
        _write_field(pos, f"{tag}:{val:05d}\r", 9)
        pos += 9

    # AX, AY
    _write_field(pos, f"AX:+{0:05d}\r", 10); pos += 10
    _write_field(pos, f"AY:+{0:05d}\r", 10); pos += 10

    # MX, MY
    _write_field(pos, f"MX:+{0:05d}\r", 10); pos += 10
    _write_field(pos, f"MY:+{0:05d}\r", 10); pos += 10

    # PD
    _write_field(pos, "PD:******\r", 10); pos += 10

    # End-of-header marker
    header[510] = 0x1A
    header[511] = 0x00

    return bytes(header)


def write_dst(pattern: EmbroideryPattern, filepath: str):
    """Write embroidery pattern to Tajima DST file.

    Args:
        pattern: The embroidery pattern to write.
        filepath: Output file path.
    """
    pattern.center_pattern()

    with open(filepath, 'wb') as f:
        # Write header
        f.write(_make_dst_header(pattern))

        # Convert stitches to relative movements
        last_x = 0.0
        last_y = 0.0

        for stitch in pattern.stitches:
            if stitch.stitch_type == StitchType.END:
                # End command
                f.write(bytes([0x00, 0x00, 0xF3]))
                break

            # Convert mm to 0.1mm units
            x_units = stitch.x * 10
            y_units = stitch.y * 10

            dx = int(round(x_units - last_x * 10))
            dy = int(round(y_units - last_y * 10))

            if stitch.stitch_type == StitchType.COLOR_CHANGE:
                # Color change: stop command
                _write_dst_move(f, dx, dy, is_jump=True)
                f.write(bytes([0x00, 0x00, 0xC3]))  # Color change
                last_x = stitch.x
                last_y = stitch.y
                continue

            if stitch.stitch_type == StitchType.TRIM:
                _write_dst_move(f, dx, dy, is_jump=True)
                last_x = stitch.x
                last_y = stitch.y
                continue

            is_jump = stitch.stitch_type == StitchType.JUMP

            # DST max step is about 121 units (12.1mm)
            _write_dst_move(f, dx, dy, is_jump=is_jump)

            last_x = stitch.x
            last_y = stitch.y

        # End of file
        f.write(bytes([0x00, 0x00, 0xF3]))


def _write_dst_move(f: BinaryIO, dx: int, dy: int,
                    is_jump: bool = False):
    """Write a DST movement, splitting if necessary."""
    MAX_STEP = 121

    while abs(dx) > MAX_STEP or abs(dy) > MAX_STEP:
        step_x = max(-MAX_STEP, min(MAX_STEP, dx))
        step_y = max(-MAX_STEP, min(MAX_STEP, dy))

        flags = 0x80 if is_jump else 0x00
        data = _encode_dst_byte(step_x, step_y, is_jump)
        f.write(data)

        dx -= step_x
        dy -= step_y

    data = _encode_dst_byte(dx, dy, is_jump)
    f.write(data)


def _encode_dst_byte(dx: int, dy: int, is_jump: bool = False) -> bytes:
    """Encode a single DST stitch using the standard bit-packing."""
    b0 = 0
    b1 = 0
    b2 = 0x03  # Two LSBs always set

    # Encode Y axis
    ay = abs(dy)
    if ay & 1:
        b2 |= 0x01  # Already set by 0x03
    if ay & 2:
        b2 |= 0x02  # Already set by 0x03
    if ay & 4:
        b1 |= 0x04
    if ay & 8:
        b1 |= 0x10
    if ay & 16:
        b0 |= 0x04
    if ay & 32:
        b0 |= 0x10
    if ay & 64:
        b0 |= 0x40

    if dy < 0:
        b2 |= 0x04
        b1 |= 0x08
        b0 |= 0x08

    # Encode X axis
    ax = abs(dx)
    if ax & 1:
        b1 |= 0x01
    if ax & 2:
        b1 |= 0x02
    if ax & 4:
        b2 |= 0x10
    if ax & 8:
        b2 |= 0x40
    if ax & 16:
        b0 |= 0x01
    if ax & 32:
        b0 |= 0x02
    if ax & 64:
        b0 |= 0x20

    if dx < 0:
        b1 |= 0x20
        b1 |= 0x40
        b0 |= 0x80

    if is_jump:
        b2 |= 0x80  # Jump flag

    return bytes([b0, b1, b2])
