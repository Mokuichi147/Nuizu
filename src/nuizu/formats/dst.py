"""Tajima DST embroidery file format writer.

DST is the most universal embroidery format, supported by
virtually all embroidery machines. Coordinates are in 0.1mm units.

File structure:
  - 512-byte header
  - Stitch data (3 bytes per command)
"""

from .common import EmbroideryPattern, StitchType

MAX_STEP = 121
END_CMD = bytes([0x00, 0x00, 0xF3])
COLOR_CHANGE_CMD = bytes([0x00, 0x00, 0xC3])

def _make_dst_header(
    pattern: EmbroideryPattern,
    stitch_records: int,
    color_changes: int,
    bounds_units: tuple[int, int, int, int],
    end_pos_units: tuple[int, int],
) -> bytes:
    """Create 512-byte DST header."""
    header = bytearray(b'\x20' * 512)

    def _write_field(offset: int, text: str, field_len: int):
        """Write a fixed-length field into the header."""
        encoded = text.encode('ascii', errors='replace')[:field_len]
        for i, b in enumerate(encoded):
            header[offset + i] = b

    # Label (20 bytes)
    label = pattern.name[:16].ljust(16)
    _write_field(0, f"LA:{label}\r", 20)

    # Record count (11 bytes)
    _write_field(20, f"ST:{stitch_records:7d}\r", 11)

    # Color change count (7 bytes)
    _write_field(31, f"CO:{color_changes:3d}\r", 7)

    # Bounds (9 bytes each), units: 0.1mm
    min_x, max_x, min_y, max_y = bounds_units

    pos = 38
    for tag, val in [
        ("+X", abs(max_x)),
        ("-X", abs(min_x)),
        ("+Y", abs(max_y)),
        ("-Y", abs(min_y)),
    ]:
        _write_field(pos, f"{tag}:{val:5d}\r", 9)
        pos += 9

    # AX, AY (final needle position in file coordinate convention)
    ax, ay = end_pos_units[0], -end_pos_units[1]
    ax_sign = "+" if ax >= 0 else "-"
    ay_sign = "+" if ay >= 0 else "-"
    _write_field(pos, f"AX:{ax_sign}{abs(ax):5d}\r", 10)
    pos += 10
    _write_field(pos, f"AY:{ay_sign}{abs(ay):5d}\r", 10)
    pos += 10

    # MX, MY
    _write_field(pos, f"MX:+{0:5d}\r", 10)
    pos += 10
    _write_field(pos, f"MY:+{0:5d}\r", 10)
    pos += 10

    # PD
    _write_field(pos, "PD:******\r", 10)
    pos += 10

    # Header end marker. Remaining bytes stay as spaces.
    if pos < 512:
        header[pos] = 0x1A

    return bytes(header)


def write_dst(pattern: EmbroideryPattern, filepath: str):
    """Write embroidery pattern to Tajima DST file.

    Args:
        pattern: The embroidery pattern to write.
        filepath: Output file path.
    """
    pattern.center_pattern()

    stitch_stream, color_changes, bounds_units, end_pos_units = (
        _build_dst_stitch_stream(pattern)
    )
    header = _make_dst_header(
        pattern=pattern,
        stitch_records=len(stitch_stream) // 3,
        color_changes=color_changes,
        bounds_units=bounds_units,
        end_pos_units=end_pos_units,
    )

    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(stitch_stream)


def _build_dst_stitch_stream(
    pattern: EmbroideryPattern,
) -> tuple[bytearray, int, tuple[int, int, int, int], tuple[int, int]]:
    """Build DST command stream and metadata stats."""
    stream = bytearray()
    color_changes = 0

    # Track integer positions in 0.1mm to prevent cumulative float drift.
    current_x = 0
    current_y = 0
    min_x = max_x = 0
    min_y = max_y = 0
    has_end = False

    for stitch in pattern.stitches:
        if stitch.stitch_type == StitchType.END:
            stream.extend(END_CMD)
            has_end = True
            break

        target_x = int(round(stitch.x * 10))
        target_y = int(round(stitch.y * 10))
        dx = target_x - current_x
        dy = target_y - current_y

        if stitch.stitch_type == StitchType.COLOR_CHANGE:
            _append_dst_move(stream, dx, dy, is_jump=True)
            stream.extend(COLOR_CHANGE_CMD)
            color_changes += 1
        elif stitch.stitch_type == StitchType.TRIM:
            _append_dst_move(stream, dx, dy, is_jump=True)
        else:
            _append_dst_move(
                stream,
                dx,
                dy,
                is_jump=(stitch.stitch_type == StitchType.JUMP),
            )

        current_x = target_x
        current_y = target_y
        min_x = min(min_x, current_x)
        max_x = max(max_x, current_x)
        min_y = min(min_y, current_y)
        max_y = max(max_y, current_y)

    if not has_end:
        stream.extend(END_CMD)

    return (
        stream,
        color_changes,
        (min_x, max_x, min_y, max_y),
        (current_x, current_y),
    )


def _append_dst_move(
    stream: bytearray, dx: int, dy: int, is_jump: bool = False
) -> None:
    """Write a DST movement, splitting if necessary."""
    while abs(dx) > MAX_STEP or abs(dy) > MAX_STEP:
        step_x = max(-MAX_STEP, min(MAX_STEP, dx))
        step_y = max(-MAX_STEP, min(MAX_STEP, dy))

        stream.extend(_encode_dst_byte(step_x, step_y, is_jump))

        dx -= step_x
        dy -= step_y

    stream.extend(_encode_dst_byte(dx, dy, is_jump))


def _encode_dst_byte(dx: int, dy: int, is_jump: bool = False) -> bytes:
    """Encode one DST move command (0.1mm units)."""
    # DST stores Y with opposite sign from common Cartesian interpretation.
    y = -dy
    x = dx
    b0 = 0
    b1 = 0
    b2 = 0x03

    if x > 40:
        b2 |= 1 << 2
        x -= 81
    if x < -40:
        b2 |= 1 << 3
        x += 81
    if x > 13:
        b1 |= 1 << 2
        x -= 27
    if x < -13:
        b1 |= 1 << 3
        x += 27
    if x > 4:
        b0 |= 1 << 2
        x -= 9
    if x < -4:
        b0 |= 1 << 3
        x += 9
    if x > 1:
        b1 |= 1 << 0
        x -= 3
    if x < -1:
        b1 |= 1 << 1
        x += 3
    if x > 0:
        b0 |= 1 << 0
        x -= 1
    if x < 0:
        b0 |= 1 << 1
        x += 1

    if y > 40:
        b2 |= 1 << 5
        y -= 81
    if y < -40:
        b2 |= 1 << 4
        y += 81
    if y > 13:
        b1 |= 1 << 5
        y -= 27
    if y < -13:
        b1 |= 1 << 4
        y += 27
    if y > 4:
        b0 |= 1 << 5
        y -= 9
    if y < -4:
        b0 |= 1 << 4
        y += 9
    if y > 1:
        b1 |= 1 << 7
        y -= 3
    if y < -1:
        b1 |= 1 << 6
        y += 3
    if y > 0:
        b0 |= 1 << 7
        y -= 1
    if y < 0:
        b0 |= 1 << 6
        y += 1

    if x != 0 or y != 0:
        raise ValueError(f"DST move out of range: dx={dx}, dy={dy}")

    if is_jump:
        b2 |= 1 << 7

    return bytes([b0, b1, b2])
