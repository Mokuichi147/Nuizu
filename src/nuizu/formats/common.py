"""Common data structures for embroidery patterns."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple


class StitchType(IntEnum):
    """Types of stitch commands."""
    NORMAL = 0    # Regular stitch
    JUMP = 1      # Move without stitching
    TRIM = 2      # Cut thread
    COLOR_CHANGE = 3  # Change thread color
    END = 4       # End of pattern


@dataclass
class Stitch:
    """A single stitch command."""
    x: float  # X position in mm
    y: float  # Y position in mm
    stitch_type: StitchType = StitchType.NORMAL


@dataclass
class ThreadColor:
    """Thread color definition."""
    r: int
    g: int
    b: int
    name: str = ""
    catalog_number: str = ""

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass
class EmbroideryPattern:
    """Complete embroidery pattern."""
    stitches: List[Stitch] = field(default_factory=list)
    colors: List[ThreadColor] = field(default_factory=list)
    name: str = "Untitled"

    # Design dimensions in mm
    width: float = 0.0
    height: float = 0.0

    def add_stitch(self, x: float, y: float,
                   stitch_type: StitchType = StitchType.NORMAL):
        self.stitches.append(Stitch(x, y, stitch_type))

    def add_color_change(self):
        if self.stitches:
            last = self.stitches[-1]
            self.stitches.append(
                Stitch(last.x, last.y, StitchType.COLOR_CHANGE)
            )

    def add_trim(self):
        if self.stitches:
            last = self.stitches[-1]
            self.stitches.append(Stitch(last.x, last.y, StitchType.TRIM))

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y) in mm."""
        if not self.stitches:
            return (0, 0, 0, 0)
        xs = [s.x for s in self.stitches]
        ys = [s.y for s in self.stitches]
        return (min(xs), min(ys), max(xs), max(ys))

    def center_pattern(self):
        """Center the pattern around origin."""
        min_x, min_y, max_x, max_y = self.get_bounds()
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        for s in self.stitches:
            s.x -= cx
            s.y -= cy
        self.width = max_x - min_x
        self.height = max_y - min_y

    def stitch_count(self) -> int:
        return sum(1 for s in self.stitches
                   if s.stitch_type == StitchType.NORMAL)

    def summary(self) -> str:
        bounds = self.get_bounds()
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        return (
            f"Pattern: {self.name}\n"
            f"  Size: {w:.1f} x {h:.1f} mm\n"
            f"  Stitches: {self.stitch_count()}\n"
            f"  Colors: {len(self.colors)}\n"
            f"  Total commands: {len(self.stitches)}"
        )
