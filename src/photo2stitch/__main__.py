#!/usr/bin/env python3
"""photo2stitch - Convert photos to embroidery files.

Usage:
    python -m photo2stitch input.jpg output.dst
    python -m photo2stitch input.png output.jef --colors 12 --width 150
    python -m photo2stitch photo.jpg design.pes --auto-angle --pull-comp 0.3
    python -m photo2stitch photo.jpg design.dst --palette brother --preview svg
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog='photo2stitch',
        description='Convert photos to embroidery machine files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported output formats:
  .dst    Tajima (most universal)
  .pes    Brother / Babylock
  .jef    JANOME

Thread palettes:
  janome    JANOME polyester threads (default)
  brother   Brother embroidery threads
  madeira   Madeira Rayon 40
  generic   Combined generic palette

Examples:
  %(prog)s photo.jpg design.dst
  %(prog)s photo.png design.jef --colors 12 --width 150
  %(prog)s photo.jpg design.pes -d 0.3 -a 30 --preview png
  %(prog)s photo.jpg design.dst --auto-angle --pull-comp 0.3
  %(prog)s photo.jpg design.jef --palette brother --skip-bg --auto-crop
  %(prog)s photo.jpg design.dst --preview svg --preview svg
        """,
    )

    parser.add_argument('input', help='Input image file (JPG, PNG, etc.)')
    parser.add_argument('output', help='Output embroidery file (.dst, .pes, .jef)')

    # Size
    size_group = parser.add_argument_group('Size options')
    size_group.add_argument(
        '-W', '--width', type=float, default=100.0,
        help='Target width in mm (default: 100)')
    size_group.add_argument(
        '-H', '--height', type=float, default=None,
        help='Target height in mm (auto if not set)')

    # Colors
    color_group = parser.add_argument_group('Color options')
    color_group.add_argument(
        '-c', '--colors', type=int, default=8,
        help='Number of thread colors (default: 8)')
    color_group.add_argument(
        '--free-colors', action='store_true',
        help='Don\'t snap to thread palette')
    color_group.add_argument(
        '--palette', type=str, default='janome',
        choices=['janome', 'brother', 'madeira', 'generic'],
        help='Thread brand palette (default: janome)')

    # Stitch
    stitch_group = parser.add_argument_group('Stitch options')
    stitch_group.add_argument(
        '-d', '--density', type=float, default=0.4,
        help='Fill row spacing in mm (lower=denser, default: 0.4)')
    stitch_group.add_argument(
        '-l', '--stitch-length', type=float, default=3.0,
        help='Maximum stitch length in mm (default: 3.0)')
    stitch_group.add_argument(
        '-a', '--angle', type=float, default=45.0,
        help='Fill angle in degrees (default: 45)')
    stitch_group.add_argument(
        '--auto-angle', action='store_true',
        help='Auto-optimize fill angle per region')
    stitch_group.add_argument(
        '--no-underlay', action='store_true',
        help='Disable underlay stitches')
    stitch_group.add_argument(
        '--no-outline', action='store_true',
        help='Disable outline stitches')
    stitch_group.add_argument(
        '--satin-outline', action='store_true',
        help='Use satin stitch for outlines')
    stitch_group.add_argument(
        '--pull-comp', type=float, default=0.0,
        metavar='MM',
        help='Pull compensation in mm (0=off, typical: 0.2-0.5)')

    # Processing
    proc_group = parser.add_argument_group('Processing options')
    proc_group.add_argument(
        '--blur', type=int, default=3,
        help='Pre-processing blur radius (0=off, default: 3)')
    proc_group.add_argument(
        '--min-region', type=float, default=0.003,
        help='Minimum region area as fraction (default: 0.003)')
    proc_group.add_argument(
        '--no-enhance', action='store_true',
        help='Disable photo enhancement preprocessing')
    proc_group.add_argument(
        '--auto-crop', action='store_true',
        help='Auto-crop to main subject')
    proc_group.add_argument(
        '--skip-bg', action='store_true',
        help='Skip stitching detected background color')

    # Output
    out_group = parser.add_argument_group('Output options')
    out_group.add_argument(
        '--preview', type=str, default=None,
        choices=['png', 'svg', 'both'],
        help='Generate preview (png, svg, or both)')
    out_group.add_argument(
        '--preview-path', type=str, default=None,
        help='Custom preview image path')
    out_group.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress progress messages')

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ext = '.' + args.output.rsplit('.', 1)[-1].lower() if '.' in args.output \
        else ''
    if ext not in ('.dst', '.pes', '.jef'):
        print(f"Error: Unsupported format: {ext}", file=sys.stderr)
        print("Supported: .dst, .pes, .jef", file=sys.stderr)
        sys.exit(1)

    from .pipeline import convert_photo_to_embroidery, generate_preview
    from .svg_preview import generate_svg_preview

    try:
        pattern = convert_photo_to_embroidery(
            image_path=args.input,
            output_path=args.output,
            target_width_mm=args.width,
            target_height_mm=args.height,
            n_colors=args.colors,
            use_thread_palette=not args.free_colors,
            thread_brand=args.palette,
            fill_density=args.density,
            stitch_length=args.stitch_length,
            fill_angle=args.angle,
            auto_angle=args.auto_angle,
            underlay=not args.no_underlay,
            outline=not args.no_outline,
            outline_satin=args.satin_outline,
            pull_compensation=args.pull_comp,
            blur_radius=args.blur,
            min_region_ratio=args.min_region,
            enhance_photo=not args.no_enhance,
            auto_crop=args.auto_crop,
            skip_background=args.skip_bg,
            verbose=not args.quiet,
        )

        # Generate previews
        if args.preview:
            base = args.preview_path or os.path.splitext(args.output)[0]

            if args.preview in ('png', 'both'):
                png_path = base + '_preview.png' if not args.preview_path \
                    else base
                generate_preview(pattern, png_path)
                if not args.quiet:
                    print(f"Preview (PNG): {png_path}", file=sys.stderr)

            if args.preview in ('svg', 'both'):
                svg_path = base + '_preview.svg' if not args.preview_path \
                    else base.replace('.png', '.svg')
                generate_svg_preview(pattern, svg_path)
                if not args.quiet:
                    print(f"Preview (SVG): {svg_path}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
