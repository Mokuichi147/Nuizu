#!/usr/bin/env python3
"""photo2stitch のCLIエントリーポイント。"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Literal

import typer


PaletteName = Literal["janome", "brother", "madeira", "generic"]
PreviewFormat = Literal["png", "svg", "both"]

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "写真を刺繍用データ（DST / PES / JEF）へ変換します。\n\n"
        "対応出力形式:\n"
        "  .dst   Tajima（高い互換性）\n"
        "  .pes   Brother / Babylock\n"
        "  .jef   JANOME\n\n"
        "使用例:\n"
        "  photo2stitch photo.jpg design.dst\n"
        "  photo2stitch photo.png design.jef --colors 12 --width 150\n"
        "  photo2stitch photo.jpg design.pes --auto-angle --pull-comp 0.3\n"
        "  photo2stitch photo.jpg design.dst --palette brother --preview svg"
    ),
)


@app.command()
def convert(
    input_path: str = typer.Argument(
        ...,
        metavar="入力画像",
        help="入力画像ファイル（JPG, PNG など）",
    ),
    output_path: str = typer.Argument(
        ...,
        metavar="出力ファイル",
        help="出力刺繍ファイル（.dst, .pes, .jef）",
    ),
    width: float = typer.Option(
        100.0,
        "--width",
        "-W",
        help="刺繍の目標幅（mm）",
    ),
    height: float | None = typer.Option(
        None,
        "--height",
        "-H",
        help="刺繍の目標高さ（mm）。未指定時は自動計算",
    ),
    colors: int = typer.Option(
        8,
        "--colors",
        "-c",
        help="使用する糸色数",
    ),
    free_colors: bool = typer.Option(
        False,
        "--free-colors",
        help="実糸パレットへのスナップを無効化",
    ),
    palette: PaletteName = typer.Option(
        "janome",
        "--palette",
        help="糸パレット（janome / brother / madeira / generic）",
    ),
    thread_width: float = typer.Option(
        0.4,
        "--thread-width",
        "-t",
        help="糸幅（mm）",
    ),
    density: float | None = typer.Option(
        None,
        "--density",
        "-d",
        help="フィル行間隔（mm）。未指定時は --thread-width と同値",
    ),
    stitch_length: float = typer.Option(
        3.0,
        "--stitch-length",
        "-l",
        help="最大ステッチ長（mm）",
    ),
    angle: float = typer.Option(
        45.0,
        "--angle",
        "-a",
        help="フィル角度（度）",
    ),
    auto_angle: bool = typer.Option(
        False,
        "--auto-angle",
        help="領域ごとにフィル角度を自動最適化",
    ),
    no_underlay: bool = typer.Option(
        False,
        "--no-underlay",
        help="アンダーレイ（下縫い）を無効化",
    ),
    no_outline: bool = typer.Option(
        False,
        "--no-outline",
        help="アウトライン縫いを無効化",
    ),
    satin_outline: bool = typer.Option(
        False,
        "--satin-outline",
        help="アウトラインをサテンステッチ化",
    ),
    pull_comp: float = typer.Option(
        0.0,
        "--pull-comp",
        help="プルコンペンセーション（mm）。推奨: 0.2-0.5",
    ),
    blur: int = typer.Option(
        3,
        "--blur",
        help="前処理ぼかし半径（0で無効）",
    ),
    min_region: float = typer.Option(
        0.001,
        "--min-region",
        help="最小領域比率",
    ),
    enhance: bool = typer.Option(
        True,
        "--enhance/--no-enhance",
        help="写真強調前処理の有効/無効",
    ),
    auto_crop: bool = typer.Option(
        False,
        "--auto-crop",
        help="主被写体を自動検出してクロップ",
    ),
    skip_bg: bool = typer.Option(
        False,
        "--skip-bg",
        help="背景色のステッチをスキップ",
    ),
    strict_colors: bool = typer.Option(
        False,
        "--strict-colors",
        help="-cで指定した色数を厳密に維持",
    ),
    preview: PreviewFormat | None = typer.Option(
        None,
        "--preview",
        help="プレビュー生成形式（png / svg / both）",
    ),
    preview_path: str | None = typer.Option(
        None,
        "--preview-path",
        help="プレビュー出力先のベースパス",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="進行メッセージを抑制",
    ),
) -> None:
    """写真から刺繍データを生成します。"""
    in_file = Path(input_path)
    if not in_file.exists() or not in_file.is_file():
        typer.echo(f"エラー: 入力ファイルが見つかりません: {input_path}", err=True)
        raise typer.Exit(code=1)

    ext = Path(output_path).suffix.lower()
    if ext not in (".dst", ".pes", ".jef"):
        typer.echo(
            f"エラー: 未対応の出力形式です: {ext or '(拡張子なし)'}",
            err=True,
        )
        typer.echo("対応形式: .dst, .pes, .jef", err=True)
        raise typer.Exit(code=1)

    fill_density = density if density is not None else thread_width

    from .pipeline import convert_photo_to_embroidery, generate_preview
    from .svg_preview import generate_svg_preview

    try:
        pattern = convert_photo_to_embroidery(
            image_path=input_path,
            output_path=output_path,
            target_width_mm=width,
            target_height_mm=height,
            n_colors=colors,
            use_thread_palette=not free_colors,
            thread_brand=palette,
            fill_density=fill_density,
            stitch_length=stitch_length,
            fill_angle=angle,
            auto_angle=auto_angle,
            underlay=not no_underlay,
            outline=not no_outline,
            outline_satin=satin_outline,
            pull_compensation=pull_comp,
            thread_width=thread_width,
            blur_radius=blur,
            min_region_ratio=min_region,
            enhance_photo=enhance,
            auto_crop=auto_crop,
            skip_background=skip_bg,
            strict_colors=strict_colors,
            verbose=not quiet,
        )

        if preview:
            base = preview_path or os.path.splitext(output_path)[0]

            if preview in ("png", "both"):
                thread_path = base if preview_path else f"{base}_preview.png"
                generate_preview(pattern, thread_path, thread_width_mm=thread_width)
                if not quiet:
                    typer.echo(f"プレビュー（糸幅）: {thread_path}", err=True)

                stitch_path = f"{base}_stitches.png"
                generate_preview(pattern, stitch_path, thread_width_mm=0)
                if not quiet:
                    typer.echo(f"プレビュー（針落ち）: {stitch_path}", err=True)

            if preview in ("svg", "both"):
                if preview_path:
                    custom = Path(base)
                    svg_thread_path = str(
                        custom.with_suffix(".svg")
                        if custom.suffix
                        else Path(f"{base}.svg")
                    )
                else:
                    svg_thread_path = f"{base}_preview.svg"
                generate_svg_preview(pattern, svg_thread_path, thread_width_mm=thread_width)
                if not quiet:
                    typer.echo(f"プレビュー（SVG 糸幅）: {svg_thread_path}", err=True)

                svg_stitch_path = f"{base}_stitches.svg"
                generate_svg_preview(pattern, svg_stitch_path, thread_width_mm=0)
                if not quiet:
                    typer.echo(f"プレビュー（SVG 針落ち）: {svg_stitch_path}", err=True)

    except Exception as exc:
        typer.echo(f"エラー: {exc}", err=True)
        traceback.print_exc()
        raise typer.Exit(code=1) from exc


def main() -> None:
    """pyproject の scripts から呼ばれるエントリーポイント。"""
    app(prog_name="photo2stitch")


if __name__ == "__main__":
    main()
