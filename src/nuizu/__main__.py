"""nuizu のCLIエントリーポイント。"""

from __future__ import annotations

import math
import os
import re
import traceback
from pathlib import Path
from typing import Literal

import typer


PaletteName = Literal["auto", "janome", "brother", "madeira", "generic"]
PreviewFormat = Literal["png", "svg", "none"]

# ポリエステル密度 (g/cm³) — 刺繍糸で最も一般的な素材
_POLYESTER_DENSITY = 1.38


def parse_thread_width(value: str) -> float:
    """糸幅の指定を解析する。

    ``#50`` のような番手（メートル番手 Nm）表記と ``0.14`` のような
    mm 直接指定の両方に対応する。

    番手からの変換式: ``d (mm) = √(4 / (π × ρ × Nm))``
    ρ にはポリエステルの密度 1.38 g/cm³ を使用。
    """
    value = value.strip()
    m = re.fullmatch(r"#(\d+(?:\.\d+)?)", value)
    if m:
        nm = float(m.group(1))
        if nm <= 0:
            raise typer.BadParameter(f"番手は正の数を指定してください: {value}")
        return math.sqrt(4.0 / (math.pi * _POLYESTER_DENSITY * nm))
    try:
        mm = float(value)
    except ValueError:
        raise typer.BadParameter(
            f"糸幅は mm（例: 0.14）または番手（例: #50）で指定してください: {value}"
        )
    if mm <= 0:
        raise typer.BadParameter(f"糸幅は正の数を指定してください: {value}")
    return mm

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "写真を刺繍用データ（DST / PES / JEF）へ変換します。\n\n"
        "使用例:\n"
        "  nuizu dst photo.jpg\n"
        "  nuizu dst photo.jpg -c 12 -s 150\n"
        "  nuizu jef photo.png output.jef --max-colors 10\n"
        "  nuizu pes photo.jpg --palette brother -b"
    ),
)


def _run_convert(
    input_path: str,
    output_path: str | None,
    ext: str,
    *,
    size: float,
    colors: int | None,
    max_colors: int,
    palette: str,
    thread_width: str,
    density: float | None,
    stitch_length: float,
    angle: float,
    auto_angle: bool,
    no_underlay: bool,
    no_outline: bool,
    satin_outline: bool,
    pull_comp: float,
    blur: int,
    min_region: float,
    auto_crop: bool,
    skip_bg: bool,
    preview: str | None,
    quiet: bool,
) -> None:
    in_file = Path(input_path)
    if not in_file.exists() or not in_file.is_file():
        typer.echo(f"エラー: 入力ファイルが見つかりません: {input_path}", err=True)
        raise typer.Exit(code=1)

    if output_path is None:
        output_path = str(in_file.with_suffix(ext))

    thread_width_mm = parse_thread_width(thread_width)
    fill_density = density if density is not None else thread_width_mm

    if colors is not None:
        n_colors = colors
        strict = True
    else:
        n_colors = max_colors
        strict = False

    from .pipeline import convert_photo_to_embroidery, generate_preview
    from .svg_preview import generate_svg_preview

    try:
        pattern = convert_photo_to_embroidery(
            image_path=input_path,
            output_path=output_path,
            target_size_mm=size,
            n_colors=n_colors,
            use_thread_palette=palette != "auto",
            thread_brand=palette,
            fill_density=fill_density,
            stitch_length=stitch_length,
            fill_angle=angle,
            auto_angle=auto_angle,
            underlay=not no_underlay,
            outline=not no_outline,
            outline_satin=satin_outline,
            pull_compensation=pull_comp,
            thread_width=thread_width_mm,
            blur_radius=blur,
            min_region_ratio=min_region,
            auto_crop=auto_crop,
            skip_background=skip_bg,
            strict_colors=strict,
            verbose=not quiet,
        )

        if preview:
            base = os.path.splitext(output_path)[0]

            if preview == "png":
                thread_path = f"{base}_preview.png"
                generate_preview(pattern, thread_path, thread_width_mm=thread_width_mm)
                if not quiet:
                    typer.echo(f"プレビュー（糸幅）: {thread_path}", err=True)

                stitch_path = f"{base}_stitches.png"
                generate_preview(pattern, stitch_path, thread_width_mm=0)
                if not quiet:
                    typer.echo(f"プレビュー（針落ち）: {stitch_path}", err=True)

            elif preview == "svg":
                svg_thread_path = f"{base}_preview.svg"
                generate_svg_preview(pattern, svg_thread_path, thread_width_mm=thread_width_mm)
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


def _make_command(ext: str, description: str):
    """サブコマンド生成ファクトリ。"""

    def command(
        input_path: str = typer.Argument(
            ...,
            metavar="入力画像",
            help="入力画像ファイル（JPG, PNG など）",
        ),
        output_path: str | None = typer.Argument(
            None,
            metavar="出力ファイル",
            help=f"出力ファイル（省略時は 入力名{ext}）",
        ),
        size: float = typer.Option(
            100.0, "--size", "-s", help="最長辺の目標サイズ（mm）",
        ),
        colors: int | None = typer.Option(
            None, "--colors", "-c", help="使用する糸色数（厳密に維持）",
        ),
        max_colors: int = typer.Option(
            8, "--max-colors", "-m", help="最大糸色数（指定数以下に自動調整）",
        ),
        palette: PaletteName = typer.Option(
            "auto", "--palette", "-p", help="糸パレット",
        ),
        thread_width: str = typer.Option(
            "#50", "--thread-width", "-t",
            help="糸幅。番手（例: #50）または mm（例: 0.14）で指定",
        ),
        density: float | None = typer.Option(
            None, "--density", "-d", help="フィル行間隔（mm）。未指定時は --thread-width と同値",
        ),
        stitch_length: float = typer.Option(
            3.0, "--stitch-length", "-l", help="最大ステッチ長（mm）",
        ),
        angle: float = typer.Option(
            45.0, "--angle", "-a", help="フィル角度（度）",
        ),
        auto_angle: bool = typer.Option(
            False, "--auto-angle", help="領域ごとにフィル角度を自動最適化",
        ),
        no_underlay: bool = typer.Option(
            False, "--no-underlay", help="アンダーレイ（下縫い）を無効化",
        ),
        no_outline: bool = typer.Option(
            False, "--no-outline", help="アウトライン縫いを無効化",
        ),
        satin_outline: bool = typer.Option(
            False, "--satin-outline", help="アウトラインをサテンステッチ化",
        ),
        pull_comp: float = typer.Option(
            0.0, "--pull-comp", help="プルコンペンセーション（mm）。推奨: 0.2-0.5",
        ),
        blur: int = typer.Option(
            3, "--blur", help="前処理ぼかし半径（0で無効）",
        ),
        min_region: float = typer.Option(
            0.001, "--min-region", help="最小領域比率",
        ),
        auto_crop: bool = typer.Option(
            False, "--auto-crop", help="主被写体を自動検出してクロップ",
        ),
        background: bool = typer.Option(
            False, "--background", "-b", help="背景色もステッチに含める",
        ),
        preview: PreviewFormat = typer.Option(
            "png", "--preview", help="プレビュー生成形式",
        ),
        quiet: bool = typer.Option(
            False, "--quiet", "-q", help="進行メッセージを抑制",
        ),
    ) -> None:
        _run_convert(
            input_path, output_path, ext,
            size=size, colors=colors,
            max_colors=max_colors, palette=palette,
            thread_width=thread_width, density=density,
            stitch_length=stitch_length, angle=angle,
            auto_angle=auto_angle, no_underlay=no_underlay,
            no_outline=no_outline, satin_outline=satin_outline,
            pull_comp=pull_comp, blur=blur, min_region=min_region,
            auto_crop=auto_crop, skip_bg=not background,
            preview=preview if preview != "none" else None, quiet=quiet,
        )

    command.__doc__ = description
    return command


app.command("dst")(_make_command(".dst", "DST形式（Tajima）で出力"))
app.command("pes")(_make_command(".pes", "PES形式（Brother / Babylock）で出力"))
app.command("jef")(_make_command(".jef", "JEF形式（JANOME）で出力"))


def main() -> None:
    """pyproject の scripts から呼ばれるエントリーポイント。"""
    app(prog_name="nuizu")


if __name__ == "__main__":
    main()
