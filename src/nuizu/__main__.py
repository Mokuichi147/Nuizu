"""nuizu のCLIエントリーポイント。"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Literal

import typer
import yaml

from importlib import resources

from .palettes import parse_thread_width as _parse_thread_width, BUILTIN_PALETTES


def _list_builtin_sets() -> list[str]:
    """組み込みセットテンプレート名の一覧を返す。"""
    ref = resources.files("nuizu.palettes").joinpath("sets")
    try:
        return sorted(p.name.removesuffix(".yaml") for p in ref.iterdir()
                      if p.name.endswith(".yaml"))
    except Exception:
        return []


def _read_builtin_set(name: str) -> str:
    """組み込みセットテンプレートの内容を返す。"""
    ref = resources.files("nuizu.palettes").joinpath(f"sets/{name}.yaml")
    return ref.read_text(encoding="utf-8")


PreviewFormat = Literal["png", "svg", "none"]


def parse_thread_width(value: str) -> float:
    """糸幅の指定を解析する（typer.BadParameter でラップ）。"""
    try:
        return _parse_thread_width(value)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

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
    stitch_length: float,
    angle: float,
    auto_angle: bool,
    no_underlay: bool,
    no_outline: bool,
    satin_outline: bool,
    pull_comp: float,
    blur: int,
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

    # パレット指定のバリデーション
    if palette not in ("auto", *BUILTIN_PALETTES):
        palette_path = Path(palette)
        if not palette_path.is_file():
            builtin_list = ", ".join(("auto", *BUILTIN_PALETTES))
            typer.echo(
                f"エラー: パレット '{palette}' が見つかりません。\n"
                f"  組み込みパレット: {builtin_list}\n"
                f"  またはYAMLファイルのパスを指定してください",
                err=True,
            )
            raise typer.Exit(code=1)

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
            fill_density=thread_width_mm,
            stitch_length=stitch_length,
            fill_angle=angle,
            auto_angle=auto_angle,
            underlay=not no_underlay,
            outline=not no_outline,
            outline_satin=satin_outline,
            pull_compensation=pull_comp,
            thread_width=thread_width_mm,
            blur_radius=blur,
            auto_crop=auto_crop,
            skip_background=skip_bg,
            strict_colors=strict,
            verbose=not quiet,
        )

        if not quiet:
            typer.echo(f"選択された色 ({len(pattern.colors)}色):", err=True)
            for i, c in enumerate(pattern.colors, start=1):
                hex_code = f"#{c.r:02X}{c.g:02X}{c.b:02X}"
                label = f"  {i:2}. {hex_code}"
                if c.catalog_number:
                    label += f"  ({c.catalog_number})"
                if c.name:
                    label += f"  {c.name}"
                typer.echo(label, err=True)

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
        palette: str = typer.Option(
            "auto", "--palette", "-p",
            help="糸パレット（組み込み名またはYAMLファイルパス）。"
                 "組み込み: janome / brother / madeira / generic",
        ),
        thread_width: str = typer.Option(
            "#50", "--thread-width", "-t",
            help="糸幅。番手（例: #50）または mm（例: 0.14）で指定",
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
            thread_width=thread_width,
            stitch_length=stitch_length, angle=angle,
            auto_angle=auto_angle, no_underlay=no_underlay,
            no_outline=no_outline, satin_outline=satin_outline,
            pull_comp=pull_comp, blur=blur,
            auto_crop=auto_crop, skip_bg=not background,
            preview=preview if preview != "none" else None, quiet=quiet,
        )

    command.__doc__ = description
    return command


app.command("dst")(_make_command(".dst", "DST形式（Tajima）で出力"))
app.command("pes")(_make_command(".pes", "PES形式（Brother / Babylock）で出力"))
app.command("jef")(_make_command(".jef", "JEF形式（JANOME）で出力"))

palette_app = typer.Typer(help="パレット管理コマンド")
app.add_typer(palette_app, name="palette")


_PALETTE_TEMPLATE = """\
# 私の刺繍糸コレクション
#
# 各フィールドの説明:
#   color:        色番号（brandと併用）または hex カラーコード（"#RRGGBB"）
#   brand:        janome / brother / madeira（色番号で指定する場合）
#   thread-width: 番手（例: "#40"）または mm（例: 0.14）。省略時は --thread-width の値を使用
#   material:     polyester / rayon / nylon / cotton。省略時は polyester
#   comment:      自由メモ
threads:
  # 色番号で指定する例（brand が必要）
  - brand: janome
    color: 225
    thread-width: "#40"
    comment: JANOME赤

  - brand: brother
    color: 800
    comment: Brother赤

  # hex カラーコードで指定する例
  - color: "#E60012"
    thread-width: 0.14
    material: rayon
    comment: 手持ちの赤糸

  - color: "#00FF00"
    comment: 緑
"""


@palette_app.command("init")
def init_palette(
    output_path: str = typer.Argument(
        "palette.yaml",
        metavar="出力ファイル",
        help="生成するパレットYAMLのパス（省略時: palette.yaml）",
    ),
    sets: list[str] = typer.Option(
        [],
        "--set", "-s",
        help="ベースにするセット名（複数指定可）。省略時はサンプルテンプレートを生成",
        show_default=False,
    ),
    list_sets: bool = typer.Option(
        False, "--list", "-l",
        help="利用可能なセット一覧を表示して終了",
    ),
) -> None:
    """パレットYAMLテンプレートを生成する。

    セット名を指定すると、そのセットの全色が記入済みのテンプレートを生成します。
    複数指定すると結合されます。

    例:
      nuizu palette init --list
      nuizu palette init --set brother
      nuizu palette init --set brother --set janome my_palette.yaml
    """
    if list_sets:
        available = _list_builtin_sets()
        typer.echo("利用可能なセット:")
        for name in available:
            typer.echo(f"  {name}")
        raise typer.Exit()

    p = Path(output_path)
    if p.exists():
        typer.echo(f"エラー: ファイルが既に存在します: {output_path}", err=True)
        raise typer.Exit(code=1)

    if sets:
        available = _list_builtin_sets()
        merged_comments: list[str] = []
        merged_threads: list[dict] = []
        for name in sets:
            if name not in available:
                typer.echo(
                    f"エラー: セット '{name}' が見つかりません。"
                    " `nuizu palette init --list` で一覧を確認してください",
                    err=True,
                )
                raise typer.Exit(code=1)
            text = _read_builtin_set(name)
            data = yaml.safe_load(text)
            # 先頭コメント行を保持
            for line in text.splitlines():
                if line.startswith("#"):
                    merged_comments.append(line)
                else:
                    break
            if isinstance(data, dict) and isinstance(data.get("threads"), list):
                merged_threads.extend(data["threads"])
        content = "\n".join(merged_comments) + "\n" if merged_comments else ""
        content += yaml.dump(
            {"threads": merged_threads},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )
    else:
        content = _PALETTE_TEMPLATE

    p.write_text(content, encoding="utf-8")
    typer.echo(f"パレットテンプレートを生成しました: {output_path}", err=True)


def main() -> None:
    """pyproject の scripts から呼ばれるエントリーポイント。"""
    app(prog_name="nuizu")


if __name__ == "__main__":
    main()
