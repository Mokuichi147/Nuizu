"""刺繍糸カラーパレットの読み込み。

組み込みパレット（janome/brother/madeira/generic）および
ユーザー定義のYAMLパレットを統一的に読み込む。
"""

import csv
import io
import math
import re
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# 組み込みパレット名
BUILTIN_PALETTES = (
    # 刺繍ミシンメーカー
    "janome",
    "brother",
    "brother-country",       # 色番号衝突のためマージ不可
    # Madeira
    "madeira",
    # Isacord / Mettler / Sulky など主要ミシン刺繍糸
    "isacord-polyester",
    "isafil-rayon",
    "isalon-polyester",
    "mettler-embroidery",
    "mettler-poly-sheen",    # 色番号衝突のためマージ不可
    "sulky-polyester",
    "sulky-rayon",           # 色番号衝突のためマージ不可
    # Coats / Anchor / DMC
    "anchor",
    "dmc",
    "coats-alcazar",
    "coats-alcazar-jazz",    # 色番号衝突のためマージ不可
    "coats-sylko",
    "coats-sylko-usa",       # 色番号衝突のためマージ不可
    # Robison-Anton（マージ済み）
    "robison-anton",
    # Marathon
    "marathon-polyester",
    "marathon-rayon",
    "marathon-rayon-v3",     # 色番号衝突のためマージ不可
    # Aurifil
    "aurifil-lana",
    "aurifil-mako",
    "aurifil-polyester",
    "aurifil-rayon",
    "aurifil-royal",         # 色番号衝突のためマージ不可
    # ARC
    "arc-polyester",
    "arc-rayon",             # 色番号衝突のためマージ不可
    # Admelody（マージ済み）
    "admelody",
    # BFC / Brildor（マージ済み）
    "bfc-polyester",
    "brildor",
    # Brothread / Simthread（Simthreadはマージ済み）
    "brothread-40",
    "brothread-80",          # 色番号衝突のためマージ不可
    "simthread",
    # その他
    "embroidex",
    "emmel",
    "fil-tec-glide",
    "floriani-polyester",
    "fufu",                  # マージ済み
    "gunold-polyester",
    "gutermann-creativ-dekor",
    "hemingworth",
    "king-star",
    "magnifico",
    "metro",
    "mtb-embroidex",
    "outback-embroidery-rayon",
    "poly-x40",
    "princess",
    "radiant-rayon",
    "ral",
    "royal",                 # マージ済み
    "sigma",
    "swist-rayon",
    "threadart",
    "tristar",               # マージ済み
    "viking-palette",
    "vyapar-rayon",
    "wonderfil-polyester",
    "wonderfil-rayon",       # 色番号衝突のためマージ不可
    # 汎用
    "generic",
)

# 材質 → 密度 (g/cm³) のマッピング
MATERIAL_DENSITY: Dict[str, float] = {
    "polyester": 1.38,
    "rayon": 1.52,
    "nylon": 1.14,
    "cotton": 1.54,
}

# デフォルト材質
DEFAULT_MATERIAL = "polyester"


def parse_thread_width(
    value: str | int | float,
    material: str = DEFAULT_MATERIAL,
) -> float:
    """糸幅の指定を解析する。

    ``#50`` や ``50`` のような番手（メートル番手 Nm）表記と
    ``0.14`` のような mm 直接指定の両方に対応する。

    番手からの変換式: ``d (mm) = √(4 / (π × ρ × Nm))``
    ρ は material に応じた繊維密度を使用。

    Raises:
        ValueError: 不正な形式や値が指定された場合。
    """
    density = MATERIAL_DENSITY.get(material.lower(), MATERIAL_DENSITY[DEFAULT_MATERIAL])

    s = str(value).strip()
    # "#50" 形式
    m = re.fullmatch(r"#(\d+(?:\.\d+)?)", s)
    if m:
        nm = float(m.group(1))
        if nm <= 0:
            raise ValueError(f"番手は正の数を指定してください: {s}")
        return math.sqrt(4.0 / (math.pi * density * nm))
    # 数値
    try:
        num = float(s)
    except ValueError:
        raise ValueError(
            f"糸幅は mm（例: 0.14）または番手（例: #50 / 50）で指定してください: {s}"
        )
    if num <= 0:
        raise ValueError(f"糸幅は正の数を指定してください: {s}")
    # 整数値 ≥ 1 は番手とみなす（mm直接指定は通常 1 未満）
    if num >= 1 and num == int(num):
        return math.sqrt(4.0 / (math.pi * density * num))
    return num


def _parse_hex_color(hex_str: str) -> Tuple[int, int, int]:
    """``#RRGGBB`` 形式の色文字列を (R, G, B) に変換する。"""
    hex_str = hex_str.strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_str):
        raise ValueError(
            f"色は #RRGGBB 形式で指定してください: {hex_str}"
        )
    return (
        int(hex_str[1:3], 16),
        int(hex_str[3:5], 16),
        int(hex_str[5:7], 16),
    )


# ----------------------------------------------------------------
# 組み込みCSV読み込み（内部テーブル用）
# ----------------------------------------------------------------

def _builtin_csv_path(name: str) -> str:
    """組み込みパレットのCSVファイルパスを返す。"""
    ref = resources.files("nuizu.palettes").joinpath(f"colors/{name}.csv")
    return str(ref)


def _load_builtin_csv(
    name: str,
) -> List[Tuple[int, int, int, str]]:
    """組み込みCSVを ``(R, G, B, comment)`` のリストとして読み込む。"""
    p = Path(_builtin_csv_path(name))
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (*_parse_hex_color(row["color"]), row.get("comment", ""))
            for row in reader
        ]


def _normalize_number(number: str) -> str:
    """色番号の先頭ゼロを除去して正規化する（例: "0310" → "310"）。"""
    return str(int(number)) if number.isdigit() else number


def _load_builtin_lookup(name: str) -> Dict[str, Tuple[int, int, int, str]]:
    """組み込みCSVから 色番号→(R, G, B, comment) のルックアップ辞書を構築する。

    先頭ゼロは除去して登録する（例: "0310" → "310"）。
    """
    p = Path(_builtin_csv_path(name))
    lookup: Dict[str, Tuple[int, int, int, str]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            number = row.get("number", "").strip()
            if number:
                key = _normalize_number(number)
                r, g, b = _parse_hex_color(row["color"])
                lookup[key] = (r, g, b, row.get("comment", ""))
    return lookup


# ----------------------------------------------------------------
# YAML パレット読み込み
# ----------------------------------------------------------------

def _decode_file(path: str) -> str:
    """ファイルを UTF-8 (BOM付き含む) または CP932 で読み取る。"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(
        f"パレットファイルのエンコーディングを判定できません: {path}\n"
        "  UTF-8 または Shift_JIS で保存してください"
    )


def load_yaml_palette(
    path: str,
) -> List[Tuple[int, int, int, str, float, str]]:
    """YAMLパレットファイルを読み込む。

    Returns:
        ``(R, G, B, comment, width_mm, catalog_number)`` のリスト。
        ``width_mm`` が 0.0 のとき「グローバルのデフォルトを使用」を意味する。
        ``catalog_number`` は ``"{brand} {number}"`` 形式。ブランド指定なしの場合は空文字。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"パレットファイルが見つかりません: {path}")

    text = _decode_file(path)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML構文エラー: {e}") from e

    if not isinstance(data, dict) or "threads" not in data:
        raise ValueError(
            "パレットYAMLに 'threads' キーがありません"
        )

    threads = data["threads"]
    if not isinstance(threads, list) or not threads:
        raise ValueError("パレットYAMLに糸が登録されていません")

    # ブランドのルックアップテーブルをキャッシュ
    brand_lookups: Dict[str, Dict[str, Tuple[int, int, int, str]]] = {}

    entries: List[Tuple[int, int, int, str, float, str]] = []

    for i, item in enumerate(threads, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"threads {i}番目: 辞書形式で記述してください")

        # color (必須)
        color_raw = item.get("color")
        if color_raw is None:
            raise ValueError(f"threads {i}番目: color が必要です")
        color_str = str(color_raw).strip()

        # brand (省略可)
        brand = str(item.get("brand", "")).strip().lower()

        # material (省略可)
        material = str(item.get("material", "")).strip().lower() or DEFAULT_MATERIAL
        if material not in MATERIAL_DENSITY:
            valid = ", ".join(sorted(MATERIAL_DENSITY))
            raise ValueError(
                f"threads {i}番目: material '{material}' は不正です。"
                f" 有効な値: {valid}"
            )

        # 色の解決
        if brand:
            # ブランド色番号 → hex ルックアップ
            if brand not in BUILTIN_PALETTES:
                raise ValueError(
                    f"threads {i}番目: brand '{brand}' は不正です。"
                    f" 有効な値は --palette オプションで確認できる組み込みパレット名です"
                )
            if brand not in brand_lookups:
                brand_lookups[brand] = _load_builtin_lookup(brand)
            lookup = brand_lookups[brand]
            lookup_key = _normalize_number(color_str) if color_str.isdigit() else color_str
            if lookup_key not in lookup:
                raise ValueError(
                    f"threads {i}番目: {brand} に色番号 '{color_str}' が"
                    " 見つかりません"
                )
            r, g, b, fallback_comment = lookup[lookup_key]
        else:
            # hex カラーコード
            try:
                r, g, b = _parse_hex_color(color_str)
            except ValueError as e:
                raise ValueError(f"threads {i}番目: {e}") from e
            fallback_comment = ""

        # thread-width (省略可)
        tw_raw = item.get("thread-width")
        width_mm = 0.0
        if tw_raw is not None:
            try:
                width_mm = parse_thread_width(tw_raw, material)
            except ValueError as e:
                raise ValueError(f"threads {i}番目: {e}") from e

        # comment (省略可)
        comment = str(item.get("comment", "")).strip() or fallback_comment

        # catalog_number: ブランド指定時は "{brand} {number}" 形式
        catalog_number = f"{brand} {color_str}" if brand else ""

        entries.append((r, g, b, comment, width_mm, catalog_number))

    return entries


# ----------------------------------------------------------------
# 組み込みCSVパレット読み込み（--palette janome 等向け）
# ----------------------------------------------------------------

def _load_builtin_palette(
    name: str,
) -> List[Tuple[int, int, int, str, float, str]]:
    """組み込みCSVを ``(R, G, B, comment, width_mm=0.0, catalog_number)`` として読み込む。"""
    p = Path(_builtin_csv_path(name))
    entries: List[Tuple[int, int, int, str, float, str]] = []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r, g, b = _parse_hex_color(row["color"])
            comment = row.get("comment", "")
            number = row.get("number", "").strip()
            catalog_number = f"{name} {number}" if number else ""
            entries.append((r, g, b, comment, 0.0, catalog_number))
    if not entries:
        raise ValueError(f"組み込みパレットが空です: {name}")
    return entries


# ----------------------------------------------------------------
# 統合エントリーポイント
# ----------------------------------------------------------------

def get_palette(
    name: str = "janome",
) -> List[Tuple[int, int, int, str, float, str]]:
    """パレットを取得する。

    組み込み名（janome/brother/madeira/generic）またはYAMLファイルパスを
    指定できる。

    Returns:
        ``(R, G, B, comment, width_mm, catalog_number)`` のリスト。
        ``catalog_number`` は ``"{brand} {number}"`` 形式。なければ空文字。
    """
    key = name.lower()
    if key in BUILTIN_PALETTES:
        return _load_builtin_palette(key)

    # ファイルパスとして扱う
    return load_yaml_palette(name)
