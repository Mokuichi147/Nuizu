# Nuizu

![PyPI - License](https://img.shields.io/pypi/l/nuizu?cacheSeconds=0)
![PyPI - Version](https://img.shields.io/pypi/v/nuizu?cacheSeconds=0)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/nuizu?cacheSeconds=0)

写真・イラスト・漫画など、あらゆる画像を刺繍用データファイルに変換するPython CLIツール。
JANOME、Brother、その他主要刺繍機向けの出力形式に対応。

## 対応出力形式

| 形式 | 拡張子 | 対応機種 |
|------|--------|----------|
| Tajima DST | `.dst` | ほぼ全機種対応（最も互換性が高い） |
| Brother PES | `.pes` | Brother / Babylock |
| JANOME JEF | `.jef` | JANOME |

## インストール

[uv](https://docs.astral.sh/uv/) がインストールされていれば、インストール不要でそのまま実行できます。

```bash
uvx nuizu dst photo.jpg
```

## クイックスタート

```bash
# 基本変換（JEF形式、出力パス省略）
uvx nuizu jef photo.jpg

# 12色、最長辺150mmで変換（プレビューはSVG形式に変更）
uvx nuizu jef photo.png design.jef --colors 12 --size 150 --preview svg

# フル機能（最大色数指定、自動角度、プルコンペンセーション、背景あり）
uvx nuizu dst photo.jpg design.dst \
  --max-colors 10 \
  --auto-angle --pull-comp 0.3 -b
```

基本構文:

```bash
uvx nuizu <dst|pes|jef> 入力画像 [出力ファイル]
```

- `出力ファイル`を省略すると、入力画像と同名で拡張子だけが各形式に変わります（例: `photo.jpg` → `photo.dst`）。

## 使用例

### 写真をJANOMEで刺繍する

```bash
uvx nuizu jef portrait.jpg portrait.jef \
  --palette janome --colors 8 --size 100 \
  --auto-angle --pull-comp 0.3
```

### カスタムパレット（手持ちの糸）を使う

手持ちの糸をYAMLファイルで定義して使用できます。

```bash
# テンプレートを生成
uvx nuizu palette init my_threads.yaml

# 特定ブランドのセットをベースにテンプレートを生成
uvx nuizu palette init --set brother-ets39 my_threads.yaml

# 生成したパレットで変換
uvx nuizu pes photo.jpg --palette my_threads.yaml
```

YAMLパレット例:

```yaml
threads:
  # ブランド色番号で指定
  - brand: brother
    color: 800
    thread-width: "#40"
    comment: 赤（Brother No.800）

  # 手持ちの糸をhexで直接指定
  - color: "#3A7D44"
    thread-width: 0.14
    material: rayon
    comment: 緑（手持ち）
```

### 被写体を自動クロップして刺繍

```bash
uvx nuizu dst flower.jpg flower.dst \
  --auto-crop --colors 6
```

### 高品質仕上げ

```bash
uvx nuizu pes logo.png logo.pes \
  --thread-width '#25' --stitch-length 2.5 --satin-outline \
  --pull-comp 0.4 --max-colors 10
```

### シンプルな変換

```bash
uvx nuizu dst photo.jpg \
  --no-underlay --no-outline --colors 4
```

## 全オプション一覧（`dst` / `pes` / `jef` 共通）

### サイズ

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-s, --size` | 100 | 最長辺の目標サイズ (mm) |

### カラー

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-c, --colors` | - | 使用する糸色数（厳密に維持） |
| `-m, --max-colors` | 8 | 最大糸色数（指定数以下に自動調整） |
| `-p, --palette` | auto | 糸パレット。`auto` / 組み込み名 / YAMLファイルパスを指定。組み込み: `janome`, `brother`, `madeira`, `generic` など60+ブランド |

### ステッチ

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-t, --thread-width` | #50 | 糸幅。番手（例: `#50`）または mm（例: `0.14`）。フィル行間隔も同値。YAMLパレットで糸ごとに個別指定も可 |
| `-l, --stitch-length` | 3.0 | 最大ステッチ長 mm |
| `-a, --angle` | 45 | フィルステッチ角度（度） |
| `--auto-angle` | - | 領域ごとに最適フィル角度を自動決定 |
| `--no-underlay` | - | アンダーレイ（下縫い）無効 |
| `--no-outline` | - | アウトライン無効 |
| `--satin-outline` | - | アウトラインをサテンステッチに |
| `--pull-comp MM` | 0 | プルコンペンセーション mm（推奨: 0.2-0.5） |

### 画像処理

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--blur` | 3 | ぼかし半径（0=無効） |
| `--auto-crop` | - | 被写体を自動検出してクロップ |
| `-b, --background` | - | 背景色もステッチに含める（デフォルトはスキップ） |

### 出力

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--preview` | png | プレビュー生成 (`png`, `svg`, `none`) |
| `-q, --quiet` | - | 進行状況メッセージを抑制 |

## プルコンペンセーションの目安

| 生地タイプ | 推奨値 (mm) |
|-----------|-------------|
| 薄い生地（オーガンジー、シルク） | 0.2 - 0.3 |
| 中厚生地（コットン、リネン） | 0.3 - 0.4 |
| 厚い生地（デニム、キャンバス） | 0.4 - 0.6 |

## 制限事項

- 出力ファイルのバイナリ形式はリバースエンジニアリングに基づいています。実機での動作は糸・生地・テンションなど多くの要因に依存します。
- 写真の変換品質は元画像のコントラストと色の明確さに大きく左右されます。
- 非常に大きな刺繍（200mm超）ではステッチ数が数万を超え、実機での刺繍時間が長くなります。

## 処理パイプライン

```
[入力画像]
    │
    ▼
[1] 前処理
    ├── リサイズ（最大800px）
    ├── Gaussianぼかし（ノイズ軽減）
    └── Medianフィルタ（アンチエイリアス・JPEG圧縮ノイズ除去）
    │
    ▼
[2] 色量子化
    ├── LAB色空間でK-meansクラスタリング
    ├── 刺繍糸パレットへスナップ（JANOME/Brother/Madeira）
    ├── 近似色クラスタの自動マージ（ΔE閾値）
    └── 明るい背景の自動検出（前景比率チェック）
    │
    ▼
[3] 領域分割
    ├── ラベルマップ平滑化（多数決投票）
    ├── 暗色ピクセルの復元（輪郭・ストローク保護）
    ├── OpenCV輪郭抽出（RETR_CCOMP: 穴あり対応）
    ├── モルフォロジー処理（小ギャップの閉鎖）
    └── ピクセル座標 → mm座標変換
    │
    ▼
[4] ステッチ順序最適化
    ├── 色グループ化（色変更最小化）
    └── グループ内で最近傍法（ジャンプ最小化）
    │
    ▼
[5] ステッチ生成
    ├── プルコンペンセーション（輪郭拡張）
    ├── 自動フィル角度決定（PCA / minAreaRect）
    ├── 細幅領域のフィルスキップ（アウトラインのみ）
    ├── アンダーレイ（垂直方向・疎）
    ├── メインフィル（スキャンライン走査）
    │   ├── スタガーオフセット中間ステッチ挿入
    │   └── 凹部横断検出・セグメント分割
    ├── フィルコンペンセーション（行端延長）
    ├── アウトライン（ランニング or サテン）
    │   └── 凹部でコンターパスに沿った中間ステッチ自動挿入
    └── ロングステッチ分割（最大7mm）
    │
    ▼
[6] ファイル出力（DST / PES / JEF）
```

## パレット管理コマンド

```bash
# テンプレート生成（サンプル入り）
uvx nuizu palette init

# 利用可能なセット一覧を表示
uvx nuizu palette init --list

# ブランドのセット売り内容をベースに生成
uvx nuizu palette init --set brother-ets39

# 複数セットを結合
uvx nuizu palette init --set brother-ets39 --set brother-ets22 my_palette.yaml
```

### 組み込みセット

| セット名 | 内容 |
|---------|------|
| `brother-ets39` | Brother ウルトラポス 39色セット (ETS39) |
| `brother-ets22` | Brother ウルトラポス 新色22色セット (ETS22) |
| `brother-cts40` | Brother カントリー刺繍糸 40色セット (CTS40) |

## アーキテクチャ

```
src/nuizu/
├── __main__.py      # CLI エントリーポイント（Typer）
├── pipeline.py      # メイン変換パイプライン
├── preprocess.py    # 画像前処理（背景検出、自動クロップ）
├── quantize.py      # 色量子化（K-means / LAB色空間 / クラスタマージ）
├── palettes/
│   ├── __init__.py  # パレット読み込みAPI（組み込み / YAML）
│   ├── colors/      # 60+ブランドの全色CSVパレット
│   └── sets/        # 実売セット単位のYAMLテンプレート
├── segment.py       # 領域分割（OpenCV contour / 穴あり階層抽出）
├── auto_angle.py    # フィル角度自動最適化（PCA / minAreaRect）
├── fill.py          # フィルステッチ生成（ラスタスキャンライン方式 / 凹部検出）
├── compensate.py    # プルコンペンセーション（布引き攣り補正）
├── optimize.py      # ステッチ順序最適化
├── svg_preview.py   # SVGプレビュー生成
└── formats/
    ├── common.py    # 共通データ構造（Stitch, Pattern, ThreadColor）
    ├── dst.py       # Tajima DST ライター
    ├── pes.py       # Brother PES ライター
    └── jef.py       # JANOME JEF ライター
```
