# photo2stitch v0.2

写真を刺繍用データファイルに変換するPython CLIツール。
JANOME、Brother、その他主要刺繍機向けの出力形式に対応。

## 対応出力形式

| 形式 | 拡張子 | 対応機種 |
|------|--------|----------|
| Tajima DST | `.dst` | ほぼ全機種対応（最も互換性が高い） |
| Brother PES | `.pes` | Brother / Babylock |
| JANOME JEF | `.jef` | JANOME |

## 必要環境

- `uv`
- Python 3.12.10+（`pyproject.toml`準拠）

```bash
uv sync
```

## クイックスタート

```bash
# 基本変換（JANOME JEF形式）
uv run photo2stitch photo.jpg output.jef

# 12色、幅150mmで変換 + プレビュー生成
uv run photo2stitch photo.png design.jef --colors 12 --width 150 --preview both

# フル機能（自動角度、プルコンペンセーション、背景スキップ）
uv run photo2stitch photo.jpg design.dst \
  --auto-angle --pull-comp 0.3 --skip-bg --preview svg
```

## 全オプション一覧

### サイズ

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-W, --width` | 100 | 刺繍幅 (mm) |
| `-H, --height` | 自動 | 刺繍高さ (mm)。未指定時はアスペクト比を維持 |

### カラー

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-c, --colors` | 8 | 糸の色数 (2-32) |
| `--palette` | janome | 糸ブランド (`janome`, `brother`, `madeira`, `generic`) |
| `--free-colors` | - | パレットスナップ無効（量子化色をそのまま使用） |

### ステッチ

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-d, --density` | 0.4 | フィル行間隔 mm（小さい＝密） |
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
| `--no-enhance` | - | 写真強調前処理を無効化 |
| `--auto-crop` | - | 被写体を自動検出してクロップ |
| `--skip-bg` | - | 背景色を自動検出してステッチをスキップ |
| `--blur` | 3 | ぼかし半径（0=無効） |
| `--min-region` | 0.003 | 最小領域サイズ（画像全体に対する比率） |

### 出力

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--preview` | - | プレビュー生成 (`png`, `svg`, `both`) |
| `--preview-path` | - | プレビューファイルのカスタムパス |
| `-q, --quiet` | - | 進行状況メッセージを抑制 |

## 処理パイプライン

```
[入力画像]
    │
    ▼
[1] 画像前処理
    ├── リサイズ（最大800px）
    ├── ノイズ除去（Non-local means）
    ├── コントラスト強調（CLAHE / LAB空間）
    ├── 彩度ブースト（×1.2）
    ├── エッジ保持平滑化（Bilateral filter）
    └── アンシャープマスク
    │
    ▼
[2] 色量子化
    ├── LAB色空間でK-meansクラスタリング
    ├── 刺繍糸パレットへスナップ（JANOME/Brother/Madeira）
    └── 小領域除去（近隣色にマージ）
    │
    ▼
[3] 領域分割
    ├── OpenCV輪郭抽出（穴あり対応）
    ├── モルフォロジー処理（クリーンアップ）
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
    ├── アンダーレイ（垂直方向・疎）
    ├── メインフィル（スキャンライン走査）
    ├── フィルコンペンセーション（行端延長）
    ├── アウトライン（ランニング or サテン）
    └── ロングステッチ分割（最大7mm）
    │
    ▼
[6] ファイル出力（DST / PES / JEF）
```

## アーキテクチャ

```
src/photo2stitch/
├── __main__.py      # CLI エントリーポイント
├── pipeline.py      # メイン変換パイプライン
├── preprocess.py    # 画像前処理（ノイズ除去、コントラスト、エッジ保持平滑化）
├── quantize.py      # 色量子化（K-means / LAB色空間）
├── palettes.py      # 実メーカー糸パレット（JANOME / Brother / Madeira）
├── segment.py       # 領域分割（OpenCV contour）
├── auto_angle.py    # フィル角度自動最適化（PCA / minAreaRect）
├── fill.py          # フィルステッチ生成（ラスタスキャンライン方式）
├── compensate.py    # プルコンペンセーション（布引き攣り補正）
├── optimize.py      # ステッチ順序最適化
├── svg_preview.py   # SVGプレビュー生成
└── formats/
    ├── common.py    # 共通データ構造（Stitch, Pattern, ThreadColor）
    ├── dst.py       # Tajima DST ライター
    ├── pes.py       # Brother PES ライター
    └── jef.py       # JANOME JEF ライター
```

## 使用例

### 写真をJANOMEで刺繍する

```bash
# JANOME JEF形式、8色、幅100mm
uv run photo2stitch portrait.jpg portrait.jef \
  --palette janome --colors 8 --width 100 \
  --auto-angle --pull-comp 0.3 --preview both
```

### 背景なしで被写体だけを刺繍

```bash
# 背景自動検出＋スキップ、自動クロップ
uv run photo2stitch flower.jpg flower.dst \
  --skip-bg --auto-crop --colors 6 --preview png
```

### 密度の高い高品質仕上げ

```bash
# 密度0.25mm、ステッチ長2.5mm、サテンアウトライン
uv run photo2stitch logo.png logo.pes \
  --density 0.25 --stitch-length 2.5 --satin-outline \
  --pull-comp 0.4 --colors 10
```

### 軽量・高速版

```bash
# アンダーレイ・アウトライン・強調処理なし
uv run photo2stitch photo.jpg quick.dst \
  --no-underlay --no-outline --no-enhance --colors 4
```

## プルコンペンセーションの目安

| 生地タイプ | 推奨値 (mm) |
|-----------|-------------|
| 薄い生地（オーガンジー、シルク） | 0.2 - 0.3 |
| 中厚生地（コットン、リネン） | 0.3 - 0.4 |
| 厚い生地（デニム、キャンバス） | 0.4 - 0.6 |

## 制限事項と注意点

- 出力ファイルのバイナリ形式はリバースエンジニアリングに基づいています。実機での動作は糸・生地・テンションなど多くの要因に依存します。
- 写真の変換品質は元画像のコントラストと色の明確さに大きく左右されます。
- 非常に大きな刺繍（200mm超）ではステッチ数が数万を超え、実機での刺繍時間が長くなります。
- 最初はシンプルな画像（ロゴ、イラスト）で試し、仕上がりを確認してからフォトリアルな変換に進むことをお勧めします。
