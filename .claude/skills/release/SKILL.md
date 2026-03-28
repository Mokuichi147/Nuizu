---
name: release
description: Pythonプロジェクトのバージョンアップ、GitHubタグ・リリース作成、PyPI公開を行う
disable-model-invocation: true
argument-hint: "[バージョン (例: 0.3.0)]"
---

Pythonプロジェクトのリリース作業を実行します。

バージョン: $ARGUMENTS（省略時は前回タグからの変更内容を確認してユーザーに提案する）

## 手順

### 1. 事前確認
- `git status` でワーキングツリーがクリーンであることを確認する。未コミットの変更があれば中断してユーザーに報告する
- `git tag --sort=-v:refname | head -1` で現在の最新タグを確認する
- `git log <最新タグ>..HEAD --oneline` でリリースに含まれる変更を把握する

### 2. バージョン更新
- `pyproject.toml` の `version` を更新する
- `uv lock` を実行して `uv.lock` を同期する
- `pyproject.toml` と `uv.lock` をコミットする（メッセージ: `v{VERSION}にバージョンアップ`）

### 3. タグ作成・プッシュ
- `git tag v{VERSION}` でタグを作成する
- `git push origin main && git push origin v{VERSION}` でプッシュする

### 4. ビルド・リリース・公開
- `uv build` で wheel と sdist をビルドする
- `gh release create v{VERSION} --title "v{VERSION}" --notes "..."` でリリースを作成し、`dist/` 内のビルド成果物（`.whl` と `.tar.gz`）を添付する
- リリースノートは **前回タグから今回タグまでの差分のみ** を日本語で記載する。過去バージョンの変更内容を含めないこと
- `uv publish` で PyPI に公開する。`dist/` 内の該当バージョンのファイルのみを指定する（`dist/nuizu-{VERSION}*`）

### 5. 完了報告
- GitHub リリース URL を表示する
- PyPI 公開が成功したことを報告する
