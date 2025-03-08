# Recipe-Ai
[本アプリケーションのリンク](https://sosuke3060-recipe-ai-srcapp-openai-nhywmb.streamlit.app/)

## プロジェクト概要
ユーザーの運動量、手持ちの食材、調理可能時間を考慮し、最適な献立を提案するアプリの開発

## 主な機能

- ユーザーの運動量、手持ちの食材、調理可能時間を考慮した最適な献立を提案
- 入力された食材を基にしたレシピ提案
- 指定された調理可能時間内で完成できるレシピのみを提案
- AIとの会話による献立の深掘りが可能

## 使用技術

- 言語モデル: IBM watsonx.ai, OpenAI (gpt-4o)
- フロントエンド: Streamlit
- バックエンド処理: LangChain

## アーキテクチャ

![画像の説明](Architecture.png)

## 環境構築方法(OpenAI APIの場合)

### 1. リポジトリをクローン

```bash
git clone https://github.com/sosuke3060/Recipe-Ai.git
```

### 2. ライブラリをインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

プロジェクトのルートディレクトリに以下の内容の`.env`ファイルを作成してください。

```bash
OPENAI_API_KEY=取得したAPIキー
OPENAI_MODEL=gpt-4o
```

### 4. アプリを起動

```bash
streamlit run app_openai.py
```