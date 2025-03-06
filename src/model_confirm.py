import openai
import os
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# 読み込めているか確認
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("環境変数 OPENAI_API_KEY が設定されていません。")

# OpenAIのクライアントを作成
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 使用可能なモデル一覧を取得
models = client.models.list()

# モデル名を表示
for model in models.data:
    print(model.id)
