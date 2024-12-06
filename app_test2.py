import streamlit as st
import os
from dotenv import load_dotenv

# LangChain系
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_ibm import WatsonxLLM

# 環境変数のロード
load_dotenv()

# LLMのインスタンス作成
WATSONX_AI_ENDPOINT = os.getenv("WATSONX_AI_ENDPOINT")
WATSONX_AI_PROJECT_ID = os.getenv("WATSONX_AI_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")

# 環境変数のエラーチェック
if not all([WATSONX_AI_ENDPOINT, WATSONX_AI_PROJECT_ID, WATSONX_API_KEY]):
    st.error("環境変数 (WATSONX_AI_ENDPOINT, WATSONX_AI_PROJECT_ID, WATSONX_API_KEY) が設定されていません。")
    st.stop()

# LLMのパラメータ設定
params = {
    "max_new_tokens": 16000,
    "min_new_tokens": 1
}

llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    url=WATSONX_AI_ENDPOINT,
    apikey=WATSONX_API_KEY,
    params=params,
    project_id=WATSONX_AI_PROJECT_ID,
)

# アプリの基本構築
st.title("献立提案アプリ")

# ユーザー入力フォームを作成
activity_level = st.selectbox("今日の運動量は？", ["軽い", "普通", "激しい"])
ingredients = st.text_input("手持ちの食材を入力 (カンマ区切り)")
cooking_time = st.slider("調理にかけられる時間 (分)", min_value=5, max_value=120, step=5)

# 記録用のデータリスト
if "records" not in st.session_state:
    st.session_state["records"] = []

# 提案ボタンをクリックしてからの処理
if st.button("提案を見る"):
    st.write("AIが献立を考え中...")

    # LangChainとwatsonx.aiの連携
    prompt = PromptTemplate(
        input_variables=["activity_level", "ingredients", "cooking_time"],
        template="""運動量が{activity_level}人向けに、以下の食材を使った献立をいくつか提案してください:
        食材: {ingredients}
        調理時間: {cooking_time}分以内
        それぞれの献立について"材料"・"調理手順"・"予測される調理時間"・"摂取カロリーと栄養素の目安"を表示してください。"""
    )

    query = prompt.format(
        activity_level=activity_level,
        ingredients=ingredients,
        cooking_time=cooking_time
    )

    try:
        response = llm.generate([query])
        st.write("提案された献立:")
        recipes = response.generations[0][0].text.split("\n\n")  # 複数の献立を分割

        for i, recipe in enumerate(recipes):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(recipe)

            # 「摂取カロリーと栄養素の目安」を抽出して解析
            if "摂取カロリーと栄養素の目安:" in recipe:
                nutritional_info_raw = recipe.split("摂取カロリーと栄養素の目安:")[1].strip()
                nutritional_info = {}
                for line in nutritional_info_raw.split("\n"):
                    if ":" in line:
                        key, value = line.split(":")
                        nutritional_info[key.strip()] = value.strip()

                # 記録ボタンを右横に表示
                with col2:
                    if st.button(f"記録", key=f"record_button_{i}"):
                        st.session_state["records"].append(nutritional_info)
                        st.success(f"献立 {i + 1} のカロリーと栄養素を記録しました！")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# 記録した内容を表示
st.subheader("記録したカロリーと栄養素")
if st.session_state["records"]:
    for idx, record in enumerate(st.session_state["records"]):
        st.write(f"記録 {idx + 1}:")
        for key, value in record.items():
            st.write(f"{key}: {value}")
else:
    st.write("まだ記録がありません。")



# チャット機能の実装
st.subheader("AIとチャットしてみましょう")
user_message = st.text_input("提案された献立について質問してみてください", key="chat_input")
if st.button("送信"):
    conversation = ConversationChain(llm=llm)

    if user_message.strip():
        try:
            ai_response = conversation.run(user_message)
            st.write(ai_response)
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.warning("質問を入力してください！")

