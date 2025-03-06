import streamlit as st
import os
from dotenv import load_dotenv

# LangChain系
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # OpenAI用のLLM

# 環境変数のロード
load_dotenv()

# OpenAI API の設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# 環境変数のエラーチェック
if not OPENAI_API_KEY:
    st.error("環境変数 (OPENAI_API_KEY) が設定されていません。")
    st.stop()

# OpenAI LLMの設定
llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    openai_api_key=OPENAI_API_KEY
)

# アプリの基本構築
st.title("献立提案アプリ")

# ユーザー入力フォーム
activity_level = st.selectbox("今日の運動量は？", ["軽い", "普通", "激しい"])
ingredients = st.text_input("手持ちの食材を入力 (カンマ区切り)")
cooking_time = st.slider("調理にかけられる時間 (分)", min_value=5, max_value=120, step=5)

if st.button("提案を見る"):
    st.write("AIが献立を考え中...")

    prompt = PromptTemplate(
        input_variables=["activity_level", "ingredients", "cooking_time"],
        template="""運動量が{activity_level}人向けに、以下の食材・条件に従い献立をいくつか提案してください:
        食材: {ingredients}
        調理時間: {cooking_time}分以内
        それぞれの献立について"材料"・"調理手順"・"予測される調理時間"・"摂取カロリーと栄養素の目安"を表示してください。
        全ての食材を使う必要はありません。"""
    )

    query = prompt.format(
        activity_level=activity_level,
        ingredients=ingredients,
        cooking_time=cooking_time
    )

    try:
        response = llm.invoke(query)
        st.write("提案された献立:")
        st.write(response.content)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# チャット機能
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
