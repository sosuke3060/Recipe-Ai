import streamlit as st
import os
import json
from dotenv import load_dotenv

# LangChain系
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

# watsonx.ai Service
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
ingredients = st.text_input("手持ちの食材を入力してください (カンマ区切り)")
cooking_time = st.slider("調理にかけられる時間 (分)", min_value=5, max_value=120, step=5)

# 提案ボタンをクリックした場合の処理
if st.button("提案を見る"):
    st.write("AIが献立を考え中...")

    # LangChainとwatsonx.aiの連携
    prompt = PromptTemplate(
        input_variables=["activity_level", "ingredients", "cooking_time"],
        template="""運動量が{activity_level}の人向けに、以下の食材を使った献立を提案してください:
        食材: {ingredients}
        調理時間: {cooking_time}分以内"""
    )

    query = prompt.format(
        activity_level=activity_level,
        ingredients=ingredients,
        cooking_time=cooking_time
    )

    try:
        # response = llm(query)
        # st.write(response)
        response = llm.generate([query])
        st.write(response.generations[0][0].text)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# チャット機能の実装
st.subheader("AIとチャットしてみましょう")
user_message = st.text_input("AIに質問してみてください", key="chat_input")
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
