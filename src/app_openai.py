import streamlit as st
import os
from dotenv import load_dotenv

# LangChain系
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI  # OpenAI用のLLM

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

# サイドバーに入力項目を配置
st.sidebar.title("献立提案アプリ - 入力")
activity_level = st.sidebar.selectbox("今日の運動量は？", ["軽い", "普通", "激しい"])
ingredients = st.sidebar.text_input("手持ちの食材 (カンマ区切り)", value="卵, 牛乳, パン")
cooking_time = st.sidebar.slider("調理可能時間 (分)", min_value=5, max_value=120, step=5, value=30)

# メイン画面のタイトルと説明
st.title("献立提案アプリ")
st.markdown("このアプリは、ユーザーの運動量、手持ちの食材、調理可能時間を考慮して、最適な献立を提案します。")

# セッションに会話メモリが存在しない場合は初期化
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory()

# 献立提案ボタン
if st.sidebar.button("献立を提案する"):
    with st.spinner("AIが献立を考え中です..."):
        prompt = PromptTemplate(
            input_variables=["activity_level", "ingredients", "cooking_time"],
            template="""運動量が{activity_level}の方に向けて、以下の食材と調理時間を考慮し、いくつかの献立を提案してください:
            食材: {ingredients}
            調理時間: {cooking_time}分以内
            各献立について、以下の情報を含めてください:
            - 材料
            - 調理手順
            - 予測される調理時間
            - 摂取カロリーと栄養素の目安
            ※全ての食材を使用する必要はありません。"""
        )

        query = prompt.format(
            activity_level=activity_level,
            ingredients=ingredients,
            cooking_time=cooking_time
        )

        try:
            response = llm.invoke(query)
            st.markdown("### 提案された献立")
            st.markdown(response.content)
            # 会話履歴に保存 (input と output を記録)
            st.session_state.conversation_memory.save_context(
                {"input": query},
                {"output": response.content}
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# チャット機能のセクション
st.markdown("---")
st.subheader("AIとチャットしてみましょう")
user_message = st.text_input("提案された献立について質問してください", key="chat_input")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("送信"):
        if user_message.strip():
            with st.spinner("AIが回答中です..."):
                try:
                    if "conversation" not in st.session_state:
                        st.session_state.conversation = ConversationChain(
                            llm=llm, 
                            memory=st.session_state.conversation_memory
                        )
                    ai_response = st.session_state.conversation.run(user_message)
                    st.markdown("**AIの回答:**")
                    st.markdown(ai_response)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
        else:
            st.warning("質問を入力してください！")

with col2:
    if st.button("チャット履歴をクリア"):
        st.session_state.conversation_memory.clear()
        if "conversation" in st.session_state:
            del st.session_state.conversation
        st.success("チャット履歴をクリアしました。")
