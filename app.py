import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Agent関連
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler

# 会話履歴の保存関連
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

# .envファイルの内容を環境変数に設定
load_dotenv()

st.title("langchain-streamlit-app")

# Agentを作成する関数
def create_agent_chain():
    chat = ChatOpenAI(
        model_name = os.environ["OPENAI_API_MODEL"],
        temperature = os.environ["OPENAI_API_TEMPERATURE"],
        # ストリーミング応答
        streaming = True,
    )
    
    # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    # Memoryを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    
    # ツールとしてddg-searchとWikipediaを用意
    tools = load_tools(["ddg-search", "wikipedia"])
    # Agentの初期化
    return initialize_agent(
        tools, 
        chat, 
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    
# 一度だけAgentを初期化する
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()
    
# st.session_stateにmessageがない場合
if "messages" not in st.session_state:
    # 空のリストで初期化
    st.session_state.messages = []
    
# ループ処理
for message in st.session_state.messages:
    # ロールごとに
    with st.chat_message(message["role"]):
        # 保存されているテキストを表示
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

# 入力された文字列がある場合
if prompt:
    # ユーザーの入力内容をst.session_state.messagesに追加
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    # ユーザーのアイコンで
    with st.chat_message("user"):
        # promptをマークダウンとして整形して表示
        st.markdown(prompt)
    
    # AIのアイコンで
    with st.chat_message("assistant"):
        # chat = ChatOpenAI(
        #     model_name = os.environ["OPENAI_API_MODEL"],
        #     temperature = os.environ["OPENAI_API_TEMPERATURE"],
        # )
        # messages = [HumanMessage(content=prompt)]
        # response = chat(messages)
        # st.markdown(response.content)
        
        # Agentの動作をStreamlitの画面上にストリーミングで表示
        callback = StreamlitCallbackHandler(st.container())
        # agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)
        
    # 応答をst.session_state.messagesに追加
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )