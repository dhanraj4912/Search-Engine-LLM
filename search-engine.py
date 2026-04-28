import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

# Tools setup
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun()

# UI
st.title("LangChain Search Agent (Groq)")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Search the web and contents"}
    ]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

# Input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter Groq API key")
        st.stop()

    # Set API key
    os.environ["GROQ_API_KEY"] = api_key

    # Groq LLM
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",  
        temperature=0.7
    )

    tools = [search, wiki, arxiv]

    agent = create_agent(
        model=llm,
        tools=tools
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

        response = agent.invoke(
            {"messages": st.session_state.messages},
            callbacks=[st_cb]
        )

        output = response["messages"][-1].content

        st.session_state.messages.append({"role": "assistant", "content": output})
        st.write(output)