import os
import sys

import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings import OpenAIEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import PineconeVectorStore
from retriever import PineconeRetriever
from streamlit_feedback import streamlit_feedback
from utils import query


load_dotenv()

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)
pinecone_api_key = os.getenv("PINECONE_API_KEY", st.secrets.pinecone_api_key)

# Initialize OpenAI Embedding
embed_model = OpenAIEmbedding()

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

st.set_page_config(
    page_title="Netdata Assistant",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("⚡Netdata Assistant")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Netdata!"}
    ]


@st.cache_resource(show_spinner=False)
def load_query_engine():
    with st.spinner(
        text="Loading and indexing the latest Netdata docs – hang tight! This should take 1-2 minutes."
    ):
        index = pinecone.Index("netdata")
        vector_store = PineconeVectorStore(pinecone_index=index)

        # define retriever
        retriever = PineconeRetriever(
            vector_store, embed_model, query_mode="default", similarity_top_k=5
        )
        # define query engine
        query_engine = RetrieverQueryEngine.from_args(retriever)

        return query_engine


query_engine = load_query_engine()

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    sys.stdout.write(f"Question: {prompt}\n")

# Check if the Reset button is clicked
if st.button("Reset"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Netdata!"}
    ]

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query(prompt, query_engine)
            file_name = response.source_nodes[0].node.metadata["file_name"]
            url = None

            if "data_v4" in file_name:
                with open(file_name, "r", encoding="utf8") as file:
                    content = file.read()
                    for line in content.splitlines():
                        # print(line)
                        if line.startswith("learn_link:"):
                            url = line.split(" ")[1].strip()
                            break

            st.write(response.response)

            # Check if url is not None before writing it
            if url:
                st.write("Read more: ", url)
            message = {"role": "assistant", "content": response.response}
            feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start")

            # Add response to message history
            st.session_state.messages.append(message)
            sys.stdout.write(f"Response: {message['content']}\n")
