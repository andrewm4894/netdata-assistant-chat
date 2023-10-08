#%%
import streamlit as st
from streamlit_feedback import streamlit_feedback
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
import os
import sys
import base64
from dotenv import load_dotenv


load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)

#%%

filename_fn = lambda filename: {"file_name": filename}

#%%

documents = SimpleDirectoryReader(
    "./data_v4/", recursive=True, file_metadata=filename_fn
).load_data()

#%%

parser = SimpleNodeParser.from_defaults()

#%%

nodes = parser.get_nodes_from_documents(documents)

#%%

service_context = ServiceContext.from_defaults(
    llm=OpenAI(
        model=openai_model,
        temperature=0.5,
        system_prompt="You are Netdata Assistant, an expert in all things related to Netdata. Respond in an intelligent and professional manner with clear and detailed answers and examples. Answer based on facts – do not hallucinate. If you do not know the answer, point the user to https://community.netdata.cloud/",
    )
)

#%%

index = VectorStoreIndex(
    nodes,
    service_context=service_context
)

#%%

chat_engine = index.as_chat_engine(chat_mode="condense_question")

#%%

query_engine = index.as_query_engine()

#%%

prompt = "can netdata monitor rest api end points?"

#%%

chat_response = chat_engine.chat(prompt)
print(chat_response)

#%%

query_response = query_engine.query(prompt)
print(query_response)

#%%
#%%