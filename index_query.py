#%%

import os
import streamlit as st
import openai
from dotenv import load_dotenv
from llama_index.embeddings import OpenAIEmbedding
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from retriever import PineconeRetriever
from utils import query, print_response


load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)
pinecone_api_key = os.getenv("PINECONE_API_KEY", st.secrets.pinecone_api_key)

embed_model = OpenAIEmbedding()

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

index = pinecone.Index("netdata")
vector_store = PineconeVectorStore(pinecone_index=index)

# define retriever
retriever = PineconeRetriever(
    vector_store,
    embed_model,
    query_mode="default",
    similarity_top_k=5
)
# define query engine
query_engine = RetrieverQueryEngine.from_args(retriever)

pinecone.list_indexes()

##%%

# define query
query_str = "can netdata monitor mysql?"
print_response(query(query_str, query_engine))

##%%
