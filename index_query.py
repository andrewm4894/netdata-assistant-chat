#%%

# Standard library imports
import os
import sys
import base64

# Third party imports
import streamlit as st
import openai
from dotenv import load_dotenv
from streamlit_feedback import streamlit_feedback

# Local application/library specific imports
from llama_index import (
    VectorStoreIndex, 
    ServiceContext, 
    Document,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from tqdm import tqdm
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores import VectorStoreQuery
from llama_index.schema import NodeWithScore
from typing import Optional
from llama_index.response.notebook_utils import display_source_node
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.query_engine import RetrieverQueryEngine
from retriever import PineconeRetriever


load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)
pinecone_api_key = os.getenv("PINECONE_API_KEY", st.secrets.pinecone_api_key)

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone.list_indexes()

#%%

pinecone_index = pinecone.Index("netdata")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

#%%

query_str = "can netdata monitor rest api end points?"

#%%

embed_model = OpenAIEmbedding()
query_embedding = embed_model.get_query_embedding(query_str)
retriever = PineconeRetriever(
    vector_store,
    embed_model,
    query_mode="default",
    similarity_top_k=5
)
query_engine = RetrieverQueryEngine.from_args(retriever)

#%%

query_str = "can netdata monitor rest api end points?"

#%%



#%%

response = query_engine.query(query_str)
print(str(response))

#%%

#%%