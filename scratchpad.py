#%%

import streamlit as st
from streamlit_feedback import streamlit_feedback
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
import os
import sys
import base64
from dotenv import load_dotenv


load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)

filename_fn = lambda filename: {"file_name": filename}

#%%

documents = SimpleDirectoryReader(
    input_dir="./data_v4/",
    recursive=True,
    file_metadata=filename_fn
).load_data()
print(len(documents))

#%%

text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)
text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

print(len(text_chunks))
print(len(doc_idxs))

#%%

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

print(len(nodes))

#%%

from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ],
    in_place=False,
)

#%%

nodes = metadata_extractor.process_nodes(nodes)

#%%

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

#%%

service_context = ServiceContext.from_defaults(
    llm=OpenAI(
        model=openai_model,
        temperature=0.5,
        system_prompt="You are Netdata Assistant, an expert in all things related to Netdata. Respond in an intelligent and professional manner with clear and detailed answers and examples. Answer based on facts – do not hallucinate. If you do not know the answer, point the user to https://community.netdata.cloud/",
    )
)
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



#%%
parser = SimpleNodeParser.from_defaults()

#%%

nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

#%%



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