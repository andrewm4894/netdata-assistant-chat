#%%

import os
import streamlit as st
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
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
from retriever import PineconeRetriever
from llama_index.query_engine import RetrieverQueryEngine

load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.openai_api_key)
pinecone_api_key = os.getenv("PINECONE_API_KEY", st.secrets.pinecone_api_key)

embed_model = OpenAIEmbedding()

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone.list_indexes()

#%%

# create index
# pinecone.create_index("netdata", dimension=1536, metric="euclidean")
# get index
pinecone_index = pinecone.Index("netdata")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

#%%

# load documents
documents = SimpleDirectoryReader(
    input_dir="./data_v4/",
    recursive=True,
    file_metadata=lambda filename: {"file_name": filename}
).load_data()

num_docs = len(documents)
print(f"Loaded {num_docs} documents")

#%%

# split documents into text chunks
text_splitter = SentenceSplitter(chunk_size=1024)
text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

num_text_chunks = len(text_chunks)
print(f"Split {num_docs} documents into {num_text_chunks} text chunks")

#%%

# create nodes
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

num_nodes = len(nodes)
print(f"Created {num_nodes} nodes")

#%%

# extract metadata
metadata_extractor_llm = OpenAI(model="gpt-3.5-turbo")
metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=metadata_extractor_llm),
        QuestionsAnsweredExtractor(questions=5, llm=metadata_extractor_llm),
    ],
    in_place=False,
)
nodes = metadata_extractor.process_nodes(nodes)

#%%

# create embeddings for each node
for node in tqdm(nodes, desc='Processing nodes', unit='node'):
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

#%%

# add nodes to index
vector_store.add(nodes)

#%%

from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores import VectorStoreQuery
from llama_index.schema import NodeWithScore
from typing import Optional
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List


class PineconeRetriever(BaseRetriever):
    """Retriever over a pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


#%%

# define retriever
retriever = PineconeRetriever(
    vector_store,
    embed_model,
    query_mode="default",
    similarity_top_k=5
)
# define query engine
query_engine = RetrieverQueryEngine.from_args(retriever)

#%%

# define query
query_str = "can netdata monitor rest api end points?"
# get response
response = query_engine.query(query_str)
print(str(response))

#%%
