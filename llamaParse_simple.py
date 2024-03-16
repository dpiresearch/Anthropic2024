import os
from llama_parse import LlamaParse  # pip install llama-parse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

# Using LlamaParse to intelligently parse PDFs

parser = LlamaParse(
    api_key=os.environ["LLAMACLOUD_API_KEY"],  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown"  # "markdown" and "text" are available
)

# doc_file = "./uber_10q_march_2022.pdf"
doc_file = "/Users/dpang/Downloads/DanielPang_2023.pdf"
# sync
documents = parser.load_data(doc_file)

# print the first 1000 characters
print(documents[0].text[:1000] + '...')

from llama_index.core.node_parser import MarkdownElementNodeParser

#
# Derive nodes
#
node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

# 
# For debugging
#
# for node in nodes:
#     print(node)
# print(nodes[0])

recursive_index = VectorStoreIndex(nodes=base_nodes+objects)
raw_index = VectorStoreIndex.from_documents(documents)

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = recursive_index.as_query_engine(
    similarity_top_k=15, 
    node_postprocessors=[reranker], 
    verbose=True
)

raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker])

print(len(nodes))
#query = "how is the Cash paid for Income taxes, net of refunds from Supplemental disclosures of cash flow information?"
query = "What jobs has Daniel Pang done?"

#
# Dumb response
#
response_1 = raw_query_engine.query(query)
print("\n***********New LlamaParse+ Basic Query Engine***********")
print(response_1)

#
# More intelligent response
#
response_2 = recursive_query_engine.query(query)
print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
print(response_2)
