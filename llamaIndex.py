import json
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import JSONNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from multion.client import MultiOn
from dotenv import load_dotenv

load_dotenv()

# Initialize MultiOn client
client = MultiOn(api_key=os.getenv("MULTION_KEY"))

def initialize_index():
    # Load JSON data from file
    with open('output_json/response.json', 'r') as file:
        json_data = json.load(file)

    # Convert JSON data to string
    input_json = json.dumps(json_data)

    # Create a Document object from the JSON data
    document = Document(text=input_json)

    # Set up LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.llm = OpenAI(model="gpt-4", temperature=0)
    Settings.node_parser = JSONNodeParser()

    # Create Chroma vector store
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("llamaindex_store")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create VectorStoreIndex
    index = VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_context,
        show_progress=True
    )

    print("Data has been indexed and stored in the Chroma database.")
    return vector_store

def get_query_engine(vector_store):
    # Load the existing index from the Chroma database
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)

    # Set up a retriever
    retriever = VectorIndexRetriever(index=loaded_index)

    # Define a query engine that uses the retriever
    return RetrieverQueryEngine.from_args(retriever=retriever)

def query(query_text, query_engine):
    response = query_engine.query(query_text)
    return str(response)

def amazon_purchase(query_text, query_engine):
    response = query(query_text, query_engine)
    if "buy" in query_text.lower() and "amazon" in query_text.lower():
        cart_response = client.browse(
            cmd=response,
            url="https://amazon.com/",
            local=True,
        )
        return cart_response.message
    return "This query is not for an Amazon purchase."

# Initialize the index and query engine when the module is imported
# vector_store = initialize_index()
# query_engine = get_query_engine(vector_store)

# if __name__ == "__main__":
    # Example usage
    # print(query("What is my favorite sticker?", query_engine))
    # print(amazon_purchase("Buy stephs favorite drink on amazon", query_engine))