
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import asyncio

load_dotenv()


# We use simple directory reader to load docs of different types
our_reader = SimpleDirectoryReader(input_dir="/Users/matthew/Documents/elevenlabs")
documents = our_reader.load_data()
print(f"Loaded {len(documents)} documents from the directory.")


# Storing
store = chromadb.PersistentClient(path="/Users/matthew/Documents/elevenlabs/chroma_db")
store_collection = store.get_or_create_collection(name="elevenlabs_docs")
vector_store = ChromaVectorStore(chroma_collection=store_collection)

# We create the Ingestion pipeline (Chunking and embedding the documents for lookup) and add the store
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=4000, chunk_overlap=200),
        NVIDIAEmbedding(model="nvidia/nemotron-3-embed-1b", truncate="END")
    ],
    vector_store=vector_store
)
async def process_documents(documents):
    nodes = await pipeline.arun(documents=documents)
    print(f"Processed {len(nodes)} nodes from the documents.")
    return nodes
asyncio.run(process_documents(documents))


# We index the documents for retrieval
embedding_model = NVIDIAEmbedding(model="nvidia/nemotron-3-embed-1b", truncate="END")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embedding_model=embedding_model)


# Querying the vector store for information
query_engine = index.as_query_engine()