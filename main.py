from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from qdrant_client import QdrantClient

load_dotenv()

embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")
Settings.embed_model = embed_model

llm = GoogleGenAI(model="gemini-3.5-flash-lite")
Settings.llm = llm

client = QdrantClient(path="./local_db")
print(client)

# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents=documents)
# query_engine = index.as_query_engine()

# response = llm.stream_complete("How do I process batch requests? Using LangGraph is it better to batch and send requests? How do I use the batched response? Should I ask the LLM to send me in a particular format for my use?")

# for r in response:
#     print(r.delta, flush=True, end="")