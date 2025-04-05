from langchain_community.chat_models import ChatOpenAI
import langgraph
from langgraph.graph import StateGraph, START, END
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MILVUS_HOST = "https://in03-c1cca92c1dece51.serverless.gcp-us-west1.cloud.zilliz.com"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "beca2579c7f8540bc327b2cfbf948dd5df482aa42e9e58d668d241bd172a0a5194b6b7a76e3bdaf1230f96cfc906dfe153207e6d"

def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text = " ".join([doc.page_content for doc in documents])
    text = text.replace("\n", " ")
    
    return text

def load_pdf_cut(path, first_sep):
    text = load_pdf(path)
    index = text.find(first_sep)

    index = text.find(first_sep)
    if index!=-1:
        cut_text = text[index+len(first_sep):]
    else:
        cut_text = text
    
    return cut_text

class SBERTEmbeddings(Embeddings):
    
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

def split_text(text, chunk_size = 500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_text(text)

def retrieve_documents(query, collection_name, top_k=5):
    embeddings = SBERTEmbeddings()
    query_vector = embeddings.embed_query(query)

    collection = Collection(name = collection_name)
    collection.load()

    search_params = {"metric_type":"COSINE", "params":{"nprobe":10}}
    results = collection.search(
        data = [query_vector],
        anns_field = "embedding",
        param = search_params, 
        limit = top_k,
        output_fields=["text"]
    )

    retrieved_docs = [hit.entity.get("text") for hit in results[0]]
    return retrieved_docs

connections.connect(
    alias="default",
    uri=MILVUS_HOST,
    token=MILVUS_TOKEN,
    secure=True
)

query = "Что такое политика?"
docs = retrieve_documents(query, "pdf_obschestvo_bogolubov_9", top_k=3)
for i, doc in enumerate(docs):
    print(f"Результат {i+1}: {doc}")