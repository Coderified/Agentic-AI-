from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Load pdf file from directory
DATA_PATH = "data/"

def load_data(datapath):
    loader = DirectoryLoader(datapath,glob='*.pdf',loader_cls=PyPDFLoader)
    #this loads all pdf files more than one also
    documents = loader.load()
    return documents

documents = load_data(datapath = DATA_PATH)
#Chunks using TextSplitter

def text_chunk(documents):
    textsplitter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = textsplitter.split_documents(documents)
    return text_chunks

doc_chunks = text_chunk(documents = documents)
print(len(doc_chunks))


def select_embedding_model():
    embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embed_model

embedding_model = select_embedding_model()

FAISS_DATAPATH = "vectordatabase/db_faiss"
db=FAISS.from_documents(doc_chunks,embedding_model)

db.save_local(FAISS_DATAPATH)





