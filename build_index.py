
# build_index.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models import get_embeddings

load_dotenv()

INDEX_PATH = "faiss_index"

def Build_index():
    print("Loading documents...")

    loader_txt = TextLoader("DATA/1.txt")
    loader_pdf = PyPDFLoader("DATA/Life.pdf")

    documents = loader_pdf.load() + loader_txt.load()

    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(documents)


    print("Creating embeddings...")
    embedding_model = get_embeddings();##Initializing the embedding model.....

    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print("Saving index locally...")
    vectorstore.save_local(INDEX_PATH)
    print("Index built successfully!")

if __name__ == "__main__":
    Build_index()
