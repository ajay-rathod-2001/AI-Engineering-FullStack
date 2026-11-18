import os
from pydantic import SecretStr
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing!")

llm = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b",
    temperature=0.7)
  
# * Load the Document 
loader = TextLoader("docs.txt")  # Ensure docs.txt exists
documents = loader.load()

# * Split the text into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# * Convert text into embeddings & store
vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings())

# * Create a retriever (fetches relevent documents)
retriever = vectorstore.as_retriever()

# * Create RetrievalQACahin
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# * Ask a Question
query = "What are the key takeaways from the documents?"
answer = qa_chain.run(query)

print("Answer:", answer)