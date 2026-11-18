import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS



load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# * Initialized the LLM
llm = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b",
    temperature=0.7
)

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

# * Manually Retrieve  Relevant Documents
query = "What are the key takeaways from the documents?"
retrieved_docs = retriever.invoke(query)

# * combine Retrieved Text into a single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# * Manually Pass Retrieved Text to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

# * print the Answer
print("Answer:", answer)


