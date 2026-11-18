
import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Initialize the LLM
llm = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Create the Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with a specific topic
topic = input("Enter the topic: ")
output = chain.run({"topic": topic})

print("Generated Blog Title:", output["text"])
