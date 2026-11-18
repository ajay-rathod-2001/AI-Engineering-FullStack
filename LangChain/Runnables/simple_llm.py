
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

llm = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Define the Input
topic = input("Enter the topic: ")

# Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# Call the LLM directly
blog_title = llm.invoke(formatted_prompt)

# Print the output
print("Generated Blog Title:", blog_title.content)
