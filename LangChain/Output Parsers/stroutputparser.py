
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

model = ChatGroq(
    api_key=my_api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)

template_1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template_2 = PromptTemplate(
    template="Summarize the following text in 5 lines:\n{text}",
    input_variables=["text"]
)

prompt1 = template_1.invoke({"topic": "black hole"})

result = model.invoke(prompt1)

prompt2 = template_2.invoke({"text": result.content})

result1 = model.invoke(prompt2)

print(result1.content)
