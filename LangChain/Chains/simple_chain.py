import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("api_key kaha hai bhai...!")

model = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b"
)

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result  = chain.invoke({"topic":"AI Learning"})

print(result)

chain.get_graph().print_ascii()