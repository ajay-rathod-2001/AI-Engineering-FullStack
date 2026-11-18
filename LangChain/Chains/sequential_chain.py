import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
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

prompt_1 = PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables = ["topic"]
)

prompt_2 = PromptTemplate(
    template = "Generate the 5 pointer from the following test \n {text}",
    input_variables = ["text"]
)

parser= StrOutputParser()

chain = prompt_1 | model | parser |  prompt_2 | model | parser 

result = chain.invoke({"topic":"Why FAANG and MAANG companies Layoff the Employee"})

print(result)

chain.get_graph().print_ascii()