import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY kaha hai bhaii...!")

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

parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser 

result = chain.invoke({"topic": "black hole"})
print(result)