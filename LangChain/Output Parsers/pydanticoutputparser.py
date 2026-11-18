
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("api_key kaha hai bhai...!")

model = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b"
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to.")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser

final_result = chain.invoke({"place":"uk"})

print(final_result)

# prompt = template.invoke({"place": "Indian"})

# print(prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
