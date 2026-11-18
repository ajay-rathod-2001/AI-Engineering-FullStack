import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Define the model

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY kaha hai bhaii...!")

model = ChatGroq(
    api_key=my_api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)

schema =[
    ResponseSchema(name= "fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name= "fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name= "fact_3", description="Fact 3 about the topic"),
        ]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "Give 3 fact about {topic} \n {format_instructions}",
    input_variables =["topic"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "Python Programming Language"})

print(result)

# prompt = template.invoke({"topic": "Python Programming Language"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
