
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY kaha hai bhaii...!")

model = ChatGroq(
    api_key=my_api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me 5 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({"topic": "cricket"})
print(result)