
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda


load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai!")

# Initialize Groq Model
model = ChatGroq(
    api_key=my_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)

# Output Parser
parser = StrOutputParser()

# Classs build
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt Template
prompt1 = PromptTemplate(
    template="""
    Classify the sentiment of the following feedback text
    into either Positive or Negative.

    Return only one word: Positive or Negative.

    Feedback: {feedback} \n {format_instruction}
    """,
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

# Classifier Chain
classifier_chain = prompt1 | model | parser2

# Prompt2
prompt2 = PromptTemplate(
    template = """
    write an appropriateto this positive feedback \n 
    
    feedbacK:{feedback}
    """,
    input_variables=["feedback"]
)

# Prompt3
prompt3 = PromptTemplate(
    template = """
    write an appropriateto this negative feedback \n 
    
    feedbacK:{feedback}
    """,
    input_variables=["feedback"]
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback":"This the terrible phone"}))
# print(chain.invoke({"feedback":"This the beautiful phone"}))
chain.get_graph().print_ascii()

# Invoke Chain
# result = classifier_chain.invoke({
#     "feedback": "This is a terrible Smartphone"
# }).sentiment

# print(result)
