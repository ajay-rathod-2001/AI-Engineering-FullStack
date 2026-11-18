import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Model 1 - Generate Notes and Merge
model_1 = ChatGroq(
   api_key=SecretStr(api_key),
    model="openai/gpt-oss-120b",
    temperature=0
)


# Model 2 - Generate Quiz
model_2 = ChatGroq(
    api_key=SecretStr(api_key),
    model="openai/gpt-oss-20b",
    temperature=0
)



# Prompt 1 - Notes
prompt_1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 2 - Quiz
prompt_2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"]
)

# Prompt 3 - Merge Notes and Quiz
prompt_3 = PromptTemplate(
    template="""
    Merge the provided notes and quiz into a single
    well-structured document.

    NOTES:
    {notes}

    QUIZ:
    {quiz}
    """,
    input_variables=["notes", "quiz"]
)

# Output Parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel({
    "notes": prompt_1 | model_1 | parser,
    "quiz": prompt_2 | model_2 | parser
})

# Merge Chain
merge_chain = prompt_3 | model_1 | parser

# Complete Chain
chain = parallel_chain | merge_chain

# Input Text
text = """
1. Introduction to SVM

Support Vector Machine (SVM) is a supervised machine learning
algorithm used for classification and regression tasks. It works
by finding the best decision boundary, called a hyperplane,
that separates different classes of data.

2. Working of SVM

SVM works by plotting data points in a feature space and finding
the optimal hyperplane that separates different classes.

3. Hyperplane

A hyperplane is a decision boundary that separates data points
belonging to different classes.

4. Support Vectors

Support vectors are the data points closest to the decision
boundary or hyperplane.

5. Margin

Margin is the distance between the hyperplane and the nearest
data points of different classes.

6. Types of SVM

Linear SVM is used when data can be separated using a straight
line. Non-Linear SVM uses kernel functions to separate complex data.

7. Kernel Trick

The kernel trick is a technique used in SVM to handle non-linear
data. Common kernels include Linear, Polynomial, RBF and Sigmoid.

8. Hard Margin and Soft Margin

Hard Margin SVM does not allow misclassification.
Soft Margin SVM allows some classification errors.

9. SVM for Classification

SVM separates data into different categories using an optimal
hyperplane.

10. SVM for Regression

Support Vector Regression (SVR) predicts continuous numerical
values.

11. Advantages of SVM

- Works effectively with high-dimensional data.
- Provides good classification accuracy.
- Supports linear and non-linear classification.

12. Disadvantages of SVM

- Training can be slow for large datasets.
- Choosing the correct kernel can be difficult.
- Sensitive to parameter selection.

13. Applications of SVM

- Email spam detection.
- Face recognition.
- Handwriting recognition.
- Medical diagnosis.
- Image classification.
- Sentiment analysis.

14. SVM in Python

SVM can be implemented using Scikit-learn.
SVC is used for classification and SVR is used for regression.
"""

# Invoke Chain
result = chain.invoke({"text": text})

# Display Result
print(result)

chain.get_graph().print_ascii()