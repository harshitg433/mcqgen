import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

#imporing necessary packages packages from langchain
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda

# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.callbacks.manager import get_openai_callback


# Load environment variables from the .env file
load_dotenv()


# Access the environment variables just like you would with os.environ
key = os.getenv("OPENAI_API_KEY")


# print("Value of MY_VARIABLE:", key)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
    temperature=0.7
)


template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template
    )

# Define quiz_chain with explicit output key and verbose
# def format_quiz_output(result):
#     return {"quiz": result} 

# ✅ Chain: Quiz Generation
quiz_chain = quiz_generation_prompt | llm 


# quiz_chain = (quiz_generation_prompt | llm).with_config({"run_verbose": True})


template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}


Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

# ✅ Output formatting function
# def format_review_output(result):
#     return {"review": result}  

# ✅ Chain: Quiz Review & Analysis
review_chain = quiz_evaluation_prompt | llm 


# Define Final RunnableMap
generate_evaluate_chain = RunnableMap({
    "quiz": quiz_chain,  # Runs quiz_chain and stores output as "quiz"
    "subject": lambda x: x["subject"]  # Pass "subject" directly
}) | review_chain

# ✅ Fix: Ensure both outputs are returned
def run_review_chain(input_data):
    """Runs review_chain and ensures both quiz and review are returned"""
    quiz_output = input_data["quiz"]
    review_output = review_chain.invoke({"quiz": quiz_output, "subject": input_data["subject"]})
    return {"quiz": quiz_output, "review": review_output}

# ✅ Fix: Ensure both quiz and review are captured
generate_evaluate_chain = RunnableMap({
    "quiz": quiz_chain,  # Generate Quiz
    "subject": lambda x: x["subject"],  # Pass subject
}) | RunnableLambda(run_review_chain)  # Run Review and Return Both