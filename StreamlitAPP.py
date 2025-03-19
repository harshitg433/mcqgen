import os
import sys
import json
import traceback
import pandas as pd
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

#loading json file

with open('/home/harshit433/mcqgen/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

#creating a title for the app
st.title("MCQs Creator Application with LangChain 🦜⛓️")

#Create a form using st.form
with st.form("user_inputs"):
    #File Upload
    uploaded_file=st.file_uploader("Uplaod a PDF or txt file")

    #Input Fields
    mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

    #Subject
    subject=st.text_input("Insert Subject",max_chars=20)

    # Quiz Tone
    tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")

    #Add Button
    button=st.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_file)
                #Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain.invoke(
                        {
                        "text": text,
                        "number": mcq_count,
                        "subject":subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                            }
                    )
                #st.write(response)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict) and "quiz" in response:
                    quiz_message = response["quiz"]

                    # Check if the response is a dictionary and contains quiz data
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        # Ensure quiz is a valid JSON string
                        if isinstance(quiz, AIMessage):
                            quiz_str = quiz.content  # Extract content from AIMessage
                        elif isinstance(quiz, dict):  
                            quiz_str = json.dumps(quiz)  # Convert dict to JSON string
                        elif isinstance(quiz, str):
                            quiz_str = quiz.strip()  # Ensure it's a clean string
                        else:
                            raise ValueError(f"Unexpected type for quiz: {type(quiz)}")

                        # Debugging: Print extracted quiz JSON
                        print("DEBUG: Extracted Quiz JSON:", quiz_str)

                        # Ensure quiz_str is now a valid JSON before passing it
                        table_data = get_table_data(quiz_str)

                        if table_data:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                        else:
                            st.error("Error in processing quiz data.")
                    else:
                        st.error("No quiz data found in the response.")
                else:
                    st.write(response)


