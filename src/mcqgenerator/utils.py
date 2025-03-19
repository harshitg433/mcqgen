import os
import PyPDF2
import json
from langchain_core.messages.ai import AIMessage
import traceback

def read_file(file):
    """Reads a PDF or TXT file and extracts text."""
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)  # ✅ Use PdfReader instead of PdfFileReader
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle possible NoneType returns
            return text
            
        except Exception as e:
            raise Exception(f"Error reading the PDF file: {str(e)}")
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    else:
        raise Exception("Unsupported file format. Only PDF and TXT files are supported.")

def get_table_data(quiz_str):
    try:
        if not quiz_str or not isinstance(quiz_str, str):
            raise ValueError("Error: quiz_str is empty or not a valid JSON string")

        print("DEBUG: Raw quiz_str before parsing →", repr(quiz_str))  # Debug output

        quiz_dict = json.loads(quiz_str)  # Convert JSON string to dict

        quiz_table_data = []
        
        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " || ".join([f"{opt}-> {opt_val}" for opt, opt_val in value["options"].items()])
            correct = value["correct"]

            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        return False

    except Exception as e:
        print(f"❌ General Error: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
