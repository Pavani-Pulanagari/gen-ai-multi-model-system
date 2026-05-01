from text_generation import generate_text
from translation import translate_text
from rag_pipeline import ask_question

def run_pipeline(task, input_text):
    
    if task == "generate":
        return generate_text(input_text)
    
    elif task == "translate":
        return translate_text(input_text)
    
    elif task == "qa":
        return ask_question(input_text)
    
    else:
        return "Invalid task. Use: generate / translate / qa"
