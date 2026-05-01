from transformers import MarianMTModel, MarianTokenizer
import torch

def translate_text(text):
    model_name = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    tokens = tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated = model.generate(**tokens)

    return tokenizer.decode(translated[0], skip_special_tokens=True)
