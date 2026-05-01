from transformers import pipeline
import torch

def generate_text(prompt):
    device = 0 if torch.cuda.is_available() else -1

    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=device
    )

    result = generator(
        prompt,
        max_length=50,
        num_return_sequences=1
    )

    return result[0]["generated_text"]
