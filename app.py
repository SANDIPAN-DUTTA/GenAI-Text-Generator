import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings

warnings.filterwarnings('ignore')
print("Loading GPT-2 model...")
try:
    model_name = 'gpt2'  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def generate_text(seed_text, max_length):
    """
    Generates text using the GPT-2 model.
    :param seed_text: The initial text to start generation from.
    :param max_length: The total length of the generated text.
    :return: The generated text as a string.
    """
    print(f"Generating text for seed: '{seed_text}' with max_length: {max_length}")
    
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=int(max_length),
        num_return_sequences=1,
        no_repeat_ngram_size=2,  
        early_stopping=True,
        temperature=0.8,  
        top_k=50,         
        top_p=0.95   
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

print("Launching Gradio interface...")

iface = gr.Interface(
    fn=generate_text,  
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a starting sentence...", label="Seed Text"),
        gr.Slider(minimum=20, maximum=200, step=10, value=50, label="Max Length")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="🤖 Gen AI Text Generator",
    description="This is a simple web app for generating text using the powerful GPT-2 model. Enter a seed text to get started!"
)

if __name__ == "__main__":
    iface.launch()