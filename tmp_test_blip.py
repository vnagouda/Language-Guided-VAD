import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

device = "cuda"
model_name = "Salesforce/blip2-opt-2.7b"
print("Loading model...")
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

img_path = "data/raw/Test/Abuse/Abuse028_x264_330.png"
print(f"Loading image {img_path}")
image = Image.open(img_path).convert("RGB")

prompts = [
    "Question: What is happening in this image? Answer:",
    "Question: Are there any violent, suspicious, or anomalous actions in this image? Answer:",
    "Question: What is the main action taking place? Answer:",
    "Question: Describe the violent action in the image. Answer:",
]

for p in prompts:
    inputs = processor(images=image, text=p, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=40)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    print(f"Prompt: '{p}'")
    
    # Strip the prompt from the output if it's echoed
    if text.startswith(p):
        generated = text[len(p):].strip()
    else:
        generated = text
        
    print(f"-> Generated: '{generated}'\n")

