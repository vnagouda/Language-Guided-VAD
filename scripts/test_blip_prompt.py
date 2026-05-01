import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading BLIP-2. This will take a few seconds...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
).to(device)

prompt_normal = "Question: What is happening in this image? Answer:"
prompt_impersonate = "Question: If you were a law enforcement agency monitoring this CCTV footage, what suspicious or anomalous activity is happening in this image? Answer:"

# Load a frame we know has an anomaly (Abuse 028)
img_path = "data/raw/Test/Abuse/Abuse028_x264_330.png"
print(f"[INFO] Loading image: {img_path}")
image = Image.open(img_path).convert("RGB")

print("\n--- TEST: Generic Baseline Prompt ---")
inputs_normal = processor(images=image, text=prompt_normal, return_tensors="pt").to(device, torch.float16)
ids_normal = model.generate(**inputs_normal, max_new_tokens=40)
print(processor.batch_decode(ids_normal, skip_special_tokens=True)[0].strip())

print("\n--- TEST: LAVAD Impersonation Prompt ---")
inputs_impersonate = processor(images=image, text=prompt_impersonate, return_tensors="pt").to(device, torch.float16)
ids_impersonate = model.generate(**inputs_impersonate, max_new_tokens=40)
print(processor.batch_decode(ids_impersonate, skip_special_tokens=True)[0].strip())
