import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading Florence-2. This will take 15-20 seconds...")
processor = AutoProcessor.from_pretrained("microsoft/florence-2-large", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/florence-2-large",
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

# Load a frame we know has an anomaly (Abuse 028)
img_path = "data/raw/Test/Abuse/Abuse028_x264_330.png"
print(f"\n[INFO] Loading image: {img_path}")
image = Image.open(img_path).convert("RGB")

task = "<MORE_DETAILED_CAPTION>"

inputs = processor(text=task, images=image, return_tensors="pt").to(device, torch.float16)

print("\n--- TEST: Florence-2 Native Spatial Task ---")
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
raw_output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed = processor.post_process_generation(raw_output, task=task, image_size=(image.width, image.height))
print(f"Result: {parsed[task]}")
