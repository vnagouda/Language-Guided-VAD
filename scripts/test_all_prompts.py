import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM

def main():
    print("[INFO] Loading Models... This will take some time.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load BLIP-2
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # 2. Load Florence-2
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    print("[INFO] Models loaded successfully!\n")

    images = {
        "Abuse": "data/raw/Test/Abuse/Abuse028_x264_330.png",
        "RoadAccidents": "data/raw/Test/RoadAccidents/RoadAccidents001_x264_140.png",
        "Explosion": "data/raw/Test/Explosion/Explosion002_x264_330.png"
    }

    prompts = {
        "BLIP-2 (Generic)": "Question: What is happening in this image? Answer:",
        "BLIP-2 (Roleplay)": "Question: If you were a law enforcement agency monitoring this CCTV footage, what suspicious or anomalous activity is happening in this image? Answer:",
        "Florence-2 (Detailed)": "<MORE_DETAILED_CAPTION>"
    }

    for name, img_path in images.items():
        print(f"============================================================")
        print(f"Testing Image: {name} ({img_path})")
        print(f"============================================================")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        # Test BLIP-2 Generic
        inputs = blip_processor(image, text=prompts["BLIP-2 (Generic)"], return_tensors="pt").to(device, blip_model.dtype)
        generated_ids = blip_model.generate(**inputs, max_new_tokens=40)
        blip_gen_ans = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"[BLIP-2 Generic]: {blip_gen_ans}")

        # Test BLIP-2 Roleplay
        inputs = blip_processor(image, text=prompts["BLIP-2 (Roleplay)"], return_tensors="pt").to(device, blip_model.dtype)
        generated_ids = blip_model.generate(**inputs, max_new_tokens=40)
        blip_rp_ans = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"[BLIP-2 Roleplay]: {blip_rp_ans}")

        # Test Florence-2
        inputs = florence_processor(text=prompts["Florence-2 (Detailed)"], images=image, return_tensors="pt").to(device, florence_model.dtype)
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        florence_ans = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        florence_ans = florence_processor.post_process_generation(
            florence_ans, 
            task=prompts["Florence-2 (Detailed)"], 
            image_size=(image.width, image.height)
        )[prompts["Florence-2 (Detailed)"]]
        print(f"[Florence-2 Detailed]: {florence_ans}\n")

if __name__ == "__main__":
    main()
