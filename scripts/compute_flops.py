import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vad_architecture import LanguageGuidedVAD
from utils.video_utils import load_config

try:
    from thop import profile, clever_format
except ImportError:
    import subprocess
    print("Installing thop for FLOPs calculation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile, clever_format

def measure_flops():
    print("\n========================================")
    print(" COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("========================================\n")
    
    config = load_config("configs/config.yaml")
    
    # Instantiate the core VAD model (Cross-Attention + MLP)
    model = LanguageGuidedVAD.from_config(config)
    model.eval()
    
    # Create dummy inputs matching the shape expected from the offline extraction
    # Shape: (Batch=1, Segments=32, Feature_Dim=512)
    dummy_visual = torch.randn(1, 32, 512)
    dummy_text = torch.randn(1, 32, 512)
    
    # Profile the model
    macs, params = profile(model, inputs=(dummy_visual, dummy_text), verbose=False)
    
    # Format the results nicely
    macs_formatted, params_formatted = clever_format([macs, params], "%.2f")
    
    # FLOPs are typically considered as 2 * MACs (Multiply-Accumulate Operations)
    flops = 2 * macs
    flops_formatted, _ = clever_format([flops, params], "%.2f")
    
    print(f"Model Name:          Language-Guided WS-VAD")
    print(f"Input Shape:         [1, 32, 512] (Visual & Text)")
    print(f"Total Parameters:    {params_formatted}")
    print(f"MACs (Operations):   {macs_formatted}")
    print(f"Calculated FLOPs:    {flops_formatted}")
    print("\nNote: This is ONLY the cost of the Cross-Attention and MLP ranking network.")
    print("      The heavy extraction (CLIP/BLIP-2) is assumed to run offline/edge-device,")
    print("      making this core architecture extremely lightweight and fast to train!")

if __name__ == "__main__":
    measure_flops()
