from pathlib import Path
import torch
import numpy as np

FEATURES_ROOT = Path("data/features")

def collect_split(split_name: str):
    split_dir = FEATURES_ROOT / split_name
    visual_files = sorted(split_dir.glob("*_visual.pt"))

    anomaly_norms = []
    normal_norms = []

    for vis_path in visual_files:
        video_name = vis_path.stem.replace("_visual", "")
        label_path = split_dir / f"{video_name}_label.pt"

        if not label_path.exists():
            continue

        visual = torch.load(vis_path, map_location="cpu", weights_only=True)  # (32, 512)
        label = int(torch.load(label_path, map_location="cpu", weights_only=True).item())

        norms = torch.norm(visual, dim=-1).numpy()  # (32,)

        if label == 1:
            anomaly_norms.extend(norms.tolist())
        else:
            normal_norms.extend(norms.tolist())

    return np.array(anomaly_norms), np.array(normal_norms)

def describe(name: str, values: np.ndarray):
    if len(values) == 0:
        print(f"{name}: no values")
        return
    print(f"{name}")
    print(f"  count:  {len(values)}")
    print(f"  mean:   {values.mean():.6f}")
    print(f"  std:    {values.std():.6f}")
    print(f"  min:    {values.min():.6f}")
    print(f"  max:    {values.max():.6f}")
    print()

def main():
    for split in ["Train", "Test"]:
        print("=" * 60)
        print(f"SPLIT: {split}")
        print("=" * 60)

        anomaly_norms, normal_norms = collect_split(split)

        describe("Anomaly segment norms", anomaly_norms)
        describe("Normal segment norms", normal_norms)

        if len(anomaly_norms) > 0 and len(normal_norms) > 0:
            diff = anomaly_norms.mean() - normal_norms.mean()
            print(f"Mean difference (anomaly - normal): {diff:.6f}")
            print()

if __name__ == "__main__":
    main()