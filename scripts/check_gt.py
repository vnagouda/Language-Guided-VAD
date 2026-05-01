"""Quick script to analyze the per-video frame-level GT files."""
from pathlib import Path

d = Path("data/Temporal_Anomaly_Annotation_For_Testing_Videos/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate")
items = sorted(d.iterdir())
print(f"Files: {len(items)}")

total_frames = 0
total_anomaly = 0
for f in items[:5]:
    with open(f) as fh:
        vals = [v.strip() for v in fh.read().strip().split("\n")]
    n_frames = len(vals)
    n_anom = sum(1 for v in vals if v == "1")
    total_frames += n_frames
    total_anomaly += n_anom
    print(f"  {f.name}: {n_frames} frames, {n_anom} anomalous")

# Count all
total_frames = 0
total_anomaly = 0
for f in items:
    with open(f) as fh:
        vals = [v.strip() for v in fh.read().strip().split("\n")]
    total_frames += len(vals)
    total_anomaly += sum(1 for v in vals if v == "1")

print(f"\nTotals across ALL {len(items)} videos:")
print(f"  Total frames: {total_frames:,}")
print(f"  Anomaly frames: {total_anomaly:,}")
print(f"  Normal frames: {total_frames - total_anomaly:,}")
print(f"  Anomaly ratio: {total_anomaly/total_frames:.1%}")
print(f"\n  vs OUR evaluation: 341,762 total, 24.6% anomaly")
print(f"  THIS is the ground truth SOTA papers use!")
