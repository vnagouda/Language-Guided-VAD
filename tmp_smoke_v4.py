import sys
sys.path.insert(0, '.')
import torch
from models.vad_architecture import LanguageGuidedVAD, NormalMemoryBank
from utils.losses import VADLoss, SelfTrainingLoss
from utils.video_utils import load_config

cfg = load_config('configs/config_v4_sota.yaml')
print('[OK] Config loaded')

model = LanguageGuidedVAD.from_config(cfg)
print(f'[OK] Model built. Params: {sum(p.numel() for p in model.parameters()):,}')

B, T, D = 4, 32, 768
vis  = torch.randn(B, T, D)
txt  = torch.randn(B, T, D)
flow = torch.randn(B, T)
scores, norms, guided = model(vis, txt, flow)
print(f'[OK] Forward: scores={tuple(scores.shape)}, norms={tuple(norms.shape)}, guided={tuple(guided.shape)}')
assert scores.shape == (B, T)
assert norms.shape  == (B, T)
assert guided.shape == (B, T, D)

bank = NormalMemoryBank(feature_dim=D, bank_size=32)
bank.update(guided[:2].detach())
feats = bank.get()
print(f'[OK] MemoryBank get: {feats.shape if feats is not None else None}')

criterion = VADLoss.from_config(cfg)
labels = torch.tensor([1, 1, 0, 0])
abn = labels == 1
nor = labels == 0
loss_dict = criterion(
    scores[abn], scores[nor],
    norms[abn], norms[nor],
    epoch=1,
    guided_abn=guided[abn],
    guided_nor=guided[nor],
    bank_features=feats,
)
print(f'[OK] VADLoss total={loss_dict["total_loss"].item():.4f}')
print(f'     contrastive_loss={loss_dict.get("contrastive_loss", "N/A")}')
print(f'     bank_loss={loss_dict.get("bank_loss", "N/A")}')

self_crit = SelfTrainingLoss()
sl = self_crit(scores[abn], scores[nor], pseudo_k=5, smooth_window=3)
print(f'[OK] SelfTrainingLoss (smoothed)={sl.item():.4f}')

print()
print('[PASS] All V4 smoke tests passed!')
