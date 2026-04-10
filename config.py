import torch

# ✅ FORCE CPU (Renderக்கு correct)
DEVICE = torch.device("cpu")

MATCH_THRESHOLD = 0.70
GAP_THRESHOLD = 0.10