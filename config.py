import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Balanced (FINAL)
MATCH_THRESHOLD = 0.70
GAP_THRESHOLD = 0.10