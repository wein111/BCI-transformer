import torch
state = torch.load("best_model.pt")
print(list(state.keys())[:40])
