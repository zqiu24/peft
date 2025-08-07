import torch


permutation_indices = torch.randperm(1024, device='cuda')
inv_permutation_indices = torch.argsort(permutation_indices)

P = torch.eye(1024, device='cuda', dtype=torch.float32, requires_grad=False)[permutation_indices]
P_T = torch.eye(1024, device='cuda', dtype=torch.float32, requires_grad=False)[inv_permutation_indices]

assert torch.allclose(P, P_T.T, atol=1e-6)

print("P == P_T.T", torch.allclose(P, P_T.T, atol=1e-6))