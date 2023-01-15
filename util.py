import torch
from torch.nn.functional import pad
import numpy as np


def rmap_iftype(X, fn, type):
    if isinstance(X, dict):
        return {k: rmap_iftype(X[k], fn, type) for k in X}
    elif isinstance(X, list):
        return [rmap_iftype(e, fn, type) for e in X]
    elif isinstance(X, tuple):
        return tuple([rmap_iftype(e, fn, type) for e in X])
    elif isinstance(X, type):
        return fn(X)
    return X

def move_to_device(X, device):
    return rmap_iftype(X, lambda x: x.to(device=device), torch.Tensor)

def torch_to_numpy(X):
    return rmap_iftype(X, lambda x: x.detach().cpu().numpy(), torch.Tensor)

def numpy_to_torch(X):
    return rmap_iftype(X, lambda x: torch.from_numpy(x), np.ndarray)

def homo_tens(x):
    return pad(x.reshape(-1, x.shape[-1]), [0, 0, 0, 1], value=1)

def dynamism(pcl: torch.Tensor, flow: torch.Tensor, rigid_transform: torch.Tensor) -> torch.Tensor:
    rigid_flow = (rigid_transform @ homo_tens(pcl.T)).T[:, :3] - pcl
    dynamic = torch.norm(rigid_flow - flow, dim=-1)
    return dynamic

def rigid_flow(data):
    pc = torch.from_numpy(data['pcl_0'])
    transform = torch.from_numpy(data['ego_motion'])
    return ((transform @ homo_tens(pc.T)).T[:, :3] - pc).numpy()
