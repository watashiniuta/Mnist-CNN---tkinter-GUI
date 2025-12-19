import torch
import random
import numpy as np

def get_device(cuda):
    use_cuda = cuda and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
