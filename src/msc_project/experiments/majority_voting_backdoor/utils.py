import torch


def noise_stepmlp(stepmlp, std=0.1):
    with torch.no_grad():
        for _, param in stepmlp.named_parameters():
            param.add_(torch.randn_like(param) * std)
