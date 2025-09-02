import torch
import numpy as np

def vectorized_mc_dropout_predict(model, x, T=100):
    model.train()
    B, S, F = x.shape
    x_repeat = x.repeat(T, 1, 1)

    with torch.no_grad():
        y_hat = model(x_repeat)
    y_hat = y_hat.view(T, B, -1)
    mean = y_hat.mean(dim=0)
    std = y_hat.std(dim=0)
    return mean.cpu().numpy(), std.cpu().numpy()
