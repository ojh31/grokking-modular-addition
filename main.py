#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from jaxtyping import Float, Int
from torch import Tensor
from functools import partial
import copy
from typeguard import typechecked

import os
import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
#%%
# SETUP AND BACKGROUND
# We train transformers to perform addition mod P. The input to the model is of the form “a b =”,
# where a and b are encoded as P-dimensional one-hot vectors, and = is a special token above which
# we read the output c. In our mainline experiment, we take P = 113 and use a one-layer ReLU
# transformer, token embeddings with d = 128, learned positional embeddings, 4 attention heads of
# dimension d/4 = 32, and n = 512 hidden units in the MLP. In other experiments, we vary the depth
# and dimension of the model. We did not use LayerNorm or tie our embed/unembed matrices.
# Our mainline dataset consists of 30% of the entire set of possible inputs (that is, 30% of the 113 ·
# 113 pairs of numbers mod P). We use full batch gradient descent using the AdamW optimizer
# (Loshchilov & Hutter, 2017) with learning rate γ = 0.001 and weight decay parameter λ = 1. We
# perform 40, 000 epochs of training. As there are only 113 · 113 possible pairs, we evaluate test loss
# and accuracy on all pairs of inputs not used for training.
#%%
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p = 113

cfg = HookedTransformerConfig(
    n_layers = 1,
    d_vocab = p+1,
    d_model = 128,
    d_mlp = 4 * 128,
    n_heads = 4,
    d_head = 128 // 4,
    n_ctx = 3,
    act_fn = "relu",
    normalization_type = None,
    device = device,
)

model = HookedTransformer(cfg)
#%%
def create_addition_dataset(base):
    '''
    Returns tuple of the form train tokens, test tokens
    '''
    triples = [(a, b, base) for a in range(base) for b in range(base)]
    random.shuffle(triples)
    inputs = torch.stack([torch.tensor(triple) for triple in triples], dim=0)
    labels = torch.tensor([a + b % base for a, b, _ in triples]) 
    return (
        inputs[: int(len(triples) * 0.3)], 
        labels[: int(len(triples) * 0.3)], 
        inputs[int(len(triples) * 0.3) :],
        labels[int(len(triples) * 0.3) :],
    )
#%%
x_train, y_train, x_test, y_test = create_addition_dataset(p)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
train_size, n_ctx = x_train.shape
assert n_ctx == 3
test_size, n_ctx = x_test.shape
assert train_size + test_size == p * p
#%%
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
# %%
# basic training loop
@typechecked
def train(
    model, 
    x_train: Int[Tensor, "batch pos"], 
    y_train: Int[Tensor, "batch"], 
    n_epochs=40_000,
):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1)
    for epoch in tqdm.trange(n_epochs):
        out: Float[Tensor, "batch pos vocab"] = model(x_train)
        y_hat: Float[Tensor, "batch vocab"] = out[:, -1, :]
        loss = F.cross_entropy(y_hat, y_train)
        optimizer.zero_grad()
        print(loss.device)
        print(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 1_000 == 0:
            print(f"epoch {epoch} loss: {loss.item()}")
    return model
#%%
trained_model = train(model, x_train, y_train)

# %%
