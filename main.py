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

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from functools import partial
import copy

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
