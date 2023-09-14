from one_d_model import deeponet
from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm
import datetime
import pickle
import math
import random
import cmath
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import List, Tuple
import sklearn
import argparse
import torch
import dmsh
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.stats import qmc
from scipy.linalg import circulant
from constants import Constants
from one_d_data_set import SonarDataset, create_loader
from utils import Domain_1d, generate_data, solve_helmholtz
from packages.my_packages import *


domain = Domain_1d(30)
# generate_data(domain, n_samples=300)

F, U = torch.load(Constants.outputs_path+'data.pt')
X = []
Y = []
X_test = []
Y_test = []
for i in range(1, 300):
    s0 = F[i]
    s1 = U[i]
    for j, y in enumerate(list(domain.interior_vertices)):
        X.append([torch.tensor(y, dtype=torch.float32),
                 torch.tensor(s0, dtype=torch.float32)])
        Y.append(torch.tensor(s1[j], dtype=torch.float32))

for i in range(1):
    s0 = F[i]
    s1 = U[i]
    for j, y in enumerate(list(domain.interior_vertices)):
        X_test.append([torch.tensor(y, dtype=torch.float32),
                      torch.tensor(s0, dtype=torch.float32)])
        Y_test.append(torch.tensor(s1[j], dtype=torch.float32))


# plt.plot([z[0] for z in X_test],Y_test)
# plt.show()

my_dataset = SonarDataset(X, Y)
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size


train_dataset, val_dataset = torch.utils.data.random_split(
    my_dataset, [train_size, val_size]
)
test_dataset = SonarDataset(X_test, Y_test)
val_dataloader = create_loader(
    val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
train_dataloader = create_loader(
    train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
test_dataloader = create_loader(
    test_dataset, batch_size=1, shuffle=False, drop_last=False)

inp, out = next(iter(test_dataset))
model = deeponet(1, inp[1].shape[0], 80)
inp, out = next(iter(train_dataloader))
model(inp)
# model=deeponet( 1, domain.vertices.shape[0]-2, 80)
print(f" num of model parameters: {count_trainable_params(model)}")
