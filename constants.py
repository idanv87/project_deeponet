import os
from typing import Any
import torch
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import Rbf


class Constants:
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dtype = torch.float32

    path = '/Users/idanversano/Documents/project_geo_deeponet/one_d_k25/'
    eps_fig_path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/k_25/'
    outputs_path=path+'outputs/'
    k=25
    batch_size=16
    num_epochs=10000


    
        