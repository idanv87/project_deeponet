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




def create_D2(x):
    Nx = len(x[1:-1])
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return scipy.sparse.csr_matrix(D2/dx/dx)


def solve_helmholtz(domain,f,k=Constants.k):
    b=f(domain.interior_vertices)

    L=create_D2(domain.vertices)
    A=-L -k* scipy.sparse.identity(L.shape[0])
    return scipy.sparse.linalg.spsolve(A, b)


def grf(domain, n, mu=0, sigma=0.1):
    np.random.seed(0)
    A=np.array([np.random.normal(mu, sigma,n) for i in range(len(domain)) ]).T

    # [plt.plot(domain, np.sqrt(2)*A[i,:]) for i in range(n)]
    # plt.show(block=False)
    torch.save(A, Constants.outputs_path+'grf.pt')
    return np.squeeze(np.sqrt(2)*A)


    
class Domain_1d:
    def __init__(self,n_points) -> None:
        self.vertices=np.linspace(0,1,30)
        self.interior_vertices=self.vertices[1:-1]
        self.L=create_D2(self.vertices)
        self.A=-self.L -Constants.k* scipy.sparse.identity(self.L.shape[0])

def generate_data(domain, n_samples):
    data_f=grf(domain.vertices,n_samples)
    data_f_interp=[scipy.interpolate.interp1d(domain.vertices,f, kind='cubic') for f in data_f]
    data_u=[solve_helmholtz(domain, f) for f in data_f_interp]
    torch.save((data_f[:,1:-1], data_u), Constants.outputs_path+'data.pt')
    fig,ax=plt.subplots(2)
    [ax[0].plot(domain.vertices, f ) for f in data_f]
    [ax[1].plot(domain.interior_vertices, u ) for u in data_u]
    plt.show(block=False)



# model([X[0].to(Constants.device),X[1].to(Constants.device)])






