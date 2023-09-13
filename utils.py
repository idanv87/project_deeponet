from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm
import datetime
import pickle
import math
import random
import cmath
import os
from tabulate import tabulate
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


def plot_solution_and_fourier(domain, times, path, eps_name, N):
    e_deeponet, r_deeponet, fourier_deeponet, x_k, solution,  solution_expansion, x_expansion, J, J_in, hint_init,gmres_err = torch.load(
        path)

    modes = [1, 6, 10]
    colors = ['blue', 'green', 'red']
    fig0, ax0 = plt.subplots(1, 3)
    [ax0[k].plot(list(range(len(fourier_deeponet[:50]))), [f[modes[k]-1]
                 for f in fourier_deeponet[:50]], color=colors[k], label=str(modes[k])) for k in range(3)]
    ax0[0].set_title('mode=1')
    ax0[1].set_title('mode=6')
    ax0[2].set_title('mode=10')
    ax0[0].set_xlabel('iteration')
    ax0[0].set_ylabel('error')
    ax0[1].set_yticks([])
    ax0[2].set_yticks([])

    times = np.asarray(times)
    all_modes = list(range(1, len(fourier_deeponet[0])+1))
    fig1, ax1 = plt.subplots(5, 5)
    fig2, ax2 = plt.subplots(5, 5)
    fig3, ax3 = plt.subplots()   # should be J+1
    fig1.supxlabel('x')
    fig2.supxlabel('mode')
    fig1.suptitle(f'solution, N={N}')
    fig2.suptitle(f'fourier coefficients, N={N}')
    fig3.suptitle(f'relative error, N={N}')
    ind = times.reshape((5, 5))
    counter = 0
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            ax1[i, j].plot(domain[1:-1], x_k[times[counter]],
                           'b', label='hints')
            ax1[i, j].plot(domain[1:-1], solution, color='r',
                           label='analytic', linestyle='dashed')
            bbox = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax1[i, j].text(0.8, 0.9, f'iter={times[counter]}', transform=ax1[i,
                           j].transAxes, fontsize=6, ha='left', va='top', bbox=bbox)
            ax2[i, j].text(0.8, 0.9, f'iter={times[counter]}', transform=ax2[i,
                           j].transAxes, fontsize=6, ha='left', va='top', bbox=bbox)
            ax2[i, j].plot(all_modes, x_expansion[times[counter]], 'b')
            ax2[i, j].plot(all_modes, solution_expansion,
                           color='r', linestyle='dashed')
            if j > 0:
                ax1[i, j].set_yticks([])
                ax2[i, j].set_yticks([])
            if (i+1) < ind.shape[0]:
                ax1[i, j].set_xticks([])
                ax2[i, j].set_xticks([])
            if counter == 0 and hint_init:
                ax1[i, j].text(0.5, 0.5, 'NN', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'NN', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            if counter == 0 and (not hint_init):
                ax1[i, j].text(0.5, 0.5, 'zero_init', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'zero_init', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            if (counter > J_in[-1]) and ((counter % J) in J_in):

                ax1[i, j].text(0.5, 0.5, 'NN', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'NN', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            counter += 1
    ax3.plot(e_deeponet, 'g')
    # ax3.plot(r_deeponet,'r',label='res.err')
    # ax3.legend()
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('error')
    ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}\n iter. {len(e_deeponet)}', transform=ax3.transAxes, fontsize=6,
             ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    fig0.savefig(eps_name+'all_fourier.eps', format='eps', bbox_inches='tight')
    fig1.savefig(eps_name+'sol.eps', format='eps', bbox_inches='tight')
    fig2.savefig(eps_name+'four.eps', format='eps', bbox_inches='tight')
    fig3.savefig(eps_name+'errors.eps', format='eps', bbox_inches='tight')
    plt.show(block=False)
    return [J, Constants.k,len(e_deeponet), e_deeponet[-1],gmres_err, N ]



# model([X[0].to(Constants.device),X[1].to(Constants.device)])






