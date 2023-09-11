from main import model
import os
import sys
import math


from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import gmres
import scipy
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

import sys
from constants import Constants

from utils import create_D2, grf
from main import domain
from one_d_data_set import create_loader
from packages.my_packages import *

main_domain=domain.vertices
domain = np.linspace(0, 1, 60)

L=create_D2(domain)

    

def deeponet(model, func):
    x=domain[1:-1]
    s1=func(main_domain[1:-1])
    with torch.no_grad():
        y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(x.shape[0],)
        s_temp=torch.tensor(s1.reshape(1,s1.shape[0]),dtype=torch.float32).repeat(x.shape[0],1)
        pred2=model([y,s_temp])
    return pred2.numpy()

def network(model,func, J, J_in, hint_init):

    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    
   
    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    # print(ev)

    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    gmres_solution,exit_cod=gmres(A, b, x0=None, tol=1e-13, restart=None, maxiter=4000)
    print(np.linalg.norm(A@gmres_solution-b)/np.linalg.norm(b))
    print(exit_cod)
    solution_expansion=[np.dot(solution,V[:,s]) for s in range(V.shape[1])]


    if hint_init:
        x=deeponet(model, func)
        
    else:
        x=deeponet(model, func)*0

    tol=[]
    res_err=[]
    err=[]
    fourier_err=[]
    x_expansion=[]
    k_it=0
    x_k = []

    # plt.plot(solution,'r');plt.plot(x,'b');plt.show()

    for temp in range(4000):
        fourier_err.append([abs(np.dot(x-solution, V[:, i]))
                           for i in range(15)])
        x_k.append(x)
        x_expansion.append([np.dot(x, V[:, s]) for s in range(V.shape[1])])

        x_0 = x
        k_it += 1
        theta = 1

        if (((k_it % J) in J_in) and (k_it > J_in[-1])):

            factor = np.max(abs(grf(main_domain, 1)))/np.max(abs(A@x_0-b))
           
            x_temp = x_0*factor + \
                deeponet(model, scipy.interpolate.interp1d(
                    domain[1:-1], (b-A@x_0)*factor, kind='cubic'))
            x = x_temp/factor

            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]

        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))

        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))

        tol.append(np.linalg.norm(x-x_0))
        if (res_err[-1] < 1e-13) and (err[-1] < 1e-13):
            return (err, res_err, fourier_err, x_k, solution, solution_expansion, x_expansion, J, J_in, hint_init)

    # torch.save(x, Constants.path+'pred.pt')

    return (err, res_err, fourier_err, x_k, solution, solution_expansion, x_expansion, J, J_in, hint_init)


experment_path = Constants.path+'runs/'
best_model = torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


def run_hints(func, J, J_in, hint_init):
    return network(model, func, J, J_in, hint_init)
    # torch.save([ err_net, res_err_net], Constants.path+'hints_fig.pt')


def plot_solution_and_fourier(times, path, eps_name):
    e_deeponet, r_deeponet, fourier_deeponet, x_k, solution,  solution_expansion, x_expansion, J, J_in, hint_init = torch.load(
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
    fig1.suptitle('solution')
    fig2.suptitle('fourier coefficients')
    fig3.suptitle('relative error')
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
                ax1[i, j].text(0.5, 0.5, 'Hints', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'Hints', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            if counter == 0 and (not hint_init):
                ax1[i, j].text(0.5, 0.5, 'zero_init', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'zero_init', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            if (counter > J_in[-1]) and ((counter % J) in J_in):

                ax1[i, j].text(0.5, 0.5, 'Hints', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2[i, j].text(0.5, 0.5, 'Hints', transform=ax2[i, j].transAxes, fontsize=6,
                               ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            counter += 1
    ax3.plot(e_deeponet, 'g')
    # ax3.plot(r_deeponet,'r',label='res.err')
    # ax3.legend()
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('error')
    ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}', transform=ax3.transAxes, fontsize=6,
             ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    fig0.savefig(eps_name+'all_fourier.eps', format='eps', bbox_inches='tight')
    fig1.savefig(eps_name+'sol.eps', format='eps', bbox_inches='tight')
    fig2.savefig(eps_name+'four.eps', format='eps', bbox_inches='tight')
    fig3.savefig(eps_name+'errors.eps', format='eps', bbox_inches='tight')
    plt.show(block=True)
    return 1


func = scipy.interpolate.interp1d(domain[1:-1],
                                  10*np.sin(10*(math.pi)*(domain[1:-1])) +
                                  10*np.sin(6*(math.pi)*(domain[1:-1])) +
                                  1*np.sin((math.pi)*(domain[1:-1])), kind='cubic')
# func=scipy.interpolate.interp1d(domain[1:-1],  grf(domain[1:-1],1), kind='cubic')
# F,U=torch.load(Constants.outputs_path+'data.pt')
# func=scipy.interpolate.interp1d(domain[1:-1],
#                                 F[0]
#                                 , kind='cubic')
func = scipy.special.legendre(10)


torch.save(run_hints(func, J=40, J_in=[0], hint_init=True), Constants.outputs_path+'modes_error.pt')
plot_solution_and_fourier(list(range(0,0+25)),Constants.outputs_path+'modes_error.pt', Constants.eps_fig_path+ 'one_d_x0_J=8_Jin=012_modes=1')










