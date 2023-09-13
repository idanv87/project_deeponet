from main import model
import os
import sys
import math


from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np

import scipy
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

import sys
from constants import Constants

from utils import create_D2, grf, plot_solution_and_fourier

from one_d_data_set import create_loader
from packages.my_packages import *


main_domain=np.linspace(0,1,30)
# torch.save(run_hints(func, J=20, J_in=[0,1], hint_init=True), Constants.outputs_path+'modes_error.pt')


def deeponet(domain, model, func):
    x=domain[1:-1]
    s0=func(main_domain[1:-1])
    with torch.no_grad():
        y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(x.shape[0],)
        s_temp=torch.tensor(s0.reshape(1,s0.shape[0]),dtype=torch.float32).repeat(x.shape[0],1)
        pred2=model([y,s_temp])
    return pred2.numpy()

def network(domain, model,func, J, J_in, hint_init):

    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))

    # m=10
    # x=domain[1:-1]
    # b=6*x-Constants.k*(x-x**3)
    # u=x-x**3
    # est=scipy.sparse.linalg.spsolve(A, b)
    # plt.plot(domain[1:-1],est,'b',linestyle='dashed')
    # plt.plot(domain[1:-1],u,'-r')
    # plt.show()
   
    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    # print(ev)

    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    tol=np.linalg.norm(A@solution-b)/np.linalg.norm(b)
    
    gmres_err=0
    # tol=np.linalg.norm(A@solution-b)/np.linalg.norm(b)
    # gmres_solution, iter=gmres(A, b, b*0, nmax_iter=N, tol=tol*10)
    # gmres_err=np.linalg.norm(gmres_solution-solution)/np.linalg.norm(solution)
    
    # print(gmres_err)
    # print(iter)

  
    solution_expansion=[np.dot(solution,V[:,s]) for s in range(V.shape[1])]


    if hint_init:
        x=deeponet(domain, model, func)
        
    else:
        x=deeponet(domain, model, func)*0


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
                deeponet(domain, model, scipy.interpolate.interp1d(
                    domain[1:-1], (b-A@x_0)*factor, kind='cubic'))
            x = x_temp/factor

            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]

        # print(np.linalg.norm(A@x-b)/np.linalg.norm(b))

        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))

        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))


        d={'N':N, 'err':err, 'res_err':res_err, 'fourier_err':fourier_err, 
       'x_k':x_k,'solution':solution, 'solution_expansion':solution_expansion, 
       'x_expansion':x_expansion, 'J':J, 'J_in':J_in,'hint_init': hint_init, 'gmres_err':gmres_err}
        # if (res_err[-1] < (tol*10)) or (res_err[-1] > (10000)) :
        #     return d

    # torch.save(x, Constants.path+'pred.pt')

    return d


experment_path = Constants.path+'runs/'
best_model = torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


def run_hints(domain, func, J, J_in, hint_init):
    d= network(domain, model, func, J, J_in, hint_init)
    return d




# func=scipy.interpolate.interp1d(domain[1:-1],domain[1:-1]*(1-domain[1:-1]**2),kind='cubic')
output=[]
N=120
domain = np.linspace(0, 1, N)

func=scipy.interpolate.interp1d(domain, grf(domain,1,mu=0.4,sigma=0.7))
# func = scipy.interpolate.interp1d(domain[1:-1], np.sin(10*(math.pi)*(domain[1:-1])) , kind='cubic')
L=create_D2(domain)

# [2, 5,15,20 ]
output=[]

for j in [ 10,15,20,30 ]:
    # d=run_hints(domain, func, J=j, J_in=[0], hint_init=True)
    # torch.save(d, Constants.outputs_path+str(j)+'.pt')
    output.append(torch.load(Constants.outputs_path+str(j)+'.pt'))
headers=['iterations', 'error']
data_x=[None for o in output] 
data_y=[o['err'] for o in output] 

labels=['J='+str(o['J']) for o in output]
P=Plotter(headers,data_x,data_y,labels, title=f'N={N}, k={Constants.k}')
P.plot_figure()
P.save_figure(Constants.eps_fig_path+'error_iter_different_J_N_'+str(N)+'_k_'+str(Constants.k)+'.eps')
# plot_table(['J','k', 'iter.', 'err', 'N'],output, Constants.outputs_path+'sin1_N_240_k_25.txt')





# func=scipy.interpolate.interp1d(domain[1:-1],  grf(domain[1:-1],1), kind='cubic')
# F,U=torch.load(Constants.outputs_path+'data.pt')
# func=scipy.interpolate.interp1d(domain[1:-1],
#                                 F[0]
#                                 , kind='cubic')
# func = scipy.special.legendre(10)













