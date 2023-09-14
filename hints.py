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
    L=create_D2(domain)
    N=len(domain)
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

 
    # x=deeponet(domain, model, func)
    # np.linalg.norm(solution-x)/np.linalg.norm(solution)
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
        if (res_err[-1] < (tol*10)) or (res_err[-1] > (100000)) :
            return d

    # torch.save(x, Constants.path+'pred.pt')

    return d


experment_path = Constants.path+'runs/'
best_model = torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


def run_hints(domain, func, J, J_in, hint_init):
    d= network(domain, model, func, J, J_in, hint_init)
    return d



# func=scipy.interpolate.interp1d(domain[1:-1],domain[1:-1]*(1-domain[1:-1]**2),kind='cubic')
def fig1():
    N=60
    domain = np.linspace(0, 1, N)
    # func=scipy.interpolate.interp1d(domain[1:-1],np.sin(math.pi*domain[1:-1]),kind='cubic')
    func=scipy.interpolate.interp1d(domain, grf(domain,1,mu=0.4,sigma=0.6))
    L=create_D2(domain)
    output=[]
    for j in [5,10,20]:
        d=run_hints(domain, func, J=j, J_in=[0], hint_init=True)
        torch.save(d, Constants.outputs_path+str(j)+'fig1.pt')
        output.append(torch.load(Constants.outputs_path+str(j)+'fig1.pt'))
    headers=['iterations', 'error']
    data_x=[None for o in output] 
    data_y=[o['err'] for o in output] 

    labels=['J='+str(o['J']) for o in output]

    fig,ax=plt.subplots()
    P=Plotter(ax,headers,data_x,data_y,labels, title=f'N={N}, k={Constants.k}', scale='log')
    P.plot_figure()
    P.save_figure(fig,Constants.eps_fig_path+'error_iter_different_J_N_'+str(N)+'_k_'+str(Constants.k)+'.eps')


def fig2():
    N=100
    domain = np.linspace(0, 1, N)
    func=scipy.interpolate.interp1d(domain[1:-1],np.sin(math.pi*domain[1:-1])
                                    +10*np.sin(5*math.pi*domain[1:-1])
                                    
                                    
                                    ,kind='cubic')
    output=[]
    J=20
    d=run_hints(domain, func, J=J, J_in=[0], hint_init=True)
    torch.save(d, Constants.outputs_path+str(J)+'fig2.pt')
    output=torch.load(Constants.outputs_path+str(J)+'fig2.pt')

    fig, ax = plt.subplots(1,2)
    modes=[1,5,10]
    headers=['iterations', '']
    labels=[str(k) for k in modes]
    data_x=[None,None,None] 
    data_y=[[o[k-1] for o in output['fourier_err'][:50]] for k in modes]
  
    labels=['mode='+str(k) for k in modes]
    P=Plotter(ax[0],headers,data_x,data_y,labels,scale='log', title='fourier modes error')
    P.plot_figure()

    P=Plotter(ax[1],headers,[None, None],[output['res_err'],output['err']],labels=['res. err', 'rel. error'],scale='log', title='solution error')
    P.plot_figure()
        
     
    P.save_figure(fig,Constants.eps_fig_path+'fourier_error'+str(N)+'_k_'+str(Constants.k)+'.eps')


fig2()
def fig3():
    N=60
    domain = np.linspace(0, 1, N)
    func=scipy.interpolate.interp1d(domain[1:-1],np.sin(math.pi*domain[1:-1])
                                    +10*np.sin(5*math.pi*domain[1:-1])
                                    +10*np.sin(10*math.pi*domain[1:-1])
                                    
                                    ,kind='cubic')

    output=[]
    J=5

    d=run_hints(domain, func, J=J, J_in=[0], hint_init=True)
    torch.save(d, Constants.outputs_path+str(J)+'fig3.pt')
    output=torch.load(Constants.outputs_path+str(J)+'fig3.pt')    
    times=list(range(25))
    hints_times=[0,5,10,15,20]
    fig, ax = plt.subplots(5, 5)
    # make 1d for easier access
    ax = np.ravel(ax)
    [ax[j].text(0.9, 0.8, 'NN', transform=ax[j].transAxes, fontsize=6,
                                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5)) 
    for j in hints_times ]                              
                                
    for j,it in enumerate(times):
        headers=['', '']
        if j==0:
            labels=['exact','numeric']
        else:    
            labels=[None,None]
        data_x=[domain[1:-1], domain[1:-1]]
        data_y=[output['solution'],output['x_k'][it]]

        P=Plotter(ax[j],headers,data_x,data_y,labels, title='')
        P.plot_figure()
        ax[j].set_xticks([])

    ax[0].text(0.5, 0.5, 'NN', transform=ax[j].transAxes, fontsize=6,
                                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))    
    ax = np.reshape(ax, (int(math.sqrt(len(times))), int(math.sqrt(len(times)))))    

    fig.suptitle(f'solution along iterations, J={J}')    
    P.save_figure(fig,Constants.eps_fig_path+'solution_error'+str(N)+'_k_'+str(Constants.k)+'.eps')


def fig4():
    N=60
    domain = np.linspace(0, 1, N)
    func=scipy.interpolate.interp1d(domain[1:-1],np.sin(math.pi*domain[1:-1])
                                    +10*np.sin(5*math.pi*domain[1:-1])
                                    +10*np.sin(10*math.pi*domain[1:-1])
                                    
                                    ,kind='cubic')

    output=[]
    J=5

    d=run_hints(domain, func, J=J, J_in=[0], hint_init=True)
    torch.save(d, Constants.outputs_path+str(J)+'fig4.pt')
    output=torch.load(Constants.outputs_path+str(J)+'fig4.pt')    
    times=list(range(25))
    hints_times=[0,5,10,15,20]
    fig, ax = plt.subplots(5, 5)
    # make 1d for easier access
    ax = np.ravel(ax)
    [ax[j].text(0.9, 0.8, 'NN', transform=ax[j].transAxes, fontsize=6,
                                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5)) 
    for j in hints_times ]                              
                                
    for j,it in enumerate(times):
        headers=['', '']
        if j==0:
            labels=['exact','numeric']
        else:    
            labels=[None,None]
        data_x=[None, None]
        data_y=[output['solution_expansion'],output['x_expansion'][it]]

        P=Plotter(ax[j],headers,data_x,data_y,labels, title='')
        P.plot_figure()
        ax[j].set_xticks([])

    ax[0].text(0.5, 0.5, 'NN', transform=ax[j].transAxes, fontsize=6,
                                ha='left', va='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))    
    ax = np.reshape(ax, (int(math.sqrt(len(times))), int(math.sqrt(len(times)))))    

    fig.suptitle(f'solution along iterations, J={J}')    
    P.save_figure(fig,Constants.eps_fig_path+'solution_error_fourier'+str(N)+'_k_'+str(Constants.k)+'.eps')






