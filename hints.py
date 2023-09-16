from main import model
import os
import sys
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    # np.linalg.norm(solution-x)/np.linalg.norm(solution)

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



def fig1(N=100):
    domain = np.linspace(0, 1, N)

    func=scipy.interpolate.interp1d(domain[1:-1],
                                    np.sin(math.pi*domain[1:-1])+
                                    5*np.sin(3*math.pi*domain[1:-1])+
                                    10*np.sin(5*math.pi*domain[1:-1]),
                                    kind='cubic')

        # func=scipy.interpolate.interp1d(domain, grf(domain,1,mu=0,sigma=0.1))


    output=[]
    all_J=list(range(5,21,1))
    plot_J=[5,7,10,20]
    conv_rates=[]
    for j in all_J:
        d=run_hints(domain, func, J=j, J_in=[0], hint_init=True)
        torch.save(d, Constants.outputs_path+str(j)+'fig1.pt')
        output.append(torch.load(Constants.outputs_path+str(j)+'fig1.pt'))

    output_J=[torch.load(Constants.outputs_path+str(j)+'fig1.pt')  for j in plot_J ]

    fig,ax=plt.subplots(1,2)
    fig.tight_layout(pad=5.0)

    headers=['iterations', 'error']
    data_x=[None for o in output_J] 
    data_y=[o['err'] for o in output_J] 
    labels=['J='+str(o['J']) for o in output_J]

  
    P1=Plotter(ax[0],headers,data_x,data_y,labels, title=f'relative error', scale='log')
    P1.plot_figure()
    conv_rates=[-np.polyfit(np.arange(len(o['err']))/( (len(o['err'])-1)*0+1 ),np.log(o['err']),1)[0] for o in output]

    P2=Plotter(ax[1],['J','conv.rate'],[all_J],[conv_rates],labels=['conv. rates'], title=f'convergence rates')
    P2.plot_figure()
    P2.save_figure(fig,Constants.eps_fig_path+'fig1a.eps')


    fig2,ax2=plt.subplots()
    P3=Plotter(ax2,['x', 'u(x)'], [domain[1:-1], domain[1:-1]],[output[0]['solution'], output[0]['x_k'][0]],labels=['exact', 'NN'])
    P3.plot_figure()
    P3.save_figure(fig2,Constants.eps_fig_path+'fig1b.eps')
   

    

    plt.show()



def fig2(N=100):
    
    domain = np.linspace(0, 1, N)
    func=scipy.interpolate.interp1d(domain[1:-1],
                                    np.sin(math.pi*domain[1:-1])+
                                    5*np.sin(3*math.pi*domain[1:-1])+
                                    10*np.sin(5*math.pi*domain[1:-1]),
                                    kind='cubic')
    output=[]
    J=7
    d=run_hints(domain, func, J=J, J_in=[0], hint_init=True)
    torch.save(d, Constants.outputs_path+str(J)+'fig2.pt')
    output=torch.load(Constants.outputs_path+str(J)+'fig2.pt')

    fig, ax = plt.subplots(1,2)
    fig.tight_layout(pad=3.0)
    modes=[1,5,10]
    headers=['iterations', '']
    labels=[str(k) for k in modes]
    data_x=[np.arange(500,600)]*3 
    data_y=[[o[k-1] for o in output['fourier_err'][500:600]] for k in modes]
  
    labels=['mode='+str(k) for k in modes]
    P=Plotter(ax[0],headers,data_x,data_y,labels,scale='log', title='fourier modes error, J='+str(J))
    P.plot_figure()


    P=Plotter(ax[1],headers,[None, None],[output['res_err'],output['err']],labels=['res. err', 'rel. error'],scale='log', title='solution error, J='+str(J))
    P.plot_figure()
        
    inset_axes(ax[1], 
                    width="50%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)
    plt.plot(output['err'][0:150],'b')
    # plt.scatter(36,output['err'][36])
    plt.xticks([0,150])
    plt.yticks([])
#plt.title('Probability')
    # plt.xticks([])
    # plt.yticks([]) 
    
    P.save_figure(fig,Constants.eps_fig_path+'fig2.eps')



def fig3(plot_fourier):
    N=100
    domain = np.linspace(0, 1, N)
    func=scipy.interpolate.interp1d(domain[1:-1],
                                    np.sin(math.pi*domain[1:-1])+
                                    5*np.sin(3*math.pi*domain[1:-1])+
                                    10*np.sin(5*math.pi*domain[1:-1]),
                                    kind='cubic')


    output=[]
    J=7

    d=run_hints(domain, func, J=J, J_in=[0], hint_init=True)
    torch.save(d, Constants.outputs_path+str(J)+'fig3.pt')
    output=torch.load(Constants.outputs_path+str(J)+'fig3.pt')  

    times=list(range(0,15))+list(range(84,94))
    hints_times=[]
    for i,t in enumerate(times):
        if t%7==0:
            hints_times.append(i)

    fig, ax = plt.subplots(5, 5)
    # make 1d for easier access
    ax = np.ravel(ax)
    for j,it in enumerate(times):
        headers=['', '']
        if j==0:
            labels=[None,None]
        else:    
            labels=[None,None]
        if plot_fourier:    
            data_x=[None, None]
            data_y=[output['solution_expansion'],output['x_expansion'][it]]
        else:    
            data_x=[domain[1:-1], domain[1:-1]]
            data_y=[output['solution'],output['x_k'][it]]

        P=Plotter(ax[j],headers,data_x,data_y,labels, title='')
        P.plot_figure()
        if j in hints_times:
            ax[j].text(0.5, 0.9, f'iter={it}', transform=ax[j].transAxes, fontsize=6,
                                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2))      
        else:
            ax[j].text(0.5, 0.9, f'iter={it}', transform=ax[j].transAxes, fontsize=6,
                                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))      

        ax[j].set_xticks([])
        ax[j].set_yticks([])
 

    ax[0].text(0.5, 0.2, f'exact',color='red', transform=ax[0].transAxes, fontsize=6,
                                ha='center', va='center')      
    ax = np.reshape(ax, (int(math.sqrt(len(times))), int(math.sqrt(len(times)))))    

    if plot_fourier:
        fig.suptitle(f'fourier modes along iterations, J={J}')    
        P.save_figure(fig,Constants.eps_fig_path+'fig3a.eps')
    else:
        fig.suptitle(f'solution along iterations, J={J}')    
        P.save_figure(fig,Constants.eps_fig_path+'fig3b.eps')

def fig5(N=100):
    domain = np.linspace(0, 1, N)
    val=np.sin(math.pi*domain[1:-1])+5*np.sin(3*math.pi*domain[1:-1])+10*np.sin(5*math.pi*domain[1:-1])
    func=scipy.interpolate.interp1d(domain[1:-1],val,kind='cubic')
    func=scipy.interpolate.interp1d(domain, grf(domain,1,mu=0,sigma=0.1))


    output=[]
    all_J=[5,10,20]
    plot_J=[5,10,20]
    conv_rates=[]
    for j in all_J:
        d=run_hints(domain, func, J=j, J_in=[0], hint_init=True)
        torch.save(d, Constants.outputs_path+str(j)+'fig5.pt')
        output.append(torch.load(Constants.outputs_path+str(j)+'fig5.pt'))

    output_J=[torch.load(Constants.outputs_path+str(j)+'fig5.pt')  for j in plot_J ]

    fig,ax=plt.subplots(1,3)
    fig.tight_layout(pad=5.0)

    headers=['iterations', 'error']
    data_x=[None for o in output_J] 
    data_y=[o['err'] for o in output_J] 
    labels=['J='+str(o['J']) for o in output_J]

  
    P1=Plotter(ax[0],headers,data_x,data_y,labels, title=f'relative error, N={N}, k={Constants.k}', scale='log')
    P1.plot_figure()
    conv_rates=[-np.polyfit(np.arange(len(o['err']))/( (len(o['err'])-1)*0+1 ),np.log(o['err']),1)[0] for o in output]

    P2=Plotter(ax[1],['J','conv.rate'],[all_J],[conv_rates],labels=['conv. rates'], title=f'convergence rates, N={N}, k={Constants.k}')
    P2.plot_figure()

    P3=Plotter(ax[2],['x', 'u(x)'], [domain[1:-1], domain[1:-1]],[output[0]['solution'], output[0]['x_k'][0]],labels=['exact', 'NN'])
    P3.plot_figure()
   

    P3.save_figure(fig,Constants.eps_fig_path+'grf_error_iter_different_J_N_'+str(N)+'_k_'+str(Constants.k)+'.eps')

    plt.show()



fig1()

# fig2()
# fig3(False)
# fig3(True)
# fig5()

