import os
import sys
import math
from matplotlib.ticker import ScalarFormatter
import time

from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf


from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D

from two_d_model import geo_deeponet, Deeponet



model=geo_deeponet( 2, 77,2, 99)
# model=Deeponet(2,77)
experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

hint_names=[Constants.path+'hints_polygons/80_tta_2_115000.pt']
# hint_names=[Constants.path+'polygons/10_115050.pt']
source_domain=torch.load(hint_names[0])
domain=source_domain
x_domain=domain['interior_points'][:,0]
y_domain=domain['interior_points'][:,1]
angle=domain['angle_fourier']
translation=domain['translation']
L=domain['M']
sigma=0.1
mu=0


sample=grf(x_domain, 1, seed=4, sigma=sigma, mu=mu )
func=interpolation_2D(x_domain,y_domain,sample[0])
J=20
# J=5

def deeponet(model, func):
    # a=expand_function(func(x_domain   , y_domain), domain )
    a=expand_function(func,domain)

    with torch.no_grad():
       
        y1=torch.tensor(domain['interior_points'],dtype=torch.float32).reshape(domain['interior_points'].shape)
        a1=torch.tensor(a.reshape(1,a.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        tran1=torch.tensor(translation.reshape(1,translation.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        angle1=torch.tensor(angle.reshape(1,angle.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        moments1=angle*0

        pred2=model([y1, a1, tran1,moments1,angle1])
    return pred2.numpy()

   
def network(model, func, J, J_in, hint_init):
    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))

    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    assert ev[0]<Constants.k
    b=func(x_domain, y_domain)
    solution=scipy.sparse.linalg.spsolve(A, b)
    predicted=deeponet(model, b)
    # print(np.linalg.norm(solution-predicted)/np.linalg.norm(solution))


    if hint_init:
        x=deeponet(model, b)
    else:
        x=x=deeponet(model, b)*0

    res_err=[]
    err=[]
    k_it=0

    for i in range(1000):
        x_0 = x
        k_it += 1
        theta=1
       
        if (((k_it % J) in J_in) and (k_it > J_in[-1])):
            
            # factor = np.max(abs(sample[0]))/np.max(abs(A@x_0-b))
            x1=A@x_0-b
            sigma=np.sqrt(np.var(x1))
            factor1=np.sqrt(2)*0.1/sigma
            # factor=np.max(abs(grf(F, 1)))/np.max(abs(A@x_0-b))
            factor=factor1
 
            x_temp = x_0*factor + deeponet(model, (b-A@x_0)*factor )
          
            
            # deeponet(model, interpolation_2D(x_domain,y_domain,(b-A@x_0)*factor )) 
            
            x=x_temp/factor
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            start=time.time()
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]
            # print(f'GS iteration in time =={time.time()-start}')


        if (k_it %10)==0:
                print(f'error:{np.linalg.norm(A@x-b)/np.linalg.norm(b)}, iter={k_it}')
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
        if (res_err[-1] < 1e-13) and (err[-1] < 1e-13):
            return err, res_err
        else:
            pass


   
    return err, res_err

def run_hints(func, J, J_in, hint_init):
    return network(model, func, J, J_in, hint_init)


def plot_solution( path, eps_name):
    e_deeponet, r_deeponet= torch.load(path)
    
    fig3, ax3 = plt.subplots()   # should be J+1
    fig3.suptitle(F'relative error, \mu={mu}, \sigma={sigma} ')

    ax3.plot(e_deeponet, 'g')
    # ax3.plot(r_deeponet,'r',label='res.err')
    # ax3.legend()
    ax3.set_xlabel('iteration')
    ax3.set_ylabel('error')
    ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}', transform=ax3.transAxes, fontsize=6,
             ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    fig3.savefig(eps_name+'errors.eps', format='eps', bbox_inches='tight')
    plt.show(block=True)
    return 1


torch.save(run_hints(func, J=J, J_in=[0], hint_init=True), Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt')
# plot_solution(Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt', 'J='+str(J)+'k='+str(Constants.k)+'errors.pt')















