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
from utils import  grf, extract_path_from_dir, plot_polygon

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D, Plot_Polygon 

from two_d_model import geo_deeponet, Deeponet
from shapely.geometry import Polygon
import shapely.plotting


model=geo_deeponet( 2, 77,2, 90)
# model=Deeponet(2,77)
experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

hint_names=[Constants.path+'hints_polygons/40_115000.pt']
trained_names=[Constants.path+'polygons/10_115000.pt']
source_domain=torch.load(trained_names[0])

domain=torch.load(hint_names[0])


x_domain=domain['interior_points'][:,0]
y_domain=domain['interior_points'][:,1]
angle=domain['angle_fourier'][:90]
translation=domain['translation']
L=domain['M']
sigma=0.1
mu=0


sample=grf(domain['interior_points'][:,0], 1, seed=8, sigma=sigma, mu=mu )
func=interpolation_2D(domain['interior_points'][:,0],domain['interior_points'][:,1],sample[0])
J=15
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

    for i in range(800):
        x_0 = x
        k_it += 1
        theta=1
       
        if (((k_it % J) in J_in) and (k_it > J_in[-1])):

            x1=A@x_0-b
            sigma=np.sqrt(np.var(x1))
            factor1=np.sqrt(2)*0.1/sigma
            factor=np.max(abs(sample[0]))/np.max(abs(A@x_0-b))
            factor=factor

            x_temp = x_0*factor + deeponet(model, (b-A@x_0)*factor )
            # print(f'NN iteration in time =={time.time()-start}')
          
            
            # deeponet(model, interpolation_2D(x_domain,y_domain,(b-A@x_0)*factor )) 
            
            x=x_temp/factor
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            start=time.time()
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]
            # print(f'GS iteration in time =={time.time()-start}')


        if (k_it %20)==0:
                print(f'error:{np.linalg.norm(A@x-b)/np.linalg.norm(b)}, iter={k_it}')
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
        if (res_err[-1] < 1e-13) and (err[-1] < 1e-13):
            return err, res_err
        else:
            pass


   
    return {'err':err, 'res_err':res_err, 'J':J, 'iterations':k_it}

def run_hints(func, J, J_in, hint_init):
    return network(model, func, J, J_in, hint_init)



# plot_solution(Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt', 'J='+str(J)+'k='+str(Constants.k)+'errors.pt')


def table1(N):
    domain = np.linspace(0, 1, N)

    func=scipy.interpolate.interp1d(domain[1:-1],
                                    np.sin(math.pi*domain[1:-1])+
                                    5*np.sin(3*math.pi*domain[1:-1])+
                                    10*np.sin(5*math.pi*domain[1:-1]),
                                    kind='cubic')

        # func=scipy.interpolate.interp1d(domain, grf(domain,1,mu=0,sigma=0.1))


    output=[]
    all_J=list(range(80,81,1))
    all_J=[20,50,80,100,120]
 
    for j in all_J:
        d=run_hints(domain, func, J=j, J_in=[0,1], hint_init=True)
        torch.save(d, Constants.outputs_path+str(j)+str(N)+'tab1.pt')
        output.append(torch.load(Constants.outputs_path+str(j)+str(N)+'tab1.pt'))
    conv_rates=[-np.polyfit(np.arange(len(o['err']))/( (len(o['err'])-1)*0+1 ),np.log(o['err']),1)[0] for o in output]
    iterations=[o['iterations'] for o in output ]
    errors=[o['err'] for o in output ]
    ind=np.argmax(conv_rates)
    return {'N':N,'k':Constants.k,'J':all_J[ind],'conv_rates':conv_rates[ind], 'iter':iterations[ind], 'err':errors[ind][-1]}


torch.save(run_hints(func, J=J, J_in=[0], hint_init=True), Constants.outputs_path+'J='+str(J)+'k='+str(Constants.k)+'errors.pt')

# table=[]
# for N in [240]:
#     d=table1(N)
#     table.append([d[name] for name in list(d)])







