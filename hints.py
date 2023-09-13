import os
import sys
import math
from matplotlib.ticker import ScalarFormatter


from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf


from constants import Constants
from utils import save_eps, plot_figures, grf
from main import  SonarDataset, generate_sample
from two_d_data_set import create_loader
from draft import create_data, expand_function
from geometry import Polygon
from packages.my_packages import Gauss_zeidel, interpolation_2D
from main import model


experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])

hint_names=[Constants.path+'polygons/115000.pt']
source_domain=torch.load(hint_names[0])
domain=source_domain
x_domain=domain['interior_points'][:,0]
y_domain=domain['interior_points'][:,1]
angle_fourier=domain['angle_fourier']
translation=domain['translation']
L=domain['M']
xi,yi,F,psi, temp1, temp2=create_data(domain)

sample=grf(F, 1, seed=1 )
func=interpolation_2D(x_domain,y_domain,generate_sample(sample[0],F, psi)[0] )



def deeponet(model, func):
    X_test_i=[]
    Y_test_i=[]
    a=expand_function(func(x_domain, y_domain), domain )
    # with torch.no_grad():
    #     y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(x_domain.shape[0],)
    #     f_temp=torch.tensor(a.reshape(1,a.shape[0]),dtype=torch.float32).repeat(x_domain.shape[0],1)
    #     tranlation_temp=torch.tensor(translation.reshape(1,translation.shape[0]),dtype=torch.float32).repeat(x_domain.shape[0],1)
    #     angle_temp=torch.tensor(angle_fourier.reshape(1,angle.shape[0]),dtype=torch.float32).repeat(x_domain.shape[0],1)
        
    #     pred2=model([y,f_temp, tranlation_temp,y*0, angle_fourier])
    # return pred2.numpy()


    for j in range(domain['interior_points'].shape[0]):
        X_test_i.append([
                        torch.tensor([x_domain[j],y_domain[j]], dtype=torch.float32), 
                         torch.tensor(a, dtype=torch.float32),
                         torch.tensor(translation, dtype=torch.float32),
                         torch.tensor(0, dtype=torch.float32),
                         torch.tensor(angle_fourier, dtype=torch.float32)
                         ])
        Y_test_i.append(torch.tensor(0, dtype=torch.float32))

    
    test_dataset = SonarDataset(X_test_i, Y_test_i)
    test_dataloader=create_loader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    coords=[]
    prediction=[]
    with torch.no_grad():    
        for input,output in test_dataloader:
            coords.append(input[0])
            prediction.append(model(input))

    coords=np.squeeze(torch.cat(coords,axis=0).numpy())
    prediction=torch.cat(prediction,axis=0).numpy()

    return prediction

def network(model, func, J, J_in, hint_init):
    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    print(ev)
    b=func(x_domain, y_domain)
    solution=scipy.sparse.linalg.spsolve(A, b)
    predicted=deeponet(model, func)
    # print(np.linalg.norm(solution-predicted)/np.linalg.norm(solution))


    if hint_init:
        x=deeponet(model, func)
    else:
        x=x=deeponet(model, func)*0

    res_err=[]
    err=[]
    k_it=0

    for i in range(400):
        x_0 = x
        k_it += 1
        theta=2/3
       
        if (((k_it % J) in J_in) and (k_it > J_in[-1])):
            
            # factor = np.max(abs(generate_sample(sample[0],F, psi)[0]))/np.max(abs(A@x_0-b))
            factor=np.max(abs(grf(F, 1)))/np.max(abs(A@x_0-b))
            x_temp = x_0*factor + \
            deeponet(model, interpolation_2D(x_domain,y_domain,(b-A@x_0)*factor )) 
            x=x_temp/factor
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       
        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
   


   
    return err, res_err

def run_hints(func, J, J_in, hint_init):
    return network(model, func, J, J_in, hint_init)
    # torch.save([ err_net, res_err_net], Constants.path+'hints_fig.pt')


torch.save(run_hints(func, J=5, J_in=[0], hint_init=True), Constants.outputs_path+'modes_error.pt')
# plot_solution_and_fourier(list(range(0,0+25)),Constants.outputs_path+'modes_error.pt', Constants.eps_fig_path+ 'one_d_x0_J=8_Jin=012_modes=1')
















