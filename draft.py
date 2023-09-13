import os
import sys
import math

# from shapely.geometry import Polygon as Pol2
from airfoils import Airfoil
import dmsh
import meshio
import optimesh
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import scipy
import torch

import sys
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.stats import qmc
import pandas as pd



from geometry import Polygon, Annulus
from utils import extract_path_from_dir, save_eps, plot_figures, grf

from constants import Constants

from two_d_data_set import create_loader 


def loss(a,*args):
        basis,f, x,y=args
        assert len(a)==len(basis)
        return np.linalg.norm(np.sum(np.array([a[i]*func(np.array([x, y]).T) for i,func in enumerate(basis)]),axis=0)-f)**2


def create_data(domain):
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    M=domain['M']
    angle_fourier=domain['angle_fourier']
    T=domain['translation']
    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    test_functions=domain['radial_basis']
    V=[func(np.array([x, y]).T) for func in test_functions]
    F=[v for v in V]
    U=[scipy.sparse.linalg.spsolve(A,b) for b in F]

    


    return x,y,F, U, angle_fourier, T

def expand_function(f,domain):
    # f is a vector of f evaluated on the domain points
    
    base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
    # base_rect=torch.load(Constants.path+'/base_polygon/base_rect.pt')
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    basis=base_rect['radial_basis']
    # plt.scatter(x,y,c=basis[10](np.array([x, y]).T))
    # plt.show()
    phi=np.array([func(np.array([x, y]).T) for func in basis]).T
    a=np.linalg.solve(phi.T@phi,phi.T@f)
    approximation=np.sum(np.array([a[i]*func(np.array([x, y]).T) for i,func in enumerate(basis)]).T,axis=1)
    error=np.linalg.norm(approximation-f)/np.linalg.norm(f)
    if error>1e-10:
         print(f'expansion of f is of error  {error}')
         
    
    return a

    #   x0=np.random.rand(len(basis),1)
    # res = minimize(loss, x0, method='BFGS',args=(basis,f,x,y), options={'xatol': 1e-4, 'disp': True})
    # return res.x

    
    


def generate_domains(S,T,n1,n2):
    names=extract_path_from_dir(Constants.path+'my_naca/')
    # for i,name in enumerate(os.listdir(Constants.path+'my_naca/')):
    for i,f in enumerate(names):
           
            file_name=f.split('/')[-1].split('.')[0]

            x1,y1=torch.load(f)
            lengeths=[np.sqrt((x1[(k+1)%x1.shape[0]]-x1[k])**2+ (y1[(k+1)%x1.shape[0]]-y1[k])**2) for k in range(x1.shape[0])]
            
            X=[]
            Y=[]
            for j in range(len(lengeths)):
                    if lengeths[j]>0:
                        p=0.7*np.array([x1[j]-0.5,y1[j]])
                        new_p=S@p+T
                        X.append(new_p[0])
                        Y.append(new_p[1])
            try:    
                
                # domain=Polygon(np.array([[0,0],[1,0],[2,1],[0,1]])) 
                domain=Annulus(np.vstack((np.array(X),np.array(Y))).T, T)

               
                
                domain.save(Constants.path+'polygons/1150'+str(n1)+str(n2)+'.pt')
                
                # domain.create_mesh(0.2)
                # domain.save(Constants.path+'hints_polygons/01_1150'+str(n1)+str(n2)+'.pt')
                print('sucess')
                domain.plot_geo(block=False)
            except:
                print('failed')    


if __name__=='__main__':
    pass

    # base_domain=Polygon(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))
    # # base_domain=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
    # base_domain.create_mesh(0.2)
    # base_domain.plot_geo()
    # base_domain.save(Constants.path+'base_polygon/base_rect.pt')

    for i,theta in enumerate(np.linspace(0,2*math.pi,10)):
        for j,T in enumerate(0.9*grf(list(range(2)),10)):
            S=np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
            generate_domains(S,T,i,j)

# base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
# print(base_rect['interior_points'].shape)







# if __name__=='__main__':
#     polygon = Pol2(shell=((0,0),(1,0),(1,1),(0,1)),
# holes=None
# fig, ax = plt.subplots()
#     plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
# plt.show()
####################################################################################################################################################################
    # p=torch.load(Constants.path+'polygons/rect.pt')
    # plt.scatter(p['interior_points'][:,0], p['interior_points'][:,1],c='b')
    # plt.scatter(p['hot_points'][:,0], p['hot_points'][:,1],c='r')
    # plt.title('interior points and hot points')
    # plt.show()
####################################################################################################################################################################







