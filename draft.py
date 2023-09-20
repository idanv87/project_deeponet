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
import random

import sys
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.stats import qmc
import pandas as pd



from geometry import Polygon, Annulus
from utils import extract_path_from_dir, save_eps, plot_figures, grf, spread_points

from constants import Constants
from packages.my_packages import *
from two_d_data_set import create_loader 
import time

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
    V=[np.array(func(x,y)) for func in test_functions]
    F=[v for v in V]
    U=[scipy.sparse.linalg.spsolve(A,b) for b in F]

    


    return x,y,F, U, angle_fourier, T

def expand_function(f,domain):
    f=np.array(f)
    # interp=interpolation_2D(domain['interior_points'][:,0],domain['interior_points'][:,1],f)
    # # # f is a vector of f evaluated on the domain points
    # a=np.array(interp(domain['hot_points'][:,0], domain['hot_points'][:,1]) )
    a=f[domain['hot_indices']]
    return a
    # base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
    # base_rect=torch.load(Constants.path+'/base_polygon/base_rect.pt')
    # x=domain['interior_points'][:,0]
    # y=domain['interior_points'][:,1]
    # basis=base_rect['radial_basis']
    

    # # V=domain['V'][:,:10]

    # # a=[np.dot(V[:,i],f).real for i in range(V.shape[-1])]
    # # approximation=np.sum(np.array([a[i]*V[:,i] for i in range(len(a))]).T,axis=1)
    # # error=np.linalg.norm(approximation-f)/np.linalg.norm(f)

  
    # phi=np.array([func(x,y) for func in basis]).T
    # # # a,e=Least_squares(phi,f)
    # # start=time.time()
    # # # a,e=Least_squares(phi,f)
    # a=np.linalg.solve(phi.T@phi,phi.T@f)

    # approximation=np.sum(np.array([a[i]*np.array(func(x,y)) for i,func in enumerate(basis)]).T,axis=1)
    # error=np.linalg.norm(approximation-f)/np.linalg.norm(f)

    # # if np.linalg.matrix_rank(phi.T@phi)<77:
    # #     pass
    # #     print(np.linalg.matrix_rank(phi.T@phi))

    # #     #  print(f'expansion of f is of error  {error}')
         
    
    # return a

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
                        p=0.5*np.array([x1[j]+0.5,y1[j]+0.5])
                        new_p=S@p+T
                        X.append(new_p[0])
                        Y.append(new_p[1])
            try:    
                
                # domain=Polygon(np.array([[0,0],[1,0],[2,1],[0,1]])) 
                domain=Annulus(np.vstack((np.array(X),np.array(Y))).T, T)
                # domain.create_mesh(0.05)
                # domain.save(Constants.path+'hints_polygons/005_1150'+str(n1)+str(n2)+'.pt')
                domain.create_mesh(1/50)
                # domain.plot_geo(domain.X, domain.cells, domain.geo)
                domain.save(Constants.path+'polygons/50_1150'+str(n1)+str(n2)+'.pt')
                domain.plot_geo(domain.X, domain.cells, domain.geo)
                # domain.save(Constants.path+'hints_polygons/30_1150'+str(n1)+str(n2)+'.pt')
                print('sucess')
                
            except:
                print('failed')    


if __name__=='__main__':
    pass

    # base_domain=Polygon(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))
    # base_domain=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
    # base_domain.create_mesh(0.1)
    # base_domain.save(Constants.path+'base_polygon/base_rect.pt')

    for i,theta in enumerate(np.linspace(0,2*math.pi,10)):
        for j,T in enumerate(0.5*grf(list(range(2)),10)):
           if j==0: 
            S=np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
            generate_domains(S,T,i,j)
            sys.exit()

# base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')

# domain=torch.load(Constants.path+'polygons/10_115000.pt')
# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1],color='b')
# plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='r')
# plt.show()


# # dmsh.show(domain['X'], domain['cells'], domain['geo'])
# x=domain['interior_points'][:,0]
# y=domain['interior_points'][:,1]
# f=domain['radial_basis'][10]
# z=f(np.array([x,y]).T)
# expand_function(z,domain)


# dmsh.show(base_rect['X'], base_rect['cells'],base_rect['geo'])
# x=base_rect['interior_points'][:,0]
# y=base_rect['interior_points'][:,1]
# f=base_rect['radial_basis'][20]
# X,Y=np.meshgrid(np.linspace(-1,1,40)[1:-1],np.linspace(-1,1,40)[1:-1])
# # z=f(np.array([x,y]).T)
# z=f(np.array([X.ravel(),Y.ravel()]).T)


# domain=torch.load(Constants.path+'polygons/115000.pt')
# # domain['hot_points']=spread_points(70, domain['interior_points'])
# plt.scatter(domain['hot_points'][2,0], domain['hot_points'][2,1],color='r')
# dmsh.show(domain['X'], domain['cells'],domain['geo'])