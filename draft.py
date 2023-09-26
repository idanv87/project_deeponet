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
    a=f[domain['hot_indices']]
    return a
   
    


def generate_domains(S,T,n1,n2):

    x1=[3/4,3/8,1/4,3/8]
    y1=[1/2,5/8,1/2,3/8] 
    X=[]
    Y=[]
    for j in range(len(x1)):
           
        p=np.array([x1[j],y1[j]])
        new_p=S@(p-np.array([0.5,0.5]))+np.array([0.5,0.5])+T
        X.append(new_p[0])
        Y.append(new_p[1])

    domain=Annulus(np.vstack((np.array(X),np.array(Y))).T, T)
    # domain.plot(domain.generators)
                # plt.gca().set_aspect('equal', adjustable='box')  
             
                # plt.xlim([0,1])
                # plt.ylim([0,1])

    domain.create_mesh(1/20)         
    # domain.save(Constants.path+'polygons/10_1150'+str(n1)+str(n2)+'.pt')
    domain.save(Constants.path+'hints_polygons/20_1150'+str(n1)+str(n2)+'.pt')

                # domain.create_mesh(0.05)
                # domain.save(Constants.path+'hints_polygons/005_1150'+str(n1)+str(n2)+'.pt')

                # # domain.plot_geo(domain.X, domain.cells, domain.geo)
                
                # domain.plot_geo(domain.X, domain.cells, domain.geo)
                # domain.save(Constants.path+'hints_polygons/80_tta_2_1150'+str(n1)+str(n2)+'.pt')
                # print('sucess')
                
            # except:
                # print('failed')    


if __name__=='__main__':
    pass

    # base_domain=Polygon(np.array([[-1,-1],[1,-1],[1,1],[-1,1]]))
    # base_domain=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
    # base_domain.create_mesh(0.1)
    # base_domain.save(Constants.path+'base_polygon/base_rect.pt')

    fig,ax=plt.subplots()
    for i,theta in enumerate(np.linspace(-math.pi/4,math.pi/4,7)[1:-1]):
        for j,theta2 in enumerate(np.linspace(3*math.pi/4,5*math.pi/4,7)[1:-1]):  
                theta=math.pi/4
                T=np.array([np.cos(theta2), np.sin(theta2)])/4
                S=np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
                generate_domains(S,T,i,j)
                sys.exit()
                

                                              
            
            
# domain=torch.load(Constants.path+'polygons/10_115000.pt')
# Polygon.plot_geo(domain['X'], domain['cells'], domain['geo'])

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