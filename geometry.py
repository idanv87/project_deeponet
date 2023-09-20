import datetime
import os
import sys
import time

from shapely.geometry import Polygon as Pol2
from pylab import figure
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.spatial.distance import euclidean, cityblock
import cmath

from utils import *
from constants import Constants
# from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex
from functions.functions import Test_function

from packages.my_packages import *







    

    
class mesh:
    ''''
        points=np.random.rand(5,2)
        points=[points[i] for i in range(5)]
        mesh(points)
    '''
    def __init__(self, points):
        #  points (50,2) as list of arrays[[],[],[]..]

        self.points=points.copy()
        self.p=[]
        for i in range(len(self.points)):
            l=self.points.copy()
            point=l[i].reshape(1,2)
            l.pop(i)
            nbhd=np.array(l)
            #  50,2
            X=np.vstack((point,nbhd))
            values=np.hstack((1,np.zeros(nbhd.shape[0])))
            self.p.append(interpolation_2D(X[:,0],X[:,1,], values))
            # self.p.append(scipy.interpolate.RBFInterpolator(X,values, function='gaussian'))
            self.T=1000

class Polygon:
    def __init__(self, generators):
        self.T=-100
        self.generators = generators
        self.n=self.generators.shape[0]
        self.geo = dmsh.Polygon(self.generators)
        self.fourier_coeff = self.fourier()
        
        

    def create_mesh(self, h):

        # if np.min(calc_min_angle(self.geo)) > (math.pi / 20):
        start=time.time()
        X, cells = dmsh.generate(self.geo, h)

        X, cells = optimesh.optimize_points_cells(
            X, cells, "CVT (full)", 1.0e-6, 120
        )
        print(f'triangulation generated with time= {time.time()-start}')
        self.X=X
        self.cells=cells
        
    # else:
    #     self.plot()    
        # dmsh.show(X, cells, self.geo)

        self.vertices = X
        self.sc = simplicial_complex(X, cells)
        self.M = (
            (self.sc[0].star_inv)
            @ (-(self.sc[0].d).T)
            @ (self.sc[1].star)
            @ self.sc[0].d
        )
        self.interior_points=[]
        self.interior_indices=[]

        for j,x in enumerate(X):
            if (not on_boundary(x,self.geo)):
                self.interior_points.append(x)
                self.interior_indices.append(j)
        
        self.interior_points=np.array(self.interior_points)
        try:
            self.hot_points=self.find_hot_points()
            self.hot_indices=self.find_hot_indices()
        except:
                self.hot_points=None
                self.hot_indices=None
        self.ev, self.V = self.laplacian()
        print(f'geometry generated with time= {time.time()-start}')
        # self.radial_functions=self.radial_basis()
        self.radial_functions=None
     
    



    def laplacian(self):
        ev,V=scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:, self.interior_indices],k=40,
            return_eigenvectors=True,
            which="SR",
        )
        return ev,V

    def find_hot_points(self):
        base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
        pts=base_rect['interior_points']
        return np.array([closest(self.interior_points,pts[i])[0] for i in range(pts.shape[0]) ])
    
    def find_hot_indices(self):
        base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
        pts=base_rect['interior_points']
        return np.array([closest(self.interior_points,pts[i])[1] for i in range(pts.shape[0]) ])
    
    def is_legit(self):
        if np.min(abs(self.sc[1].star.diagonal())) > 0:
            return True
        else:
            return False

    def save(self, path):
        assert self.is_legit()
        data = {
            "vertices":self.vertices,
            "ev": self.ev,
            "principal_ev": self.ev[-1], 
            "interior_points": self.interior_points,
            "hot_points": self.hot_points,
            "hot_indices":self.hot_indices,
            # "hot_points": self.hot_points[np.lexsort((self.hot_points[:,1], self.hot_points[:,0]))],
            "generators": self.generators,
            "M": self.M[self.interior_indices][:, self.interior_indices],
            'radial_basis':self.radial_functions,
             'angle_fourier':self.fourier_coeff,
             'translation':self.T,
             'cells':self.cells,
             'X':self.X,
             'geo':self.geo,
             'V':self.V,
            "legit": True,
            'type': 'polygon'
        }
        torch.save(data, path)

    def plot2(self):
        plt.scatter(self.interior_points[:, 0],
                    self.interior_points[:, 1], color='black')
        plt.show()

    def radial_basis(self):
        m=mesh([self.vertices[i] for i in range(self.vertices.shape[0])])
        return [m.p[i] for i in self.interior_indices]
    
    
    def fourier(self):
        x1=self.generators[:,0]
        y1=self.generators[:,1]
        dx=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]

        theta=[np.arctan2(y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]) for k in range(x1.shape[0])]

        l=[h/np.sum(dx) for h in dx]

        coeff=step_fourier(l,theta)
        return coeff
    @classmethod
    def plot(cls,generators, title='no title was given'):
        assert generators.shape[1]==2
        x1=generators[:,0]
        y1=generators[:,1]
        polygon = Pol2(shell=[[x1[k],y1[k]] for k in range(x1.shape[0])],holes=None)
        fig, ax = plt.subplots()
        ax.set_title(title)
        plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
        


        
    @classmethod
    def plot_geo(cls,x,y,geo):
        dmsh.show(x, y, geo)




# geo= dmsh.Rectangle(-1, +1, -1, +1)- dmsh.Rectangle(-0.5, 0.5, -0.5, 0.5)

# X, cells = dmsh.generate(geo, 0.2)
# boundary=[]
# ind=[]
# for i,x in enumerate(X):
#     if on_boundary(x,geo):
#         boundary.append(x)
#         ind.append(i)
# plt.scatter(X[ind,0], X[ind,1]);plt.show()
# dmsh.show(X, cells, geo)


class Annulus(Polygon):
    def __init__(self, generators,T):
        self.generators = generators
        self.n=self.generators.shape[0]
        self.geo =dmsh.Rectangle(0, 1, 0, 1)- dmsh.Polygon(self.generators)
        self.fourier_coeff = self.fourier()
        self.T=T












 