import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from tabulate import tabulate

class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        return torch.linalg.norm(x-y)/(torch.linalg.norm(y)+1e-10)
    @classmethod
    def relative_L1(cls,x,y):
        return torch.nn.L1Loss()(x,y)/(torch.nn.L1Loss(y,y*0)+1e-10)
    
def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params    


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, log_path, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.path = log_path

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                self.path+'best_model.pth',
            )

def Gauss_zeidel(A, b, x, theta):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
          
            x_new[i] = (1-theta)*x[i]+ theta*(b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(A@x-b)/np.linalg.norm(b)<1e-15:
             x = x_new
             return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]
            
        x = x_new

    return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]      


class interpolation_2D:
    def __init__(self, X,Y,values):
        self.rbfi = Rbf(X, Y, values)

    def __call__(self, x,y):
        return list(map(self.rbfi,x,y  ))
    
    

def plot_table(headers, data, path=None):
    try:
       
        print(tabulate(data, headers=headers, tablefmt='orgtbl'), file=path)
    except:
        print(tabulate(data, headers=headers, tablefmt='orgtbl'))
    




def gmres(A, b, x0, nmax_iter, tol):
    b_start=b.copy()
    r = b - np.asarray(A@x0).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
        y = np.asarray(A@ q[k]).reshape(-1)

        for j in range(k + 1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b)[0]

        C=np.dot(np.asarray(q).transpose(), result) + x0
        x.append(C)
        if (np.linalg.norm(A@C-b_start)/np.linalg.norm(b_start))<tol:
            return C,k


    return C, k



class Plotter:
    def __init__(self,headers,data_x,data_y,labels, **kwargs) -> None:
        self.headers=headers
        self.data_x=data_x
        self.data_y=data_y
        self.colors=['red','blue','green','black', 'orange']
        self.linestyles=['solid']*5
        self.labels=labels
        self.fig, self.ax=plt.subplots()

        try:
            self.fig.suptitle(kwargs['title'])
        except:
            pass    
        
    def plot_figure(self):
        
        for i in range(len(self.data_x)):
            self.plot_single(self.headers,[self.data_x[i],self.data_y[i]],color=self.colors[i],label=self.labels[i])
        if len(self.data_x)>1:
            self.fig.legend()
        self.ax.set_yscale("log")   

        plt.show(block=False)
            
    
    def save_figure(self, path):
         self.fig.savefig(path, format='eps', bbox_inches='tight')
         plt.show(block=True)

    def plot_single(self,headers, data, **kwargs ):
            try:
                self.ax.plot(data[0],data[1],label=kwargs['label'],color=kwargs['color'])
            except:
                self.ax.plot(data[1],label=kwargs['label'],color=kwargs['color'])    
            self.ax.set_xlabel(headers[0])
            self.ax.set_ylabel(headers[1])
            plt.show(block=False)


    
def example():
    d={'a':1, 'b':2}
    return d


