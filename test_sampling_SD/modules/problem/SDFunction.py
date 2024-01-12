import numpy as np
import matplotlib.pyplot as plt
from scimba.equations import domain
from modules.problem.Geometry import ParametricCurves,Circle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDCircle(domain.SignedDistance):
    def __init__(self, form : Circle, threshold: float = 0.0):
        """Circle domain centered at (x_0,y_0) with radius r
        """
        super().__init__(2, threshold)

        self.x0,self.y0 = form.x0,form.y0
        self.r = form.r
        self.eps = 0.5-self.r
        self.bound_box = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def sdf(self,x):
        """Level set function for the circle domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x1, x2 = self.get_coordinates(x)
        return -self.r**2+(x1-self.x0)**2+(x2-self.y0)**2

class SDMVP(domain.SignedDistance):
    def __init__(self, form : ParametricCurves, threshold: float = 0.0):
        assert isinstance(form,ParametricCurves)
        super().__init__(2, threshold)
        self.form = form
        self.bound_box = [[self.form.bord_a,self.form.bord_b],[self.form.bord_a2,self.form.bord_b2]]

    def prod_scal(self,X,Y):
        return X[:,:,0]*Y[:,0]+X[:,:,1]*Y[:,1]

    def rect_method(self,f,a,b,N):
        t = np.linspace(a,b,N)
        dt = t[1]-t[0]
        val = torch.sum(f(t),dim=1)*dt
        return val

    def W_p(self,x,p=1):
        def W_p_t(tab_t):
            c_t = self.form.c(tab_t).to(device)
            c_prime_rot_t = self.form.c_prime_rot(tab_t).to(device)

            diff = torch.transpose(c_t,0,1)-x[:, None, :]
            num = self.prod_scal(diff,torch.transpose(c_prime_rot_t,0,1))
            den = torch.sqrt(torch.sum(diff**2,dim=2))**(2+p)

            return num/den
        
        # intégration numérique
        N = 1000
        val = self.rect_method(W_p_t,0,1,N)

        return val

    def sdf(self,x,p=1):
        val = 1./self.W_p(x,p)
        val[np.isnan(val.cpu().detach().numpy())] = 0
        return (val**(1./p))[:,None]
    
    # def plot_phi(self,savedir="./"):
    #     N = 200

    #     lin = np.linspace(self.form.bord_a,self.form.bord_b,N)
    #     lin2 = np.linspace(self.form.bord_a2,self.form.bord_b2,N)

    #     XX,YY = np.meshgrid(lin,lin2)

    #     plt.figure(figsize=(15,5))

    #     plt.subplot(1,2,1)
    #     val = self.sdf(np.array([XX,YY]))
    #     plt.contourf(XX,YY,val,levels=100)
    #     self.form.plot_curve()
    #     plt.title("phi")
    #     plt.colorbar()

    #     plt.subplot(1,2,2)
    #     pos_val = val>0
    #     val[pos_val] = 0.
    #     plt.contourf(XX,YY,-val,levels=100)
    #     self.form.plot_curve()
    #     plt.title("phi (zero outside)")
    #     plt.colorbar()

    #     plt.savefig(savedir+"MVP_"+self.form.name+".png")
    #     plt.show()
    
