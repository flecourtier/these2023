import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from scimba.equations import domain

from scar.geometry.Geometry2D import ParametricCurves,Circle
from scar.utils import read_config
from scar.equations.run_Eikonal2D import run_Eikonal2D
from scar.equations.run_EikonalLap2D import run_EikonalLap2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current = Path(__file__).parent.parent.parent.parent

class SDCircle(domain.SignedDistance):
    def __init__(self, form : Circle, threshold: float = 0.0):
        """Circle domain centered at (x_0,y_0) with radius r
        """
        super().__init__(2, threshold)

        self.x0,self.y0 = form.x0,form.y0
        self.r = form.r
        self.eps = 0.5-self.r
        self.bound_box = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def phi(self,pre,xy):
        """Level set function for the circle domain

        :param x: x coordinate
        :param y: y coordinate
        :return: Level set function evaluated at (x,y)
        """
        x,y = xy
        return -self.r**2+(x-self.x0)**2+(y-self.y0)**2
    
    def sdf(self,x):
        """Level set function for the circle domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x1, x2 = self.get_coordinates(x)
        return self.phi(None,[x1,x2])
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain
        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(xy)
    
    def call_Omega(self, xy):
        """Returns True if (x,y) is in the Omega domain
        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi(None,xy)<0

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

class SDEikonal(domain.SignedDistance):
    def __init__(self, form : ParametricCurves, form_config, threshold: float = 0.0):
        super().__init__(2, threshold)

        self.form = form
        self.form_config = form_config
        self.bound_box = [[form.bord_a,form.bord_b],[form.bord_a2,form.bord_b2]]

        self.init_eik()

        self.mu = torch.tensor([])

    def init_eik(self):
        class_name = self.form.__class__.__name__
        form_dir_name = current / "networks" / "Eikonal2D" / class_name / "models" / ("config_"+str(self.form_config)+".json")
        dict_config = read_config(form_dir_name)
        self.eik_pinns, self.form_trainer = run_Eikonal2D(self.form,self.form_config,dict_config)
        self.pde = self.eik_pinns.pde

    # def phi(self,pre,xy):
    #     """Level set function for the circle domain

    #     :param x: x coordinate
    #     :param y: y coordinate
    #     :return: Level set function evaluated at (x,y)
    #     """
    #     x,y = xy
    #     return -self.r**2+(x-self.x0)**2+(y-self.y0)**2
    
    def sdf(self,x):
        """Level set function for the circle domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.form_trainer(x,self.mu)

class SDEikonalReg(SDEikonal):
    def __init__(self, form : ParametricCurves, form_config, threshold: float = 0.0):
        super().__init__(form, form_config, threshold)

    def init_eik(self):
        class_name = self.form.__class__.__name__
        form_dir_name = current / "networks" / "EikonalRegularised2D" / class_name / "models" / ("config_"+str(self.form_config)+".json")
        dict_config = read_config(form_dir_name)
        self.eik_pinns, self.form_trainer = run_Eikonal2D(self.form,self.form_config,dict_config,regularised=True)
        self.pde = self.eik_pinns.pde

class SDEikonalLap(SDEikonal):
    def __init__(self, form : ParametricCurves, form_config, threshold: float = 0.0):
        super().__init__(form, form_config, threshold)

    def init_eik(self):
        class_name = self.form.__class__.__name__
        form_dir_name = current / "networks" / "EikonalLap2D" / class_name / "models" / ("config_"+str(self.form_config)+".json")
        dict_config = read_config(form_dir_name)
        self.eik_pinns, self.form_trainer = run_EikonalLap2D(self.form,self.form_config,dict_config)
        self.pde = self.eik_pinns.pde