import numpy as np
import torch

class Circle():
    def __init__(self):
        """Circle domain centered at (x_0,y_0) with radius r
        """
        self.x0,self.y0 = (0.5,0.5)
        self.r = np.sqrt(2)/4
        self.eps = 0.5-self.r
        self.domain_O = [[self.x0-self.r-self.eps,self.x0+self.r+self.eps],[self.y0-self.r-self.eps,self.y0+self.r+self.eps]]

    def phi(self, pre, xy):
        """Level set function for the circle domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x,y=xy
        return -self.r**2+(x-self.x0)**2+(y-self.y0)**2

    def phi_construct(self, pre, xy):
        """Level set function for the circle domain (for creation of the Omega domain)

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(pre, xy)
    
    def levelset(self,X):
        """Level set function for the circle domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(torch, X.T)
    
    def levelset_construct(self,X):
        """Level set function for the circle domain (for creation of the Omega domain)

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        """Returns True if (x,y) is in the Omega domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain

        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(None, xy)

class Square():
    def __init__(self):
        """Square domain with side length 1
        """
        self.eps = 0.5
        self.domain_O = [[0-self.eps,1+self.eps],[0-self.eps,1+self.eps]]

    def phi(self, pre, xy):
        """Level set function for the square domain

        :param pre: Preconditioner 
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        x,y=xy
        return x*(1-x)*y*(1-y)
    
    def phi_construct(self, pre, xy):
        """Level set function for the square domain (for creation of the Omega domain)

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return np.linalg.norm(np.array(xy)-0.5,np.inf,axis=0)-0.5
        # return torch.linalg.norm(xy-0.5,float('inf'),axis=0)-0.5

    def levelset(self,X):
        """Level set function for the square domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(torch, X.T)
    
    def levelset_construct(self,X):
        """Level set function for the square domain (for creation of the Omega domain)

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        """Returns True if (x,y) is in the Omega domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain

        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(None, xy)
    

class Random_domain:
    def __init__(self):
        """Random domain 
        """
        self.domain_O = [[0.,1.],[0.,1.]]
        self.n_mode = 4
        # np.random.seed(seed)
        # self.coefs = np.random.uniform(-1, 1, size=[self.n_mode, self.n_mode])
        # print(self.coefs)
        self.coefs = np.array([[6.03134517e-01,2.01975206e-01,6.23703508e-02,4.68719852e-02],[ 1.69200589e-02,-4.87544372e-02,1.89872347e-02,-8.39480121e-03],[ 7.69011926e-02,2.01064605e-02,-3.16876341e-03,5.43652473e-03],[-4.31846518e-03,8.54690021e-04,-7.36286438e-05,2.33468667e-03]])

    def phi(self, pre, xy):
        """Level set function for the square domain

        :param pre: Preconditioner 
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        val = 0.4*torch.ones_like(xy[0])
        for k in range(self.n_mode):
            for l in range(self.n_mode):
                val -= self.coefs[k,l]*torch.sin(np.pi*(k+1)*xy[0])*torch.sin(np.pi*(l+1)*xy[1])
        return val
    
    def phi_construct(self, pre, xy):
        """Level set function for the square domain (for creation of the Omega domain)

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        # return np.linalg.norm(np.array(xy)-0.5,np.inf,axis=0)-0.5
        return self.phi(pre, xy)

    def levelset(self,X):
        """Level set function for the square domain

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi(torch, X.T)
    
    def levelset_construct(self,X):
        """Level set function for the square domain (for creation of the Omega domain)

        :param X: (x,y) coordinates
        :return: Level set function evaluated at (x,y)
        """
        return self.phi_construct(torch, X.T)
    
    def call_Omega(self, pre, xy):
        """Returns True if (x,y) is in the Omega domain

        :param pre: Preconditioner
        :param xy: (x,y) coordinates
        :return: True if (x,y) is in the Omega domain
        """
        return self.phi_construct(pre,xy)<0
    
    def Omega_bool(self, x, y):
        """Returns True if (x,y) is in the Omega domain

        :param x: x coordinate
        :param y: y coordinate
        :return: True if (x,y) is in the Omega domain
        """
        xy = (x,y)
        return self.call_Omega(None, xy)

    # def generate_domain(size, i, n_mode=4, seed=2023):
    #     """Generate a random connected domain and associated levelset.

    #     Parameters:
    #         size (tuple): Size of the domain as (nx, ny).
    #         i (int): Index used for random seed.
    #         n_mode (int, optional): Number of modes for basis functions. Default is 4.
    #         seed (int, optional): Random seed. Default is 2023.

    #     Returns:
    #         domain (ndarray): Random connected domain as a binary numpy array.
    #         levelset (ndarray): Levelset associated with the domain.
    #         coefs (ndarray): Coefficients of the basis functions used in generating the domain.
    #     """

    #     batch_size = 1
    #     nx, ny = size
    #     x = np.linspace(0.0, 1.0, nx)
    #     y = np.linspace(0.0, 1.0, ny)
    #     modes = np.array(list(range(1, n_mode + 1)))

    #     def make_basis_1d(x):
    #         l = np.pi
    #         onde = lambda x: np.sin(x)
    #         return onde(l * x[None, :] * modes[:, None])

    #     basis_x = make_basis_1d(x)  # (n_mode,nx)
    #     basis_y = make_basis_1d(y)  # (n_mode,ny)
    #     basis_2d = (
    #         basis_x[None, :, None, :] * basis_y[:, None, :, None]
    #     )  # (n_mode_y, n_mode_x, n_y, n_x)

    #     if seed is not None:
    #         np.random.seed(seed + i)
    #     else:
    #         seed = seed + i
    #     coefs = np.random.uniform(-1, 1, size=[batch_size, n_mode, n_mode])

    #     coefs /= (modes[None, :, None] * modes[None, None, :]) ** 2
    #     levelset = 0.4 - (
    #         np.sum(
    #             coefs[:, :, :, None, None] * basis_2d[None, :, :, :, :],
    #             axis=(1, 2),
    #         )
    #     )
    #     domain = (levelset < 0.0).astype(int)
    #     return domain[0, :, :], levelset[0, :, :], coefs






