import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MVP():
    def __init__(self,pts_polygon):
        self.dim = pts_polygon.shape[1]
        self.pts_polygon = pts_polygon.to(device)
        self.pts_polygone_plus = torch.cat([self.pts_polygon,self.pts_polygon[0].unsqueeze(0)])
        self.nb_pts = len(pts_polygon)

    def det(self,X,Xp):
        x,y = X
        xp,yp = Xp
        return x*yp-y*xp 

    def prod_scal(self,X,Xp):
        x,y = X
        xp,yp = Xp
        return x*xp+y*yp

    def norm(self,X):
        return torch.sqrt(self.prod_scal(X,X))

    def get_tj(self,X,j=0,vectorized=True):
        if vectorized:
            # print("get_tj")
            Rj = (self.pts_polygon[:,:,None]-X.to(device)).permute(1,0,2)
            # print("Rj.shape",Rj.shape)
            Rjp = torch.roll(Rj,-1,1)
            # print("Rjp.shape",Rjp.shape)
            rj = self.norm(Rj)
            # print("rj.shape",rj.shape)
            rjp = torch.roll(rj,-1,0)
            # print("rjp.shape",rjp.shape)
            return self.det(Rj,Rjp)/(rj*rjp+self.prod_scal(Rj,Rjp))
        else:
            Xj = self.pts_polygone_plus[j].to(device)
            Rj = Xj[:,None]-X.to(device)
            rj = self.norm(Rj)
            Xjp = self.pts_polygone_plus[j+1].to(device)
            Rjp = Xjp[:,None]-X.to(device)
            rjp = self.norm(Rjp)
            return self.det(Rj,Rjp)/(rj*rjp+self.prod_scal(Rj,Rjp))

    def W(self,X, vectorized=True):
        if vectorized:
            add_dim = False
            if len(X.shape)==3:
                add_dim = True
                X = X[:,:,0]
            # print("X.shape",X.shape)
            # print("self.pts_polygon.shape",self.pts_polygon.shape)

            Rj = (self.pts_polygon[:,:,None]-X.to(device)).permute(1,0,2)
            # print("Rj.shape",Rj.shape)
            rj = self.norm(Rj)
            # print("rj.shape",rj.shape)
            rjp = torch.roll(rj,-1,0)
            # print("rjp.shape",rjp.shape)
            tj = self.get_tj(X)
            # print("tj.shape",tj.shape)

            val = torch.sum(tj*(1/rj+1/rjp),dim=0)
            # print("val.shape",val.shape)

            if add_dim:
                val = val[:,None]
            return val
        else:
            val = torch.zeros_like(X[0]).to(device)
            for j in range(self.nb_pts):
                Xj = self.pts_polygone_plus[j][:,None].to(device)
                rj = self.norm(Xj-X.to(device))
                Xjp = self.pts_polygone_plus[j+1][:,None].to(device)
                rjp = self.norm(Xjp-X.to(device))
                tj = self.get_tj(X.to(device),j,vectorized)
                val += tj.to(device)*(1/rj+1/rjp)
            return val
        
    def __call__(self, X, vectorized=True):
        val = -2./self.W(X, vectorized)
        val[torch.isnan(val)] = 0 # X on \partial\Omega => phi=0
        return val