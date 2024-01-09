from forms import *

def prod_scal(X,Y):
    x,y = X
    xp,yp = Y
    return x*xp+y*yp

def rect_method(f,a,b,N):
    t = np.linspace(a,b,N)
    dt = t[1]-t[0]
    val = 0
    for i in range(N):
        val += f(t[i])
    return val*dt

class MVP:
    def __init__(self,form):
        assert isinstance(form,ParametricCurves)
        self.form = form

    def W_p(self,x,p=1):
        def W_p_t(t):
            c_t = self.form.c(t)
            c_prime_rot_t = self.form.c_prime_rot(t)

            num = prod_scal(c_t[:,None,None]-x,c_prime_rot_t[:,None,None])
            den = np.linalg.norm(c_t[:,None,None]-x,axis=0)**(2+p)

            return num/den
        
        # intégration numérique
        N = 10000
        val = rect_method(W_p_t,0,1,N)

        return val

    def phi(self,x,p=1):
        val = 1./self.W_p(x,p)
        val[np.isnan(val)] = 0
        return val**(1./p)
    
    def plot_phi(self):
        N = 200

        lin = np.linspace(self.form.bord_a,self.form.bord_b,N)
        lin2 = np.linspace(self.form.bord_a2,self.form.bord_b2,N)

        XX,YY = np.meshgrid(lin,lin2)

        plt.figure(figsize=(15,5))

        plt.subplot(1,2,1)
        val = self.phi(np.array([XX,YY]))
        plt.contourf(XX,YY,val,levels=100)
        self.form.plot_curve()
        plt.title("phi")
        plt.colorbar()

        plt.subplot(1,2,2)
        pos_val = val>0
        val[pos_val] = 0.
        plt.contourf(XX,YY,-val,levels=100)
        self.form.plot_curve()
        plt.title("phi (zero outside)")
        plt.colorbar()

        plt.savefig("MVP_"+self.form.name+".png")
        plt.show()
    
form = Bean()
mvp = MVP(form)
mvp.plot_phi()