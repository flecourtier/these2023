include "getARGV.idp"
load "iovtk"
load "MUMPS"
//load "PETSc"

// Mesh
real xmin = -1;
real xmax =  1;
real ymin = -1;
real ymax =  1;

int hot = 1;
int cold = 2;
int adiab = 3;
border left(t=ymin,ymax){ x=xmin; y=ymax+ymin-t ; label= hot;}
border right(t=ymin,ymax){ x=xmax; y=t; label= cold;}
border bot(t=xmin,xmax){ x=t; y=ymin; label= adiab;}
border top(t=xmin,xmax){ x=xmax+xmin-t; y=ymax; label= adiab;}

int n = getARGV("-n",100);
mesh Oh = buildmesh(
	left(n) + bot(n) + right(n) + top(n)
);

//if (mpirank==0){
//	savevtk("initial_mesh.vtu", Oh);
//}
savevtk("initial_mesh.vtu", Oh);

// parameters
real cp = 1.0;
real rho = 1.0;
real cond = getARGV("-cond", 1.0);
real nu = getARGV("-nu", 1.0);
real Thot  = 1.0;
real Tcold = -1.0;
real bulk = 0.1;
real gx = 0;
real gy = -10.0;

// Approximated solution space
fespace Vh2(Oh, P2);
Vh2 u, v;
Vh2 du, dv;
Vh2 uh, vh;
Vh2 dT, T, w;
Vh2 uBest, vBest, pBest, TBest;

fespace Vh1(Oh, P1);
Vh1 dp, p, q;

u[] = 0;
v[] = 0;
p[] = 0;
T[] = 0;

int nGuess = getARGV("-nguess", 5);
problem NVSGuess(u, v, p, uh, vh, q, solver=CG, tgv=-1) =
	int2d(Oh)(
		// gradient pressure
		1/rho * (dx(p) * uh + dy(p) * vh)
		// laplacian
		+ nu * (dx(u)*dx(uh) + dy(u)*dy(uh) + dx(v)*dx(vh) + dy(v)*dy(vh))
		// penalization of the continuity equation
		- q * (dx(u)+dy(v))
	)
	+ int2d(Oh)(
		// buoyancy term
		bulk*T*(gx*uh + gy*vh)
	)
	- int2d(Oh)(
		(gx*uh+gy*vh)
	)
	// velocity/no slip (u,v)
	+ on(adiab, hot, cold, u = 0, v = 0);

problem HeatGuess(T, w, solver=sparsesolver) =
	// Heat equation
	int2d(Oh)(
		cp * w * (u*dx(T) + v*dy(T))
		+ cond/rho*(dx(T)*dx(w) + dy(T)*dy(w))
	)
	// dirichlet conditions on T (hot/cold)
	+ on(hot, T = Thot)
	+ on(cold, T = Tcold);

for(int i=0; i<nGuess; i++){
	cout << "Guess iteration "<< i+1 <<endl;
	NVSGuess;
	HeatGuess;
}
int[int] Order = [1,1,1];
savevtk("initial_guess.vtu", Oh, [u,v,0], p, T, dataname="velocity p T", order=Order);

// Newton loops
int nNewton = getARGV("-newton", 15);
int iter;
real err, errBest=100;

problem NvsHeat([du, dv, dp, dT], [uh, vh, q, w], solver=sparsesolver) =
	//-----------------------
	// DF
	//-----------------------
	// Heat equation
	int2d(Oh)(
		cp * w * (u*dx(dT) + v*dy(dT) + du*dx(T) + dv*dy(T))
		+ cond/rho*(dx(dT)*dx(w) + dy(dT)*dy(w))
	)
	// dirichlet conditions on T (hot/cold)
	+ on(hot, cold, dT = 0)
	// Navier-Stokes equations
	+ int2d(Oh)(
		// convection
		uh * (u*dx(du) + v*dy(du) + du*dx(u) + dv*dy(u))
		+ vh * (u*dx(dv) + v*dy(dv) + du*dx(v) + dv*dy(v))
		// gradient pressure
		+ 1/rho * (dx(dp) * uh + dy(dp) * vh)
		// laplacian
		+ nu * (dx(du)*dx(uh) + dy(du)*dy(uh) + dx(dv)*dx(vh) + dy(dv)*dy(vh))
		// buoyancy term
		+ bulk*dT*(gx*uh+gy*vh)
		// penalization of the continuity equation
		- q * (dx(du)+dy(dv))
		// add stabilization
		- dp * q * 1E-8
	)
	// velocity/no slip (u,v)
	+ on(adiab, hot, cold, du = 0, dv = 0)
	//-----------------------
	// F
	//-----------------------
	// Heat equation
	- int2d(Oh)(
		cp * w * (u*dx(T) + v*dy(T))
		+ cond/rho*(dx(T)*dx(w) + dy(T)*dy(w))
	)
	// Navier-Stokes equations
	- int2d(Oh)(
		// convection
		uh * (u*dx(u) + v*dy(u))
		+ vh * (u*dx(v) + v*dy(v))
		// gradient pressure
		+ 1/rho * (dx(p) * uh + dy(p) * vh)
		// laplacian
		+ nu * (dx(u)*dx(uh) + dy(u)*dy(uh) + dx(v)*dx(vh) + dy(v)*dy(vh))
		// buoyancy term
		+ bulk*T*(gx*uh+gy*vh)
		// penalization of the continuity equation
		- q * (dx(u)+dy(v))
	)
	// RHS
	+ int2d(Oh)(
		gx*uh+gy*vh
	)
;

for(iter=0; iter<nNewton; iter++){
	// step
	NvsHeat;

	// variable updates
	u[] -= du[];
	v[] -= dv[];
	p[] -= dp[];
	T[] -= dT[];


        //real Lu=u[].linfty, Lv=v[].linfty, Lp=p[].linfty, LT=T[].linfty;
        err = du[].linfty/u[].linfty + dv[].linfty/v[].linfty + dp[].linfty/p[].linfty + dT[].linfty/T[].linfty;

	//	if(err<errBest){
	//	errBest = err;
	//	uBest[] = u[];
	//	vBest[] = v[];
	//	pBest[] = p[];
	//	TBest[] = T[];
	//}
	if (err<1e-4) break;

	cout << "Newon iteration = "<< iter << "| err = "<< err << endl;

}
//savevtk("resultsBest.vtu", Oh, [uBest, vBest, 0], pBest, TBest, dataname="velocity p T", order=Order);
savevtk("results.vtu", Oh, [u, v, 0], p, T, dataname="velocity p T", order=Order);

////////////////////////////////////////////////////////////
