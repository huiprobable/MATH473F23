import numpy as np

def Poisson1D_SOR_OneStep(b, u, stencil = [-1, 2, -1], omega=1.5):
    n = u.shape[0]
    r = 0
    for i in range(1, n-1):
        u[i] = (1-omega)*u[i]+omega*(b[i]-stencil[0]*u[i-1]-stencil[2]*u[i+1])/stencil[1]
    for i in range(1, n-1):
        r += (b[i]-stencil[0]*u[i-1]-stencil[1]*u[i]-stencil[2]*u[i+1])**2
    r = np.sqrt(r)
    return u, r

def Poisson1D_Jacobi_OneStep(b, u, stencil = [-1, 2, -1]):
    v = np.copy(u)
    n = u.shape[0]
    for i in range(1, n-1):
        v[i] = (b[i]-stencil[0]*u[i-1]-stencil[2]*u[i+1])/stencil[1]
    r = 0
    for i in range(1, n-1):
        r += (b[i]-stencil[0]*u[i-1]-stencil[1]*u[i]-stencil[2]*u[i+1])**2
    r = np.sqrt(r)
    return v, r

def Poisson1D_GS_OneStep(b, u, stencil = [-1, 2, -1]):
    n = u.shape[0]
    r = 0
    for i in range(1, n-1):
        u[i] = (b[i]-stencil[0]*u[i-1]-stencil[2]*u[i+1])/stencil[1]
    for i in range(1, n-1):
        r += (b[i]-stencil[0]*u[i-1]-stencil[1]*u[i]-stencil[2]*u[i+1])**2
    r = np.sqrt(r)
    return u, r

methods = {'SOR':Poisson1D_SOR_OneStep, 'Jacobi': Poisson1D_Jacobi_OneStep, 'GS': Poisson1D_GS_OneStep}

def Poisson1D(f, n, domain=[0,1], bdry_cond=[0,0], eps=1e-5, stencil=[-1, 2, -1], method='GS'):
    # set up the numerical parameters, as well as initial step. 
    x = np.linspace(domain[0],domain[1],n+1)
    h = (domain[1]-domain[0])/n
    b = f(x)*h**2
    u = np.zeros(n+1)
    u[0] = bdry_cond[0]
    u[-1] = bdry_cond[1]
    # for loop or while loop for the iterations. 
    num_iter = 0
    r = 1
    R = np.array([])
    while r>eps and num_iter<100000:
        u, r = methods[method](b, u, stencil=stencil)
        num_iter += 1
        R = np.append(R,r)
    return u, num_iter, R