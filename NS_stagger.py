import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
#from scipy.sparse.linalg import inv
import datetime
from numpy.linalg import matrix_rank

#from numpy.linalg import pinv
#from numpy.linalg import matrix_rank
start_t = datetime.datetime.now()
'''
Nx: rows
Ny: Column
np.savetxt('matrix_A.dat', Ax, fmt = '%.2f')
'''

def init_A():
    '''
         ## Ax: dim [Nx*(Ny-1)), (Ny-1)*Nx]
         ## Ay: dim [(Nx-1*Ny), (Nx-1)*Ny) ]
         ## Gx: [ Nx*(Ny-1)*Nx, Ny*Nx]  Tgx: [Ny-1,Ny]
         ## Gy: [(Nx-1)*Ny, Ny*Nx]
         ## Dx: [Ny*Nx, (Ny-1)*Nx];  Tdx: Ny*(Ny-1)；        
         ## Dy: [(Nx*Ny),  (Nx-1*Ny)]
         The bottom of where u[i,j] = -u[i,j+1]; therefore, the first and last T are
         [-5  1  0 ...
           1  -5  1  ...
           0... 1 -5 1     ]
           
    '''
    Tx = sparse.spdiags([np.ones(Ny-1), -4*np.ones(Ny-1), np.ones(Ny-1)], [-1, 0,1], Ny-1, Ny-1); # the dim is the unkow in one row
    Tx_index = sparse.eye(Nx)  ##dim = # of rows
    Ix  = sparse.eye(Ny-1)
    Ix_index = sparse.spdiags([ np.ones(Nx), np.ones(Nx)], [-1, 1], Nx, Nx) ##dim = # of rows
    Lx= sparse.kron(Tx_index, Tx) + sparse.kron(Ix_index, Ix)
    Lx = Lx.toarray()
    for i in range(Ny-1):
        Lx[i,i] = -5
        Lx[-i-1, -i-1] = -5
    Ax = rho / dt *np.eye(Nx* (Ny-1)) - mu / 2. *Lx /(dx*dx)

    Ty_dia = -4*np.ones(Ny); Ty_dia[0] = -5; Ty_dia[-1] = -5
    Ty = sparse.spdiags([np.ones(Ny), Ty_dia, np.ones(Ny)], [-1, 0,1], Ny, Ny); # the dim is the unkow in one row
    Ty_index = sparse.eye(Nx-1) ##dim = # of rows
    Iy  = sparse.eye(Ny)
    Iy_index = sparse.spdiags([ np.ones(Nx-1), np.ones(Nx-1)], [-1, 1], Nx-1, Nx-1) ## dim = # of rows
    Ly= sparse.kron(Ty_index, Ty) + sparse.kron(Iy_index, Iy)
    Ay = rho / dt *np.eye((Nx-1)*(Ny)) - mu / 2. *Ly /(dx*dx)

    Tgx = np.zeros((Ny-1, Ny))
    for i in range(Ny-1):
        Tgx[i, i] =  -1;
        Tgx[i, i+1] =1
    Gx = sparse.kron(np.eye(Nx), Tgx)/dx

    Gy = np.zeros(((Nx-1)*Ny, Ny*Nx))
    for i in range((Nx-1)*Ny):
        Gy[i,i] = -1
        Gy[i, i+Ny] = 1
    Gy = Gy / dx

    Tdx = np.zeros((Ny,Ny-1))
    Tdx[0,0] = 1;
    a1 = sparse.spdiags([-np.ones(Ny-1), np.ones(Ny-1)], [0,1], Ny-1, Ny-1)
    Tdx[1:, :] = a1.toarray()
    Dx = sparse.kron(np.eye(Nx),Tdx) / dx

    Dy = np.zeros( (Ny*Nx, (Nx-1)*Ny) )
    for i in range(Ny):
        Dy[i,i] = 1
    for i in range( Ny, Ny*(Nx-1)):
        Dy[i, i-Ny] = -1
        Dy[i,i] =1
    for i in range(1, Ny+1):
        Dy[-i,-i] = -1
    Dy = Dy / dx

    A_rows = Nx*(Ny-1) + (Nx-1)*Ny + Nx*Ny
    A_columns = u_unknow + v_unknow+p_unknow
    A = np.zeros((A_rows, A_columns), dtype=np.float64)
    A[:(Ny-1)*Nx, : (Ny-1)*Nx] = Ax
    A[:(Ny-1)*Nx, -Nx*Ny:] = Gx.toarray()
    A[(Ny-1)*Nx : 2*(Ny-1)*Nx, (Ny-1)*Nx: 2*(Ny-1)*Nx] = Ay
    A[(Ny-1)*Nx : 2*(Ny-1)*Nx, -Nx*Ny:] = Gy
    A[-Nx*Ny:, :(Ny-1)*Nx] = -Dx.toarray()
    A[-Nx*Ny:, (Ny-1)*Nx:2*(Ny-1)*Nx ] = -Dy
##    A[:,-1] = 0
##    A[-1,-1] =-1
    return A

def RHS(uk, vk):
    b = np.zeros((b_size,1))# (Ny-1)*Nx + (Nx-1)*Ny + Nx*Ny rows
    bx = np.zeros((u_unknow,1)); by = np.zeros(( v_unknow,1)); bp = np.zeros((p_unknow,1))
    #uk = un; vk = vn # u is the last time; uk is this k step , u^(n+1/2, k) = (u^(n+1,k)+u)/2; un is what we want to get
    u_half = (u + uk) / 2. ; v_half = (v + vk) / 2.
    rhs_first = np.zeros((Nx, Ny-1)) # rhsfirst =  (rho/dt*I +mu/2*Lx)u
    adv_x = np.zeros((Nx, Ny-1)) # N^(n+1/2, k) = [u^(n+1/2, k) . \labada u^(n+1/2, k)]; 
    '''
    rhs_first[:,:] = rho / dt*u[1:-1, 1:-1] + ( mu/2 *(u[2:,1:-1] + u[:-2, 1:-1]+\
                                                        u[1:-1,2:]+u[1:-1,:-2] - 4*u[1:-1,1:-1]) )/dx/dx # u[1:-1] : without the first and the last element
    adv_x[:,:] = (u_half[1:-1,1:-1]*(u_half[1:-1, 2:] - u_half[1:-1, :-2])/dx/2 +(v_half[:-1, 2:-1] + v_half[:-1, 1:-2]+ v_half[1:,1:-2]+v_half[1:, 2:-1])/4 *(u_half[2:, 1:-1] - u_half[:-2, 1:-1]/2/dy) )
    '''
    for j in range(1, Ny):
        for i in range(1, Nx+1):
            # first term on the RHS of Eq. (31) from Boyce 2009 JCP
            rhs_first[i-1,j-1] = rho/dt*u[i,j] + mu/2 *(u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4*u[i,j])/dx/dx;
            adv_x[i-1,j-1] = u_half[i,j]*(u_half[i,j+1]-u_half[i,j-1])/2/dx \
                                 +(v_half[i-1,j]+v_half[i-1,j+1]+v_half[i,j]+v_half[i,j+1])/4*(u_half[i+1,j]-u_half[i-1,j])/2/dy;
    
    bx = (rhs_first - adv_x*rho).reshape(u_unknow,1)
    
    #bx = (rhs_first.T - (adv_x*rho).T).reshape(u_unknow,1)
    rhs_second = np.zeros((Nx-1, Ny));adv_y = np.zeros((Nx-1, Ny));
    '''
    rhs_second[:,:] = rho/dt*v[1:-1, 1:-1] + mu/2*(v[2:,1:-1] + v[:-2, 1:-1]+\
                                                   v[1:-1,2:]+v[1:-1,:-2] - 4*v[1:-1,1:-1])/dy/dy
    adv_y[:,:] = (v_half[1:-1,1:-1]*(v_half[2:, 1:-1] - v_half[:-2, 1:-1])/dy/2 +\
                       (u_half[2:-1,:-1] + u_half[1:-2, :-1]+ u_half[1:-2, 1:]+u_half[2:-1, 1:])/4 *(v_half[1:-1, 2:] - v_half[1:-1, :-2]/2/dx))
    '''
    for j in range(1, Ny+1):
        for i in range(1, Nx):
            rhs_second[i-1,j-1] = rho/dt*v[i,j] + mu/2*(v[i+1,j]+v[i-1,j]+v[i,j+1]+v[i,j-1]-4*v[i,j])/dx/dx;
            adv_y[i-1,j-1] = (u_half[i,j-1]+u_half[i,j]+u_half[i+1,j-1]+u_half[i+1,j])/4*(v_half[i,j+1]-v_half[i,j-1])/2/dx\
                             +v_half[i,j]*(v_half[i+1,j]-v_half[i-1,j])/2/dx;
    
    by = (rhs_second - adv_y*rho).reshape((v_unknow, 1))

    for i in range(1, Nx+1):
        bx[(i-1)*(Ny-1)] += mu/2*u[i,0]/dx/dx # left bc for u, for the Laplace term in mu/2 Lx, the first colum of u
        bx[i*(Ny-1)-1] += mu/2*u[i,-1]/dx/dx # right bc for u,for the Laplace term in mu/2 Lx, the first colum of u 
    for i in range(1,Ny):
        bx[-i] +=  mu/2*2/dx/dx; # top bc for u,for the Laplace term in mu/2 Lx
    
    # B.C. for v: Nx+1, Ny+2
    for i in range(Ny):
        by[i] += mu / 2*v[0,i] /dx/dx # the bottom
        by[-(Ny-i)]  += mu / 2*v[-1,i] / dx/dx

    for i in range (Nx-1):
        bp[Ny*i] -=  u[i+1,0] / dx # left for u because of -Dx
        bp[Ny*(i+1)-1] += u[i+1,-1] / dx # right for u
    for i in range(Ny):
        bp[i] -= v[0,i+1] / dy # bottom for v because of -Dy
        bp[-(Ny-i)] += v[-1, i+1] /dy
    #construct b
    b[:u_unknow] = bx[:]
    b[u_unknow:u_unknow+v_unknow] = by[:]
    b[-p_unknow: ] = bp[:]
    return b;

def get_uvp(un, vn, pn):
    u_temp = x[:Nx*(Ny-1)].reshape(Nx,Ny-1);
    v_temp = x[Nx*(Ny-1) : Nx*(Ny-1)+(Nx-1)*Ny].reshape(Nx-1, Ny)
    p_temp = x[ -Nx*Ny:].reshape(Nx, Ny)
    un[1:-1, 1:-1] = u_temp;  vn[1:-1, 1:-1] = v_temp;  pn[1:-1, 1:-1] = p_temp
##    un[1:-1, 1:-1] = x[1:Nx*(Ny-1)].reshape(Nx,Ny-1);
##    vn[1:-1, 1:-1] = x[Nx*(Ny-1) : Nx*(Ny-1)+(Nx-1)*Ny].reshape(Nx-1, Ny)
##    pn[1:-1,1:-1]= x[ -Nx*Ny:].reshape(Nx, Ny)
    ##    ghost node velocity for u in upper and bottom boundaries
    un[0,1:-1] = -un[1,1:-1]; un[-1,1:-1] = 2 - un[-2,1:-1];
    vn[1:-1,0] = -vn[1:-1, 1]; vn[1:-1, -1] = -vn[1:-1, -2]
    pn[0, 1:Ny+1] = pn[1, 1:Ny+1]; pn[-1, 1:Ny+1] = pn[-2, 1:Ny+1];
    pn[1:Ny+1, 0] = pn[1:Ny+1, 1]; pn[1:Ny+1, -1] = pn[ 1:Ny+1, -2]
    return un, vn,pn

Nx = 30; Ny = 30; Lenx =1 ; Leny =1 ; dx = Lenx/Nx; dy = Leny/Ny;
#assert(Lenx/Nx == Leny/Ny);
rho = 1;  Re =1000; # Re = rho * u* L / mu
U = 1.0 # initial velocity of uppest layer
mu = Lenx*U*rho/Re;  # kenematic visocity of the fluid
dt = 0.0125*2; tend = 10; t_step =int( tend /dt)
CFL = U*dt/dx

print(CFL)
#assert(CFL <1 )
u_unknow = Nx*(Ny-1); v_unknow = (Nx-1)*Ny; p_unknow = Nx*Ny
b_size = u_unknow + v_unknow+p_unknow
A = init_A(); Ainv = np.linalg.inv(A)
#np.seterr(all = 'ignore', over = 'ignore')
un = np.zeros((Nx+2, Ny+1)); u = un; uc = np.zeros((Nx+1, Ny+1)) 
vn = np.zeros((Nx+1, Ny+2)); v = vn; vc = np.zeros((Nx+1, Ny+1)) 
pn = np.zeros((Nx+2, Ny+2)); p = pn; pc = np.zeros((Nx+1, Ny+1))
# initialize for the top boundary
u[-2:, :] = U  # the top boundary
un[-2:,:] = U

for t in range(t_step):
    for k in range(1):
        b = RHS(un, vn)
        x = np.matmul(Ainv,b)
        un, vn, pn = get_uvp(un, vn, pn)   
    u = un; v = vn; p = pn;
    if t % 100 == 0:
        print('\t time is \t', t*dt)
    
for i in range(Nx+1):
    for j in range(Ny+1):
        uc[i,j] = 0.5*(un[i,j] + un[i+1,j]);
        vc[i,j] = 0.5*(vn[i,j] + vn[i,j+1]);
        pc[i,j] = 0.25*(pn[i,j]+pn[i,j+1]+pn[i+1,j]+pn[i+1,j+1]);
xgrid, ygrid = np.meshgrid(np.linspace(0, Lenx, Nx+1), np.linspace(0, Lenx, Nx+1))
##plt.figure(1)
##plt.streamplot(xgrid, ygrid, uc, vc, color=uc, linewidth = 1)
##plt.figure(2)
##plt.contour(xgrid,ygrid,uc,10);
fig, axs = plt.subplots(1, 3, figsize=(9, 6))
axs[0].streamplot(xgrid, ygrid, uc, vc, color=uc, linewidth = 1,cmap='autumn')
axs[1].contour(xgrid,ygrid,uc,10)
axs[2].plot(np.linspace(0, Lenx, Nx+1),vc[int(Nx/2+1),:])
end_t = datetime.datetime.now()
print('time \t', end_t - start_t)
plt.show()


