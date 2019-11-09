import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from matplotlib import animation


'''
    Notes for myself:
    matrix product: np.matmul(aa, bb)
    matrix elemetry product is: aa*bb
    matlab: fft(a,[], 1) apply fft on columns
                fft(a, [],2) apply fft on rows
    scipy: fft(a): apply fft on rows
    matlab: fft(a,[],1) = scipy fft(a, axis=0)
'''
N = 64; L = 1.0; h = L / N
Nb = int(np.ceil(np.pi*(L/2)/(h/2))) 
K = 1; rho = 1; mu = 0.01; tmax = 1; dt = 0.01; time_steps = int( np.ceil( tmax / dt) )
dtheta = np.pi * 2 / Nb; c=dtheta/(h*h);
ygrid, xgrid = np.meshgrid( np.linspace(0, L, N), np.linspace(0, L, N))

ibase = np.arange(N,dtype = 'int') # index base for ip and im, from 0 to 63
ip = np.zeros(N,dtype = 'int'); ip[:-1] = ibase[1:]
im = np.zeros(N, dtype = 'int'); im[1:] = ibase[:-1]; im[0] = N-1

kbase = np.arange(Nb,dtype = 'int') # index base for ip and im, from 0 to 63
kp = np.zeros(Nb,dtype = 'int'); kp[:-1] = kbase[1:]
km = np.zeros(Nb, dtype = 'int'); km[1:] = kbase[:-1]; km[0] = Nb-1

def initial (N, h, L) :
    theta = np.linspace(0, Nb-1, Nb)*dtheta; #[64,1] k is from 0 to 201, indicating the order of a vector
    X_coordinate = np.zeros((Nb,2));  
    X_coordinate[:,0] = np.cos(theta); X_coordinate[:,1] = np.sin(theta) # x, y coordinate of initial Boundary
    X = L/2 + (L/4)* X_coordinate
    u = np.zeros((N,N,2))
    for i in range(N):
        u[i,: ,1] = np.sin(2*np.pi*i*h / L)
    vorticity = np.zeros((N,N))# dv/dx - du/dy
    vorticity = (u[ip, :, 1] - u[im, :, 1] - u[:,ip,0] + u[:,im,0])/(2*h);
    dvorticity= (np.max(vorticity)-np.min(vorticity) )/5;
    values= np.linspace(-10*dvorticity, 10*dvorticity, 21)
    return [u, vorticity, X, values]

def init_a():
    a = np.zeros((N,N,2,2))
    a[:,:, 0,0] = 1
    a[:,:, 1,1] = 1
    for m1 in range(N):
        for m2 in range(N):
            if not (((m1==0)|(m1==N/2))&((m2==0)|(m2==N/2))):
                t = (2*np.pi / N) *np.matrix([m1,m2]) # 1*2 matrix
                s = np.sin(t)
                ss = (s.T* s) / (s* s.T)
                a[m1, m2, :, :] = a[m1, m2, :, :] - ss
    for m1 in range(N):
        for m2 in range(N):
            t = np.pi / N * np.matrix([m1, m2])
            s = np.sin(t);
            a[m1, m2, :, :] = a[m1, m2, :, :] / ( 1+ dt /2 * (mu / rho)* ( 4/ (h*h)) *( s*s.T) )
    return a

def vec_phi1(r):
    w = np.zeros((4, r.size,4))
    q = np.sqrt( 1+ 4 *r -4* r*r)
    for i in range(r.size):
        w[3, i, :] = np.tile( (1 + 2*r[i] - q[i])/8, (1,1,4))
        w[2, i, :] = np.tile( (1 + 2*r[i] + q[i])/8, (1,1,4))
        w[1, i, :] = np.tile( (3 - 2*r[i] + q[i])/8, (1,1,4))
        w[0, i, :] = np.tile( (3 - 2*r[i] - q[i])/8, (1,1,4))
    return w

def vec_phi2(r):
    w = np.zeros((4, r.size,4))
    q = np.sqrt( 1+ 4 *r - 4*r*r)
    for i in range(r.size):
        w[:, i, 3] = np.tile( (1 + 2*r[i] - q[i])/8, (4))
        w[:, i, 2] = np.tile( (1 + 2*r[i] + q[i])/8, (4))
        w[:, i, 1] = np.tile( (3 - 2*r[i] + q[i])/8, (4))
        w[:, i, 0] = np.tile( (3 - 2*r[i] - q[i])/8, (4))
    return w

### from fluid to Solid: Velocity ###  
def  vec_interp(u, X):
    s = X / h ; i = np.floor(s); r = s -i
    U = np.zeros((Nb, 2)) # 2D vector, Ux, Uy
    w = ( vec_phi1(r[:,0])*vec_phi2(r[:,1]) ).transpose(0,2,1) # return [5, 5,Nb] matrix   
    for k in range(Nb):
        i1 = np.array( [ i[k,0]-1, i[k,0], i[k,0]+1, i[k,0]+2], dtype = 'int' ) % N
        i2 = np.array( [i[k,1]-1, i[k,1], i[k,1]+1, i[k,1]+2], dtype = 'int'  )% N
        ww = w[:,:,k]
        U[k, 0] = sum(sum(ww*u[i1.reshape((4,1)),i2,0]))
        U[k, 1] = sum(sum(ww*u[i1.reshape((4,1)),i2,1]))      
    return U

def Force(X):
    F = np.zeros((Nb,2))
    F[:,:] = K*( X[kp, :] + X[km, :] - 2*X[:,:]) /(dtheta**2)
    return F
    
def vec_spread(F, X):
    s = X / h ; i = np.floor(s); r = s -i
    w = ( vec_phi1(r[:,0])*vec_phi2(r[:,1]) ).transpose(0,2,1) # return [5, 5,Nb] matrix
    f = np.zeros((N,N,2))    
    for k in range(Nb):
        i1 = np.array( [ i[k,0]-1, i[k,0], i[k,0]+1, i[k,0]+2], dtype = 'int' ) % N
        i2 = np.array( [i[k,1]-1, i[k,1], i[k,1]+1, i[k,1]+2], dtype = 'int'  )% N
        ww = w[:,:,k]
        f[i1.reshape((4,1)),i2,0] =  f[i1.reshape((4,1)),i2,0] + (c*F[k,0])*ww;
        f[i1.reshape((4,1)),i2,1] =  f[i1.reshape((4,1)),i2,1] + ( c*F[k,1])*ww 
    return f

def sk(uk, g):
    f = ( (uk[ip, :, 0] + uk[:,:, 0])*g[ip,:] - (uk[im,:, 0]+uk[:,:,0])*g[im,:]\
          +(uk[:, ip, 1] + uk[:,:, 1])*g[:, ip] - (uk[:, im,1]+uk[:,:,1])*g[:, im] ) / 4*h
    return f

def skew(u1):
    w = np.zeros((N,N,2))
    w[:,:,0] = sk(u1, u1[:,:,0])
    w[:,:,1] = sk(u1, u1[:,:,1])
    return w

def laplacian(u):
    w = (u[ip, :,:] + u[im,:,:] + u[:, ip, :] + u[:, im, :] - 4*u ) / h**2
    return w

def fluid(u, ff):
    w = u - (dt/2)*skew(u)+(dt/(2*rho))*ff
    w= fft(w,axis = 0); w= fft(w, axis = 1)
    uu = np.zeros((N,N,2), dtype = 'complex'); # dtype needs to be complex, otherwise, error
    uu[:,:,0]=a[:,:,0,0]*w[:,:,0]+a[:,:,0,1]*w[:,:,1] # Solve for LHS
    uu[:,:,1]=a[:,:,1,0]*w[:,:,0]+a[:,:,1,1]*w[:,:,1]
    uu = ifft(uu, axis = 1); uu = ifft(uu, axis = 0).real
    
    uuu = np.zeros((N,N,2), dtype = 'complex')
    w = u-dt*skew(uu)+(dt/rho)*ff+(dt/2)*(mu/rho)*laplacian(u)
    w= fft(w, axis = 0); w= fft(w, axis = 1)
    uuu[:,:,0]=a[:,:,0,0]*w[:,:,0]+a[:,:,0,1]*w[:,:,1] # Solve for LHS
    uuu[:,:,1]=a[:,:,1,0]*w[:,:,0]+a[:,:,1,1]*w[:,:,1]
    uuu = ifft(uuu, axis = 1); uuu = ifft(uuu, axis = 0).real
    return [uuu, uu]

[u, vorticity, X,values] = initial(N, h, L); a = init_a()
images = []
fig = plt.figure(figsize=(6,6))
for t in range(30):
    images.append( plt.plot(X[:,0],X[:,1],'ro', markersize = 2), )
    plt.xlim((0,L))
    plt.ylim((0,L)) 
    XX = X+ dt/2* vec_interp(u,X)
    ff = vec_spread(Force(XX), XX) # checked
    [u, uu] = fluid(u, ff)
    X = X + dt * vec_interp(uu, XX)
    vorticity = (u[ip, :, 1] - u[im, :, 1] - u[:,ip,0] + u[:,im,0])/(2*h)
    print('Time steps:', t)
im_ani = animation.ArtistAnimation(fig, images, interval=500,repeat = False, blit=True)
plt.show()


