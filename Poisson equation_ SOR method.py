import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

start =time.time()
N = 200                    # Number of grid points
x = np.linspace(0, 1, N)   # Length of the domain in x
y = np.linspace(0,1,N)     # Length of the domain in y

u = np.zeros((N, N))       # Initialize the solution array


# SOR method
@njit
def sor(u, rho, omega=1.5, tol=1e-6, max_iter=10000):
    N = u.shape[0]
    h = 1/(N - 1)
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, N-1):
            for j in range(1, N-1):
                old_u = u[i, j]
                new_u = (1 - omega) * old_u + (omega / 4) * (
                    u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h**2 * rho[i, j])
                u[i, j] = new_u
                max_diff = max(max_diff, abs(new_u - old_u))

        if max_diff < tol:
            print(f"Converged in {it} iterations.")
            break
    else:
        print(" Did not converged.")
    
    return u

X, Y = np.meshgrid(x, y)
rho = 10 * ((X - 0.5)**2 + (Y - 0.5)**2)

u[0, :] = 0

u[-1, :] = np.sin(4 * np.pi * x )**2

u[:, 0] = 0

u[:, -1] =np.sin(4 * np.pi * y )**2

sol = sor(u, rho)
end  = time.time()


plt.imshow(sol, origin='lower', cmap='viridis')

plt.colorbar(label=r'\u (x,y)')
plt.title('Solution of the Poisson equation using SOR method')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.show()
print(f" The execution time was: {end - start:.3f} seconds")

