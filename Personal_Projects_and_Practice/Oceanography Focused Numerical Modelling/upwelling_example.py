import numpy as np
import matplotlib.pyplot as plt

# Constants
Lx, Lz = 100e3, 500  # Domain length (m), depth (m)
Nx, Nz = 100, 50  # Number of grid points
dx, dz = Lx / Nx, Lz / Nz  # Grid spacing
dt = 100  # Time step (s)
total_time = 10 * 24 * 3600  # Total simulation time (10 days)

# Physical parameters
Kz = 1e-4  # Vertical diffusivity (m^2/s)
upwelling_velocity = 1e-5  # Upwelling velocity (m/s)

# Initial conditions: Linear temperature gradient
z = np.linspace(0, -Lz, Nz)
x = np.linspace(0, Lx, Nx)
T = np.tile(25 + z / 10, (Nx, 1)).T  # Warm surface, cooler depths

# Simulation loop
timesteps = int(total_time / dt)
for t in range(timesteps):
    # Horizontal advection: Shift temperature to simulate wind forcing
    T[:, 0] += upwelling_velocity * dt / dz * T[:, 1]
    
    # Vertical diffusion
    T[1:-1, :] += Kz * dt / dz**2 * (T[2:, :] - 2 * T[1:-1, :] + T[:-2, :])
    
    # Boundary conditions: No flux at bottom and top
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]
    
    # Visualization every few days
    if t % (24 * 3600 // dt) == 0:
        plt.contourf(x / 1e3, z, T, levels=20, cmap='coolwarm')
        plt.colorbar(label='Temperature (°C)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (m)')
        plt.title(f"Time: {t * dt / 86400:.1f} days")
        plt.pause(0.1)

plt.show()