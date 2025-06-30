import numpy as np
import numba as nb

@nb.njit  
def delta(theta1, theta2):
    return 2 - np.cos(theta1 - theta2)**2

@nb.njit
def dtheta1_dt(theta1, theta2, p1, p2):
    Δ = delta(theta1, theta2)
    return (p1 - p2 * np.cos(theta1 - theta2)) / Δ

@nb.njit
def dtheta2_dt(theta1, theta2, p1, p2):
    Δ = delta(theta1, theta2)
    return (2 * p2 - p1 * np.cos(theta1 - theta2)) / Δ

@nb.njit
def dp1_dt(theta1, theta2, p1, p2, g):
    Δ = delta(theta1, theta2)
    sin_diff = np.sin(theta1 - theta2)
    cos_diff = np.cos(theta1 - theta2)
    if p1 < -1e6:
        p1 = -1e6
    elif p1 > 1e6:
        p1 = 1e6

    if p2 < -1e6:
        p2 = -1e6
    elif p2 > 1e6:
        p2 = 1e6
    term1 = (p1**2 + 2 * p2**2 - p1 * p2 * cos_diff) * cos_diff 
    term2 = p1 * p2 
    return  (sin_diff / (Δ**2)) * term1 - (sin_diff / Δ) * term2 - 2 * g * np.sin(theta1)

@nb.njit
def dp2_dt(theta1, theta2, p1, p2, g):
    Δ = delta(theta1, theta2)
    sin_diff = np.sin(theta1 - theta2)
    cos_diff = np.cos(theta1 - theta2)
    if p1 < -1e6:
        p1 = -1e6
    elif p1 > 1e6:
        p1 = 1e6
    
    if p2 < -1e6:
        p2 = -1e6
    elif p2 > 1e6:
        p2 = 1e6
    term1 = (p1**2 + 2 * p2**2 - 2 * p1 * p2 * cos_diff) * cos_diff
    term2 = p1 * p2
    return  - (sin_diff / (Δ**2)) * term1 + (sin_diff / Δ) * term2 - g * np.sin(theta2)

@nb.njit
def K(state, g, dt):
    theta1, theta2, p1, p2 = state
    theta1 = theta1 % (2 * np.pi)
    theta2 = theta2 % (2 * np.pi)
    dtheta1 = dt*dtheta1_dt(theta1, theta2, p1, p2)
    dtheta2 = dt*dtheta2_dt(theta1, theta2, p1, p2)
    dp1 = dt*dp1_dt(theta1, theta2, p1, p2, g)
    dp2 = dt*dp2_dt(theta1, theta2, p1, p2, g)
    K = np.array([dtheta1, dtheta2, dp1, dp2])
    return K

@nb.njit
def RungeKutta(state, g, dt):
    k1 = K(state, g, dt)
    k2 = K(state + 0.5 * k1, g, dt)
    k3 = K(state + 0.5 * k2, g, dt)
    k4 = K(state + k3, g, dt)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

@nb.njit
def Energia(state, g):
    theta1= state[0]
    theta2= state[1]
    Δ = delta(theta1, theta2)
    cosdiff = np.cos(theta1 - theta2)
    p1= state[2]
    p2= state[3]
    T = ( p1**2 + 2 * p2**2 - 2 * p1 * p2 * cosdiff) / (2 * Δ) 
    V = - g * (2 * np.cos(theta1) + np.cos(theta2))
    E = T + V
    return E, T, V

@nb.njit
def PenduloDoble(theta1, theta2, p1, p2, g, dt, t):
    steps = int(t / dt)
    state = np.array([theta1, theta2, p1, p2])
    trajectory = np.zeros((steps, 4))
    energias = np.zeros((steps, 3)) # E, T, V
    states = np.zeros((steps, 4))  # theta1, theta2, p1, p2
    for i in range(steps):
        x1 = np.sin(state[0])
        y1 = -np.cos(state[0])
        x2 = x1 + np.sin(state[1])
        y2 = y1 - np.cos(state[1])
        trajectory[i] = [x1, y1, x2, y2]  
        energias[i] = Energia(state, g)  
        states[i] = state  
        state = RungeKutta(state, g, dt)
    return states, trajectory, energias

def guardar_trayectoria(trajectory, filename):
    header = "x1,y1,x2,y2"
    np.savetxt(filename, trajectory, delimiter=",", header=header, comments="", fmt="%.6f")

def guardar_energia(energias, filename):
    header = "energia,T,V"
    np.savetxt(filename, energias, delimiter=",", header=header, comments="", fmt="%.6f")

def guardar_estados(states, filename):
    header = "theta1,theta2,p1,p2"
    np.savetxt(filename, states, delimiter=",", header=header, comments="", fmt="%.6f")

if __name__ == "__main__":
    # Parámetros iniciales
    theta1_0 = np.pi/2          # Ángulo inicial del primer péndulo (rad)
    theta2_0 = np.pi/2         # Ángulo inicial del segundo péndulo (rad)
    Δ = delta(theta1_0, theta2_0)
    E = 5.0          # Energía total del sistema
    p2_0 = 0.0            # Momento inicial del segundo péndulo
    p1_0 = -np.sqrt(2 * E * Δ)          # Momento inicial del primer péndulo
    g = 9.81              # Gravedad (m/s^2)
    dt = 0.0001            # Paso de tiempo (s)
    t_total = 10        # Tiempo total de simulación (s)

    # Simular el movimiento del péndulo doble
    states, trajectory, energias = PenduloDoble(theta1_0, theta2_0, p1_0, p2_0, g, dt, t_total)

    # Guardar la trayectoria en un archivo
    guardar_trayectoria(trajectory, "trayectoria.txt")
    # Guardar la energía en un archivo
    guardar_energia(energias, "energia.txt")
    # Guardar los estados en un archivo
    guardar_estados(states, "estados.txt")