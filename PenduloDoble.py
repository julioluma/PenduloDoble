import numpy as np
import numba as nb

@nb.njit  
def delta(theta1, theta2):
    # Calculate the delta value, useful for the equations of motion
    return 2 - np.cos(theta1 - theta2)**2

@nb.njit
def dtheta1_dt(theta1, theta2, p1, p2):
    # Calculate the derivative of theta1 with respect to time
    Δ = delta(theta1, theta2)
    return (p1 - p2 * np.cos(theta1 - theta2)) / Δ

@nb.njit
def dtheta2_dt(theta1, theta2, p1, p2):
    # Calculate the derivative of theta2 with respect to time
    Δ = delta(theta1, theta2)
    return (2 * p2 - p1 * np.cos(theta1 - theta2)) / Δ

@nb.njit
def dp1_dt(theta1, theta2, p1, p2, g):
    # Calcular la derivada de p1 con respecto al tiempo
    Δ = delta(theta1, theta2)
    sin_diff = np.sin(theta1 - theta2)
    cos_diff = np.cos(theta1 - theta2)

    # Limitar los valores de p1 y p2 para evitar desbordamientos
    p1 = np.clip(p1, -1e6, 1e6)
    p2 = np.clip(p2, -1e6, 1e6)

    term1 = (p1**2 + 2 * p2**2 - p1 * p2 * cos_diff) * cos_diff 
    term2 = p1 * p2 
    return  (sin_diff / (Δ**2)) * term1 - (sin_diff / Δ) * term2 - 2 * g * np.sin(theta1)

@nb.njit
def dp2_dt(theta1, theta2, p1, p2, g):
    # Calcular la derivada de p2 con respecto al tiempo
    Δ = delta(theta1, theta2)
    sin_diff = np.sin(theta1 - theta2)
    cos_diff = np.cos(theta1 - theta2)

    # Limitar los valores de p1 y p2 para evitar desbordamientos
    p1 = np.clip(p1, -1e6, 1e6)
    p2 = np.clip(p2, -1e6, 1e6)

    term1 = (p1**2 + 2 * p2**2 - 2 * p1 * p2 * cos_diff) * cos_diff
    term2 = p1 * p2
    return  - (sin_diff / (Δ**2)) * term1 + (sin_diff / Δ) * term2 - g * np.sin(theta2)

@nb.njit
def K(state, g, dt):
    # Calculate the derivatives of the state variables
    # state = [theta1, theta2, p1, p2]
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
    """
    Simulates the motion of a double pendulum using the Runge-Kutta method.
    
    Parameters:
    theta1 (float): Initial angle of the first pendulum (in radians).
    theta2 (float): Initial angle of the second pendulum (in radians).
    p1 (float): Initial momentum of the first pendulum.
    p2 (float): Initial momentum of the second pendulum.
    g (float): Acceleration due to gravity.
    dt (float): Time step for the simulation.
    t (float): Total time for the simulation.
    
    Returns:
    np.ndarray: Array containing the angles and momenta at each time step.
    """
    steps = int(t / dt)
    state = np.array([theta1, theta2, p1, p2])
    trajectory = np.zeros((steps, 4))
    energias = np.zeros((steps, 3)) # Array to store energy, T, and V
    states = np.zeros((steps, 4))  # Array to store states for debugging
    for i in range(steps):
        # Actualizar el porcentaje completado
        porcentaje = (i + 1) / steps * 100
        print(f"Progreso: {porcentaje:.2f}% ", end="\r")  # Sobrescribe la línea anterior
        
        
        x1 = np.sin(state[0])
        y1 = -np.cos(state[0])
        x2 = x1 + np.sin(state[1])
        y2 = y1 - np.cos(state[1])

        trajectory[i] = [x1, y1, x2, y2]  
        energias[i] = Energia(state, g)  
        states[i] = state  
        # Actualizar el estado del péndulo
        state = RungeKutta(state, g, dt)
    return state, trajectory, energias



def guardar_trayectoria(trajectory, filename):
    """
    Guarda los ángulos theta1 y theta2 de la trayectoria en un archivo de texto.

    Parameters:
    trajectory (np.ndarray): Array con la trayectoria del péndulo doble.
    filename (str): Nombre del archivo donde se guardará la trayectoria.
    """
    # Extraer theta1 y theta2
    theta1_theta2 = trajectory[:, :2]

    # Guardar en el archivo con el formato especificado
    header = "theta1,theta2"
    np.savetxt(filename, theta1_theta2, delimiter=",", header=header, comments="", fmt="%.6f")

def guardar_energia(energias, filename):
    """
    Guarda la energía del péndulo doble en un archivo de texto.

    Parameters:
    trajectory (np.ndarray): Array con la trayectoria del péndulo doble.
    filename (str): Nombre del archivo donde se guardará la energía.
    """

    # Guardar en el archivo con el formato especificado
    header = "energia, T, V"
    np.savetxt(filename, energias, delimiter=",", header=header, comments="", fmt="%.6f")

def guardar_estados(states, filename):
    """
    Guarda los estados del péndulo doble en un archivo de texto.

    Parameters:
    states (np.ndarray): Array con los estados del péndulo doble.
    filename (str): Nombre del archivo donde se guardarán los estados.
    """
    header = "theta1,theta2,p1,p2"
    np.savetxt(filename, states, delimiter=",", header=header, comments="", fmt="%.6f")

if __name__ == "__main__":
    # Parámetros iniciales
    theta1_0 = np.pi/2          # Ángulo inicial del primer péndulo (rad)
    theta2_0 = np.pi/2         # Ángulo inicial del segundo péndulo (rad)
    Δ = delta(theta1_0, theta2_0)  # Delta inicial
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