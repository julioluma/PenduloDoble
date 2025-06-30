import numpy as np
import matplotlib.pyplot as plt

# --- Ajusta esta función si tus momentos NO son las derivadas ---
def calcular_omegas(theta1, theta2, p1, p2):
    # Copia las funciones dtheta1_dt y dtheta2_dt aquí si no las tienes importadas
    def dtheta1_dt(theta1, theta2, p1, p2):
        Δ = 2 - np.cos(theta1 - theta2)**2
        return (p1 - p2 * np.cos(theta1 - theta2)) / Δ

    def dtheta2_dt(theta1, theta2, p1, p2):
        Δ = 2 - np.cos(theta1 - theta2)**2
        return (2 * p2 - p1 * np.cos(theta1 - theta2)) / Δ

    omega1 = dtheta1_dt(theta1, theta2, p1, p2)
    omega2 = dtheta2_dt(theta1, theta2, p1, p2)
    return omega1, omega2

# Leer los datos
estados = np.loadtxt("estados.txt", delimiter=",", skiprows=1)
theta1 = estados[:, 0]
theta2 = estados[:, 1]
p1 = estados[:, 2]
p2 = estados[:, 3]

# Calcular las derivadas angulares
omega1, omega2 = calcular_omegas(theta1, theta2, p1, p2)

# Opcional: tomar una sección de Poincaré (por ejemplo, cada N pasos)
N = 1000
theta1_p = theta1[::N]
theta2_p = theta2[::N]
omega1_p = omega1[::N]
omega2_p = omega2[::N]

# Crear los mapas de Poincaré
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. theta1 vs theta2
axs[0].plot(theta1_p, theta2_p, 'k-', linewidth=1)
axs[0].set_xlabel(r'$\theta_1$ (rad)')
axs[0].set_ylabel(r'$\theta_2$ (rad)')
axs[0].set_title(r'Mapa de Poincaré: $\theta_1$ vs $\theta_2$')
axs[0].grid()

# 2. theta1 vs omega1
axs[1].plot(theta1_p, omega1_p, 'b-', linewidth=1)
axs[1].set_xlabel(r'$\theta_1$ (rad)')
axs[1].set_ylabel(r'$\omega_1$ (rad/s)')
axs[1].set_title(r'Mapa de Poincaré: $\theta_1$ vs $\omega_1$')
axs[1].grid()

# 3. theta2 vs omega2
axs[2].plot(theta2_p, omega2_p, 'r-', linewidth=1)
axs[2].set_xlabel(r'$\theta_2$ (rad)')
axs[2].set_ylabel(r'$\omega_2$ (rad/s)')
axs[2].set_title(r'Mapa de Poincaré: $\theta_2$ vs $\omega_2$')
axs[2].grid()

plt.tight_layout()
plt.show()