import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import os
import time

inicio = time.time()
# Parámetros físicos
g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0

@nb.jit (parallel=True)
def angulos_a_xy(θ1, θ2):
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    return x1, y1, x2, y2


# Ecuaciones del movimiento
@nb.jit (parallel=True)
def deriv(t, y):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1

    den1 = (m1 + m2)*L1 - m2*L1*np.cos(Δ)**2
    den2 = (L2/L1)*den1

    dω1 = (m2*L1*ω1**2*np.sin(Δ)*np.cos(Δ) +
           m2*g*np.sin(θ2)*np.cos(Δ) +
           m2*L2*ω2**2*np.sin(Δ) -
           (m1 + m2)*g*np.sin(θ1)) / den1

    dω2 = (-m2*L2*ω2**2*np.sin(Δ)*np.cos(Δ) +
           (m1 + m2)*g*np.sin(θ1)*np.cos(Δ) -
           (m1 + m2)*L1*ω1**2*np.sin(Δ) -
           (m1 + m2)*g*np.sin(θ2)) / den2

    return np.array([ω1, dω1, ω2, dω2])

# Método de Runge-Kutta de 4º orden
@nb.jit (parallel=True)
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1 / 2)
    k3 = f(t + dt/2, y + dt * k2 / 2)
    k4 = f(t + dt,   y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# Energía total
@nb.jit (parallel=True)
def energia_total(θ1, ω1, θ2, ω2):
    y1 = -L1 * np.cos(θ1)
    y2 = y1 - L2 * np.cos(θ2)
    v1_sq = (L1 * ω1)**2
    v2_sq = (L1 * ω1)**2 + (L2 * ω2)**2 + 2 * L1 * L2 * ω1 * ω2 * np.cos(θ1 - θ2)
    PE = m1 * g * (y1+2) + m2 * g * (y2+1)
    KE = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    return PE + KE


def simular_y_lyapunov_con_rk4(y0, T=50, dt=0.01, delta0=1e-8):
    steps = int(T / dt)
    yA = np.array(y0)
    yB = np.array(y0) + np.array([delta0, 0, 0, 0])
    log_divs = []
    trayectoria = []

    for i in range(steps):
        t = i * dt
        trayectoria.append(yA.copy())  # Guardar estado para animación
        yA = rk4_step(deriv, t, yA, dt)
        yB = rk4_step(deriv, t, yB, dt)
        delta = yB - yA
        dist = np.linalg.norm(delta)
        log_divs.append(np.log(dist / delta0))
        delta = delta / dist * delta0
        yB = yA + delta

    lyap = np.mean(log_divs) / dt
    return lyap, np.array(trayectoria)  # Devuelve también la trayectoria completa


# Bucle sobre distintos ángulos iniciales
theta_vals = np.linspace(0.01, np.pi, 10)
energias = []
lyapunovs = []
trayectorias = []  # Aquí se guardan todas las trayectorias

for θ1 in theta_vals:
    θ2 = 0.0
    ω1 = 0.0
    ω2 = 0.0
    y0 = [θ1, ω1, θ2, ω2]
    E = energia_total(θ1, ω1, θ2, ω2)
    λ, trayectoria = simular_y_lyapunov_con_rk4(y0, T=40, dt=0.01)
    energias.append(E)
    lyapunovs.append(λ)
    trayectorias.append(trayectoria)
    print(f"θ1={θ1:.2f} rad -> E={E:.2f} J, λ_max={λ:.4f}")

final = time.time()

# Graficar resultado
plt.plot(energias, lyapunovs, marker='o')
plt.xlabel("Energía total inicial [J]")
plt.ylabel("Exponente de Lyapunov máximo [s⁻¹]")
plt.title("RK4: Caos vs Energía en el Péndulo Doble\n Tiempo de ejecución: {:.2f} s".format(final - inicio))
plt.grid(True)

guardar = True  # Cambia a False si no deseas guardar la figura

if guardar:
    # Crear carpeta 'plots' si no existe
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/lyapunov_vs_energia.png")

plt.show()



# Directorio de salida
os.makedirs("trayectorias_txt", exist_ok=True)

# Guardar cada trayectoria como archivo de texto
for i, trayectoria in enumerate(trayectorias):
    np.savetxt(f"trayectorias_txt/trayectoria_{i}.txt", trayectoria)
    print(f"Guardada trayectoria_{i}.txt")

