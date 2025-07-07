import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import os
import time

inicio = time.time()
# Parámetros físicos
E_vals = np.linspace(0.0, 35.0, 36)  # Ajusta el rango según tu sistema
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

@nb.jit (parallel=True)
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt * k1 / 2)
    k3 = f(t + dt/2, y + dt * k2 / 2)
    k4 = f(t + dt,   y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

@nb.jit (parallel=True)
def energia_total(θ1, ω1, θ2, ω2):
    y1 = -L1 * np.cos(θ1)
    y2 = y1 - L2 * np.cos(θ2)
    v1_sq = (L1 * ω1)**2
    v2_sq = (L1 * ω1)**2 + (L2 * ω2)**2 + 2 * L1 * L2 * ω1 * ω2 * np.cos(θ1 - θ2)
    PE = m1 * g * (y1+2) + m2 * g * (y2+1)
    KE = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    return PE + KE

@nb.jit (parallel=True)
def buscar_theta1_para_energia(E_objetivo, θ2=0.0, ω1=0.0, ω2=0.0):
    # Busca θ1 tal que energia_total(θ1, ω1, θ2, ω2) ≈ E_objetivo
    θ1_min, θ1_max = 0.01, np.pi-0.01
    for _ in range(100):
        θ1_mid = 0.5 * (θ1_min + θ1_max)
        E_mid = energia_total(θ1_mid, ω1, θ2, ω2)
        if np.abs(E_mid - E_objetivo) < 1e-4:
            return θ1_mid
        if E_mid < E_objetivo:
            θ1_min = θ1_mid
        else:
            θ1_max = θ1_mid
    return θ1_mid  # Devuelve el mejor encontrado

def simular_y_lyapunov_con_rk4(y0, T=50, dt=0.01, delta0=1e-8):
    steps = int(T / dt)
    yA = np.array(y0)
    yB = np.array(y0) + np.array([delta0, 0, 0, 0])
    log_divs = []
    trayectoria = []
    trayectoria2 = []
    for i in range(steps):
        t = i * dt
        trayectoria.append(yA.copy())
        trayectoria2.append(yB.copy())
        yA = rk4_step(deriv, t, yA, dt)
        yB = rk4_step(deriv, t, yB, dt)
        delta = yB - yA
        dist = np.linalg.norm(delta)
        log_divs.append(np.log(dist / delta0))
        delta = delta / dist * delta0
        yB = yA + delta
    lyap = np.mean(log_divs) / dt
    return lyap, np.array(trayectoria), np.array(trayectoria2)

# Bucle sobre distintos valores de energía total inicial
energias = []
lyapunovs = []
trayectorias = []
trayectorias2 = []

for E in E_vals:
    θ2 = 0.0
    ω1 = 0.0
    ω2 = 0.0
    θ1 = buscar_theta1_para_energia(E, θ2, ω1, ω2)
    y0 = [θ1, ω1, θ2, ω2]
    E_real = energia_total(θ1, ω1, θ2, ω2)
    λ, trayectoria, trayectoria2 = simular_y_lyapunov_con_rk4(y0, T=40, dt=0.01)
    energias.append(E_real)
    lyapunovs.append(λ)
    trayectorias.append(trayectoria)
    trayectorias2.append(trayectoria2)
    print(f"E={E_real:.2f} J (θ1={θ1:.2f} rad) -> λ_max={λ:.4f}")

final = time.time()

from scipy.optimize import curve_fit

# Definir la función exponencial para el ajuste
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Realizar el ajuste
popt, pcov = curve_fit(exp_func, energias, lyapunovs, maxfev=10000)

# Generar valores ajustados para graficar la curva
energias_fit = np.linspace(min(energias), max(energias), 200)
lyapunovs_fit = exp_func(energias_fit, *popt)

# Después de hacer el ajuste:
popt, pcov = curve_fit(exp_func, energias, lyapunovs, maxfev=10000)

# popt contiene [a, b, c]
a, b, c = popt
# Los errores estándar son la raíz cuadrada de la diagonal de la matriz de covarianza
a_err, b_err, c_err = np.sqrt(np.diag(pcov))

print(f"Parámetros del ajuste exponencial:")
print(f"a = {a:.4e} ± {a_err:.4e}")
print(f"b = {b:.4e} ± {b_err:.4e}")
print(f"c = {c:.4e} ± {c_err:.4e}")

# Graficar datos y ajuste
plt.plot(energias, lyapunovs, marker='o', label="Datos")
plt.plot(energias_fit, lyapunovs_fit, 'r--', label="Ajuste exponencial")
plt.xlabel("Energía total inicial [J]")
plt.ylabel("Exponente de Lyapunov máximo [s⁻¹]")
plt.title("RK4: Caos vs Energía en el Péndulo Doble\n Tiempo de ejecución: {:.2f} s".format(final - inicio))
plt.grid(True)
plt.legend()



guardar = True  # Cambia a False si no deseas guardar la figura

if guardar:
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/lyapunov_vs_energia.png")

plt.show()

# Directorio de salida
os.makedirs("trayectorias_txt", exist_ok=True)
for i, trayectoria in enumerate(trayectorias):
    np.savetxt(f"trayectorias_txt/trayectoria_{i}.txt", trayectoria)
    print(f"Guardada trayectoria_{i}.txt")

# Directorio de salida
os.makedirs("trayectorias_mod_txt", exist_ok=True)
for i, trayectoria2 in enumerate(trayectorias2):
    np.savetxt(f"trayectorias_mod_txt/trayectoria2_{i}.txt", trayectoria2)
    print(f"Guardada trayectoria2_{i}.txt")