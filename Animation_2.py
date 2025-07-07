import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parámetros del péndulo
L1 = L2 = 1.0


def angulos_a_xy(θ1, θ2):
    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)
    return x1, y1, x2, y2

# === Elegir las trayectorias a animar ===
for idx1 in [1, 3, 5, 10, 15]:  # Cambia el rango según las trayectorias que tengas

    tray1 = np.loadtxt(f"trayectorias_txt/trayectoria_{idx1}.txt")
    tray2 = np.loadtxt(f"trayectorias_mod_txt/trayectoria2_{idx1}.txt")

    # Convertir a posiciones cartesianas
    def extraer_posiciones(tray):
        x1, y1, x2, y2 = [], [], [], []
        for θ1, _, θ2, _ in tray:
            xi1, yi1, xi2, yi2 = angulos_a_xy(θ1, θ2)
            x1.append(xi1); y1.append(yi1)
            x2.append(xi2); y2.append(yi2)
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = extraer_posiciones(tray1)
    x1_2, y1_2, x2_2, y2_2 = extraer_posiciones(tray2)

    # === Configurar animación ===
    fig, ax = plt.subplots()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title("Animación de dos trayectorias del péndulo doble")

    line1, = ax.plot([], [], 'o-', color='royalblue', lw=2, label=f'Trayectoria {idx1}')
    line2, = ax.plot([], [], 'o-', color='orange', lw=2, label=f'Trayectoria {idx1} Modificada')
    ax.legend()

    def update(i):
        line1.set_data([0, x1_1[i], x2_1[i]], [0, y1_1[i], y2_1[i]])
        line2.set_data([0, x1_2[i], x2_2[i]], [0, y1_2[i], y2_2[i]])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(x1_1), blit=True, interval=20)

    # === Crear carpeta 'animations' y guardar video ===
    guardar = True  # Cambia a False si no deseas guardar

    if guardar:
        os.makedirs("animations", exist_ok=True)
        nombre_salida = f"animations/animacion_{idx1}_mod.mp4"
        ani.save(nombre_salida, writer='ffmpeg', fps=50)
        print(f"Video guardado en {nombre_salida}")
    else:
        plt.show()
