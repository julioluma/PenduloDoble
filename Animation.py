import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Longitudes de los péndulos
L1 = 1.0  # Longitud del primer péndulo (m)
L2 = 1.0  # Longitud del segundo péndulo (m)

# Función para leer el archivo de texto
def leer_trayectoria(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data

def leer_energia(filename):
    """
    Lee las energías desde un archivo de texto.

    Parameters:
    filename (str): Ruta del archivo de texto.

    Returns:
    tuple: Arrays con la energía total, cinética y potencial.
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Saltar la primera fila (encabezado)
    energia_total = data[:, 0]
    energia_cinetica = data[:, 1]
    energia_potencial = data[:, 2]
    return energia_total, energia_cinetica, energia_potencial


# Leer la trayectoria desde el archivo
filename_trayectoria = "trayectoria.txt"
trayectoria = leer_trayectoria(filename_trayectoria)
x1 = trayectoria[:, 0]
y1 = trayectoria[:, 1]
x2 = trayectoria[:, 2]
y2 = trayectoria[:, 3]

# Leer las energías desde el archivo
filename_energia = "energia.txt"
energia_total, energia_cinetica, energia_potencial = leer_energia(filename_energia)


# Crear la figura con dos subplots
fig, (ax_anim, ax_energy) = plt.subplots(1, 2, figsize=(12, 6))

# Configurar el subplot de la animación
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.set_aspect('equal')
ax_anim.grid()
line, = ax_anim.plot([], [], 'o-', lw=2, label="Péndulo")
trail, = ax_anim.plot([], [], 'r.', markersize=2, label="Trayectoria")
ax_anim.legend()

# Configurar el subplot de las energías
ax_energy.set_xlim(0, len(energia_total))
ax_energy.set_ylim(
    min(np.min(energia_total), np.min(energia_cinetica), np.min(energia_potencial)) * 1.1,
    max(np.max(energia_total), np.max(energia_cinetica), np.max(energia_potencial)) * 1.1
)
ax_energy.set_title("Energías del sistema")
ax_energy.set_xlabel("Tiempo (frames)")
ax_energy.set_ylabel("Energía")
energy_total_line, = ax_energy.plot([], [], 'b-', label="Energía total")
energy_kinetic_line, = ax_energy.plot([], [], 'g-', label="Energía cinética")
energy_potential_line, = ax_energy.plot([], [], 'r-', label="Energía potencial")
ax_energy.legend()

# Inicializar la animación
def init():
    line.set_data([], [])
    trail.set_data([], [])
    energy_total_line.set_data([], [])
    energy_kinetic_line.set_data([], [])
    energy_potential_line.set_data([], [])
    return line, trail, energy_total_line, energy_kinetic_line, energy_potential_line

# Actualizar la animación
def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    trail.set_data(x2[:frame], y2[:frame])
    energy_total_line.set_data(range(frame), energia_total[:frame])
    energy_kinetic_line.set_data(range(frame), energia_cinetica[:frame])
    energy_potential_line.set_data(range(frame), energia_potencial[:frame])
    return line, trail, energy_total_line, energy_kinetic_line, energy_potential_line

# Crear la animación con opción de velocidad
speed_factor = 100  # Factor de velocidad (1 = normal, >1 = más rápido, <1 = más lento)
ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True, interval=20 / speed_factor)

# Guardar la animación en un archivo MP4
guardar = True  # Cambia a True si deseas guardar la animación
if guardar:
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("pendulo_doble.mp4", writer=writer)
    print("Animación guardada como 'pendulo_doble.mp4'.")

# Mostrar la animación
plt.tight_layout()
plt.show()