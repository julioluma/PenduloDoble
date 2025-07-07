import numpy as np
import matplotlib.pyplot as plt
import os

# Parametros 
guardar = True  # Cambia a False si no deseas guardar la figura 
E = 10
for tipo_mapa in range(1, 4):

    # Cambia el nombre del archivo si quieres analizar otra trayectoria
    filename = "trayectorias_txt/trayectoria_" + str(E) + ".txt"

    # Cargar la trayectoria
    data = np.loadtxt(filename)

    if tipo_mapa == 1:
        i = 0
        j = 2
        n1 = "θ1 (rad)"
        n2 = "θ2 (rad)"
    elif tipo_mapa == 2:
        i = 0
        j = 1
        n1 = "θ1 (rad)"
        n2 = "ω1 (rad/s)"
    elif tipo_mapa == 3:
        i = 2
        j = 3
        n1 = "θ2 (rad)"
        n2 = "ω2 (rad/s)"


    omega1 = data[:, i]
    theta2 = data[:, j]

    plt.figure(figsize=(6, 6))
    plt.scatter(omega1, theta2, s=1, color='navy')
    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title("Mapa de Poincaré: " + n1 +  " vs " + n2)
    plt.grid(True)
    plt.tight_layout()

    # Crear carpeta para guardar la figura
    os.makedirs("plots/"+ str(E), exist_ok=True)
    if guardar:
        plt.savefig(f"plots/{E}/poincare_{tipo_mapa}_{E}.png")

    plt.show()