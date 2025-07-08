import numpy as np
import matplotlib.pyplot as plt
import os

# Crear la carpeta "Graficas" si no existe
if not os.path.exists("plots"):
    os.makedirs("plots")

# ================================================================================
# Parámetros
Guardar = True  # Si es True, guarda la gráfica en un archivo
# ================================================================================

file1name = f"Tiempos/JOEL1_T=50_dt=001.txt"
file2name = f"Tiempos/JOEL2_T=50_dt=001.txt"
file3name = f"Tiempos/JOEL3_T=50_dt=001.txt"

if not os.path.exists(file1name):
    raise FileNotFoundError(f"El archivo {file1name} no existe. Asegúrate de que el archivo esté en la carpeta Tiempos.")

if not os.path.exists(file2name):
    raise FileNotFoundError(f"El archivo {file2name} no existe. Asegúrate de que el archivo esté en la carpeta Tiempos.")

if not os.path.exists(file3name):
    raise FileNotFoundError(f"El archivo {file3name} no existe. Asegúrate de que el archivo esté en la carpeta Tiempos.")

# Cargar los datos de los archivos
data1 = np.loadtxt(file1name, delimiter=",")
data2 = np.loadtxt(file2name, delimiter=",")
data3 = np.loadtxt(file3name, delimiter=",")
# Extraer los datos
threads1 = data1[:, 0]  # Primera columna (Número de hilos)
tiempos1 = data1[:, 1]  # Segunda columna (Tiempos en segundos)
tiempos2 = data2[:, 1]  # Segunda columna (Tiempos en segundos)
tiempos3 = data3[:, 1]  # Segunda columna (Tiempos en segundos)
# Media de tiempos
tiempos = np.zeros(len(tiempos1))  # Inicializar un array para los tiempos medios
error = np.zeros(len(tiempos1))  # Inicializar un array para los errores
for i in range(len(tiempos1)):
    tiempos[i] = (tiempos1[i] + tiempos2[i] + tiempos3[i]) / 3
    error[i] = np.std([tiempos1[i], tiempos2[i], tiempos3[i]]) / np.sqrt(3)  # Error estándar


# Graficar
plt.figure(figsize=(8, 5))
plt.errorbar(threads1, tiempos, yerr=error, fmt='o-', capsize=4, label='Mi JOEL')
plt.xlabel("Número de threads")
plt.ylabel("Tiempo (segundos)")
plt.title("Tiempo de ejecución vs Número de threads")
plt.grid()
plt.legend()
if Guardar:
    # Guardar el gráfico
    output_filename = f"plots/Tiempo_vs_Threads.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.show()