import time
import random
import matplotlib.pyplot as plt
import numpy as np
import p213 as p2

# Asegúrate de incluir también todas las definiciones de funciones
# y cualquier otro código necesario que no esté en este fragmento.

# Actualización de los parámetros para pruebas con un rango mayor de nodos y más grafos
n_graphs = 100
n_nodes_ini = 10
n_nodes_fin = 100
step = 10

# Debido al gran número de nodos y grafos, el tiempo de ejecución puede ser considerablemente largo.
# Se ejecutarán las funciones y se capturarán los tiempos de ejecución.

# Tiempos de ejecución de Kruskal completo para un rango mayor de nodos
# Debido a la complejidad y el tiempo de ejecución, se utilizará un try-except para capturar posibles errores o interrupciones.
try:
    times_kruskal_large = p2.time_kruskal(n_graphs, n_nodes_ini, n_nodes_fin, step)
    times_cd_only_large = p2.time_kruskal_2(n_graphs, n_nodes_ini, n_nodes_fin, step)

    # Preparación de los datos para la gráfica
    nodes_large = [n for n, _ in times_kruskal_large]
    times_complete_large = [time for _, time in times_kruskal_large]
    times_cd_only_large = [time for _, time in times_cd_only_large]

    # Creación de las gráficas para el rango ampliado
    plt.figure(figsize=(12, 6))

    # Gráfica para el tiempo de ejecución de Kruskal completo con un rango ampliado de nodos
    plt.subplot(1, 2, 1)
    plt.plot(nodes_large, times_complete_large, marker='o', color='blue', label='Kruskal Completo')
    plt.title("Tiempo de Ejecución: Kruskal Completo (Rango Ampliado)")
    plt.xlabel("Número de Nodos")
    plt.ylabel("Tiempo Promedio (s)")
    plt.legend()
    plt.grid(True)

    # Gráfica para el tiempo de ejecución del CD en Kruskal con un rango ampliado de nodos
    plt.subplot(1, 2, 2)
    plt.plot(nodes_large, times_cd_only_large, marker='o', color='red', label='Solo Conjunto Disjunto')
    plt.title("Tiempo de Ejecución: Solo Conjunto Disjunto en Kruskal (Rango Ampliado)")
    plt.xlabel("Número de Nodos")
    plt.ylabel("Tiempo Promedio (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Se ha producido un error durante la ejecución: {e}")
