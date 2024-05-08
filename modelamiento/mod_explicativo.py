##################################
# Sección 1: Importaciones
##################################
# Esta sección contiene todas las importaciones necesarias para el script
from pandas import read_csv
from funciones import crear_mod_descr
from sklearn.model_selection import train_test_split
from numpy import mean, abs
from itertools import product 
from time import time
import matplotlib.pyplot as plt
from os import path, getcwd


##################################
# Sección 2: Lectura de Datos
##################################

ruta_carpeta_proy = path.dirname(getcwd())
directorio_padre = ruta_carpeta_proy + '/actd_proyecto_2'


ruta_datos = directorio_padre+'/limpieza_y_alistamiento/datos_limpios.csv'
datos = read_csv(ruta_datos)

##################################
# Sección 3: Crear Perceptron Mult.
##################################
# En esta sección se crea la red neuronal
# Dividir los datos en características (X) y etiquetas (y)
x = datos.drop('default_payment_next_month', axis=1)
y = datos['default_payment_next_month']

# Dividir los datos en conjunto de entrenamiento, conjunto de validación y conjunto de prueba
x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
datos_modelo = x_train, x_valid, y_train, y_valid

'''
## Buscar el mejor modelo ------------------------------------------------------
# Hiperparámetros
num_neuronas_lista = [100, 200]
funciones_activacion_lista = ['sigmoid', 'tanh']
optimizadores_lista = ['adam', 'rmsprop', 'adadelta']

# Parámetros de comparación
mejor_perdida = float('inf')
peor_perdida = float('-inf')
otra_perdida = 0
mejor_precision = 0
peor_precision = 1
otra_precision = 0
mejores_hiperparametros = {}
peores_hiperparametros = {}
otros_hiperparametros = {}
mejor_modelo = None
peor_modelo = None
otro_modelo = None

# Contador de iteraciones
total_iteraciones = 0

# Iniciar el temporizador
start_time = time()

# Iterar a través de todas las combinaciones de hiperparámetros
for num_neur_1, num_neur_2, fun_act_1, fun_act_2, opti in product(num_neuronas_lista, num_neuronas_lista, funciones_activacion_lista, funciones_activacion_lista, optimizadores_lista):
    # Incrementar el contador de iteraciones
    total_iteraciones += 1
    
    # Cambiar los hiperparámetros y entrenar el modelo
    perdida, precision, modelo = crear_mod_descr(num_neur_1, num_neur_2, fun_act_1, fun_act_2, opti, datos_modelo)
    
    # Actualizar el mejor modelo si la pérdida mejora
    if perdida < mejor_perdida:
        mejor_perdida = perdida
        mejor_precision = precision
        mejores_hiperparametros = {'num_neur_1': num_neur_1, 'num_neur_2': num_neur_2, 'fun_act_1': fun_act_1, 'fun_act_2': fun_act_2, 'opti': opti}
        mejor_modelo = modelo
    # Actualizar el peor modelo si la pérdida empeora
    if perdida > peor_perdida:
        peor_perdida = perdida
        peor_precision = precision
        peores_hiperparametros = {'num_neur_1': num_neur_1, 'num_neur_2': num_neur_2, 'fun_act_1': fun_act_1, 'fun_act_2': fun_act_2, 'opti': opti}
        peor_modelo = modelo
    # Guardar otro modelo durante la iteración (no el mejor ni el peor)
    if otro_modelo is None and perdida != mejor_perdida and perdida != peor_perdida:
        otra_perdida = perdida
        otra_precision = precision
        otros_hiperparametros = {'num_neur_1': num_neur_1, 'num_neur_2': num_neur_2, 'fun_act_1': fun_act_1, 'fun_act_2': fun_act_2, 'opti': opti}
        otro_modelo = modelo

# Calcular el tiempo de ejecución
tiempo_ejecucion = time() - start_time

# Imprimir resultados
print("Mejor Pérdida:", mejor_perdida)
print("Mejor Precisión:", mejor_precision)
print("Mejores Hiperparámetros:", mejores_hiperparametros)
print("Peor Pérdida:", peor_perdida)
print("Peor Precisión:", peor_precision)
print("Peores Hiperparámetros:", peores_hiperparametros)
print("Total de Iteraciones:", total_iteraciones)
print("Tiempo de Ejecución:", tiempo_ejecucion, "segundos")

## Comparar resultados ------------------------------------------------------
# Definir los grupos y las métricas
grupos = ['Modelo 1', 'Modelo 2', 'Modelo 3']
losses = [mejor_perdida, otra_perdida, peor_perdida]
accuracies = [mejor_precision, otra_precision, peor_precision]

# Crear la gráfica de barras
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Colores para las barras
colors_loss = ['#4C72B0', '#4169E1', '#3555C7']  # Tonalidades de azul
colors_accuracy = ['#6A5ACD', '#4682B4', '#1E90FF']  # Tonalidades de azul

# Gráfico para loss
ax[0].bar(grupos, losses, color=colors_loss)
ax[0].set_title('Loss')
ax[0].set_ylabel('Valor')

# Gráfico para accuracy
ax[1].bar(grupos, accuracies, color=colors_accuracy)
ax[1].set_title('Accuracy')
ax[1].set_ylabel('Valor')

# Mostrar la gráfica
plt.tight_layout()
plt.show()
'''


##################################
# Sección 4: Responder la pregunta
##################################
# Calcular las preddiciones del modelo
_, _, mejor_modelo = crear_mod_descr(200, 200, 'tanh', 'sigmoid', 'rmsprop', datos_modelo)
predicciones_completas = mejor_modelo.predict(x)

# Calcular las influencias de cada característica
weights_capa_oculta = mejor_modelo.layers[1].get_weights()[0]
importancia_caracteristicas = mean(abs(weights_capa_oculta), axis=0)

# Mostrar resultados
nombres_caracteristicas = x.columns
dict_importancia = dict(zip(nombres_caracteristicas, importancia_caracteristicas))
dict_importancia_ordenado = dict(sorted(dict_importancia.items(), key=lambda item: item[1], reverse=True))
for caracteristica, importancia in dict_importancia_ordenado.items():
    print(f"{caracteristica}: {importancia}")

# Crear función que retorna los resultados
def result_mod_descr() -> dict:
    """
    Esta función devuelve un diccionario que contiene la importancia de las características
    derivada de un modelo previamente entrenado.

    Returns:
    - dict_importancia (dict): Un diccionario que mapea nombres de características a su importancia relativa.
    """
    return dict_importancia