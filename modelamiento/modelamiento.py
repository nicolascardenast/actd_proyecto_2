##################################
# Sección 1: Importaciones
##################################
# Esta sección contiene todas las importaciones necesarias para el script
from pandas import read_csv
from funciones import cambiar_hiper, calc_proba
from sklearn.model_selection import train_test_split
from itertools import product 
from time import time


##################################
# Sección 2: Lectura de Datos
##################################
# En esta sección se lee la información necesaria para el modelo
ruta_datos = 'limpieza_y_alistamiento/datos_limpios.csv'
datos = read_csv(ruta_datos)
col_innecesarias = ['SEX', 'EDUCATION', 'MARRIAGE']
col_necesarias = [col for col in datos.columns if col not in col_innecesarias]
datos = datos[col_necesarias]


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
num_neuronas_lista = [50, 100, 200]
funciones_activacion_lista = ['relu', 'sigmoid', 'tanh']
optimizadores_lista = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'nadam']

# Parámetros de comparación
mejor_perdida = float('inf')
mejor_precision = 0
mejores_hiperparametros = {}

# Contador de iteraciones
total_iteraciones = 0

# Iniciar el temporizador
start_time = time()

# Iterar a través de todas las combinaciones de hiperparámetros
for num_neur_1, num_neur_2, fun_act_1, fun_act_2, opti in product(num_neuronas_lista, num_neuronas_lista, funciones_activacion_lista, funciones_activacion_lista, optimizadores_lista):
    # Incrementar el contador de iteraciones
    total_iteraciones += 1
    
    # Cambiar los hiperparámetros y entrenar el modelo
    perdida, precision, _ = cambiar_hiper(num_neur_1, num_neur_2, fun_act_1, fun_act_2, opti, datos)
    
    # Actualizar el mejor modelo si la pérdida mejora
    if perdida < mejor_perdida:
        mejor_perdida = perdida
        mejor_precision = precision
        mejores_hiperparametros = {'num_neur_1': num_neur_1, 'num_neur_2': num_neur_2, 'fun_act_1': fun_act_1, 'fun_act_2': fun_act_2, 'opti': opti}

# Calcular el tiempo de ejecución
tiempo_ejecucion = time() - start_time

# Imprimir resultados
print("Mejor Pérdida:", mejor_perdida)
print("Mejor Precisión:", mejor_precision)
print("Mejores Hiperparámetros:", mejores_hiperparametros)
print("Total de Iteraciones:", total_iteraciones)
print("Tiempo de Ejecución:", tiempo_ejecucion, "segundos")
'''


##################################
# Sección 4: Predicciones
##################################
# En esta sección se realizan las predicciones
_, _, _, modelo = cambiar_hiper(50, 50, 'relu', 'sigmoid', 'nadam', datos_modelo)


##################################
# Sección 5: Función del Tablero
##################################
# Definir función que calcula probabilidad de default de un cliente
def proba_dado_id(id: int) -> float:
    """
    Función que calcula la probabilidad de default de un cliente.
    
    Args:
    - id: ID del cliente.
    
    Returns:
    - Float: Probabilidad de default de cliente
    """
    proba = calc_proba(id, datos, modelo)[0][0]
    print("Probabilidad de default para el cliente con ID", str(id) + ":",
           str(round(proba * 100, 2)) + '%')
    return proba

proba_dado_id(12)