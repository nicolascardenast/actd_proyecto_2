##################################
# Sección 1: Importaciones
##################################
# Esta sección contiene todas las importaciones necesarias para el script
from pandas import read_csv
from funciones import cambiar_hiper, calc_proba, crear_mod_descr
from sklearn.model_selection import train_test_split
from itertools import product 
from time import time
import matplotlib.pyplot as plt


##################################
# Sección 2: Lectura de Datos
##################################
# En esta sección se lee la información necesaria para el modelo
ruta_datos = 'limpieza_y_alistamiento/datos_limpios.csv'
datos = read_csv(ruta_datos)
col_innecesarias = ['LIMIT_BAL', 'SEX', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4',
                     'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                         'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
datos = datos.drop(col_innecesarias, axis=1)


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


##################################
# Sección 4: Solución Preguntas
##################################
loss, acc, mejor_modelo = crear_mod_descr(200, 200, 'tanh', 'sigmoid', 'rmsprop', datos_modelo)

# Obtener los coeficientes de las neuronas de entrada
coeficientes = mejor_modelo.layers[1].get_weights()[0]

# Calcular la importancia relativa de cada variable de entrada
importancia_edad = coeficientes[0][0]
importancia_educacion = coeficientes[1][0]

# Imprimir resultados
print("Importancia relativa de la edad:", importancia_edad)
print("Importancia relativa de la educación:", importancia_educacion)