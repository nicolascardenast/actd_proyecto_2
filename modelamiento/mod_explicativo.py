##################################
# Sección 1: Importaciones
##################################
# Esta sección contiene todas las importaciones necesarias para el script
from pandas import read_csv
from funciones import cambiar_hiper, calc_proba
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