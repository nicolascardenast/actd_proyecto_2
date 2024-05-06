##################################
# Sección 1: Importaciones
##################################
# Esta sección contiene todas las importaciones necesarias para el script
from pandas import read_csv
from funciones import cambiar_hiper
from sklearn.model_selection import train_test_split


##################################
# Sección 2: Lectura de Datos
##################################
# En esta sección se lee la información necesaria para el modelo
ruta_datos = 'limpieza_y_alistamiento/datos_limpios.csv'
datos = read_csv(ruta_datos)
col_innecesarias = ['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',]
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

# Crear red neuronal
datos = x_train, x_valid, y_train, y_valid
cambiar_hiper(25, 'linear', 'adam', datos)