import tensorflow as tf     # Para la red neuronal
import time                 # Para medir el tiempo de ejecución
from pandas import DataFrame

# Definir la función que permitirá modificar los hiperparámetros del modelo
def cambiar_hiper(num_neur_1: int, num_neur_2: int, fun_act_1: str , fun_act_2: str, opti: str, datos: tuple) -> tuple:
    """
    Función para modificar los hiperparámetros del modelo y medir su rendimiento.
    
    Args:
    - num_neur_i: Número de neuronas para la capa oculta i.
    - fun_act_i: Función de activación para la capa oculta i.
    - opti: Optimizador para compilar el modelo.
    - datos: Los datos para la red neuronal
    
    Returns:
    - Tuple: Una tupla que contiene la pérdida, precisión, tiempo de ejecución y el modelo
    """
    # Inicia el temporizador
    start_time = time.time()

    # Desempaquetar datos
    X_train, X_valid, y_train, y_valid = datos
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[X_train.shape[1]]))  # Capa input
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_neur_1, activation=fun_act_1))     # Capa oculta
    model.add(tf.keras.layers.Dense(num_neur_2, activation=fun_act_2))     # Capa oculta
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))              # Capa output

    # Información
    hidden1 = model.layers[1]
    weights, biases = hidden1.get_weights()

    # Tercera modificación.
    model.compile(loss="binary_crossentropy",
                  optimizer=opti,
                  metrics=["accuracy"])
    
    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        verbose=0)
    
    # Detiene el temporizador
    end_time = time.time()  

    # Calcula el tiempo de ejecución
    execution_time = 0
    execution_time += (end_time - start_time)

    # Métricas para evaluar el modelo utilizando los datos de validación.
    loss, accuracy = model.evaluate(X_valid, y_valid)

    return (round(loss, 4), round(accuracy, 4), round(execution_time, 2), model)


# Definir función que calcula probabilidad de default de un cliente
def calc_proba(id: int, datos:DataFrame, modelo) -> float:
    """
    Función que calcula la probabilidad de default de un cliente.
    
    Args:
    - id: ID del cliente.
    - datos: DataFrame con los datos.
    - modelo: el modelo predictivo
    
    Returns:
    - Float: Probabilidad de default de cliente
    """
    client_data = datos.loc[datos['ID'] == id]
    x_client = client_data.drop(['default_payment_next_month'], axis=1)
    default_probability_client = modelo.predict(x_client)
    return default_probability_client