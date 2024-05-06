import tensorflow as tf     # Para la red neuronal
import time                 # Para medir el tiempo de ejecución

# Definir la función que permitirá modificar los hiperparámetros del modelo
def cambiar_hiper(num_neur: int, fun_act: str, opti: str, datos: tuple) -> tuple:
    """
    Función para modificar los hiperparámetros del modelo y medir su rendimiento.
    
    Args:
    - num_neur: Número de neuronas para la primera capa oculta.
    - fun_act: Función de activación para la segunda capa oculta.
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
    model.add(tf.keras.layers.InputLayer(input_shape=[X_train.shape[1]]))
    model.add(tf.keras.layers.Flatten())

    # Primera modificación.
    model.add(tf.keras.layers.Dense(num_neur, activation="relu"))

    # Segunda modificación.
    model.add(tf.keras.layers.Dense(100, activation=fun_act))
    
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    hidden1 = model.layers[1]
    weights, biases = hidden1.get_weights()

    # Tercera modificación.
    model.compile(loss="sparse_categorical_crossentropy",
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

    return (round(loss, 4), round(accuracy, 4), round(execution_time, 2))