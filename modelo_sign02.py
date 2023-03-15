########################################### MODELO ###########################################
# CNN-ReLU-MaxPooling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D

import tensorflow.keras as keras

def create_model_sign02 (
        
    #lote de datos
    input_shape = (100, 100, 3),
    num_labels = 10,                #Fijo! son 10 números del 0 al 9 (10 clases)
    pool_size = 2,
    
    #Filtros para N capas convolucionales (en orden)
    filters = [96, 128, 160, 192],
    kernel_sizes = [3, 3, 3, 3],
    activations = ['relu', 'relu', 'relu', 'relu'],
    maxpool = [1,1,1,0],

    #Capas adicionales
    DenseDims = [128, 256],
    DenseActivations = ['relu', 'relu'],
    DropoutRates = [0.4, 0.2]

):

    model = Sequential()

    # Se agregan tantas capas convolucionales como se indiquen en los parámetros
    count = len(filters)
    for i in range(count):
        model.add(
            Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                activation=activations[i],
                input_shape=input_shape
            )
        )
        if maxpool[i] == 1:
            model.add(
                MaxPooling2D(pool_size)
            )

    # Capa de GlobalAveragePooling2D
    model.add(
        GlobalAveragePooling2D(
            data_format=None, 
            keepdims=False
        )
    )

    # Capa de Flatten
    model.add(
        Flatten()
    )

    # Se agregan tantas capas dense-dropout como se indiquen en los parámetros
    count = len(DenseDims)
    for i in range(count):
        model.add(
            Dense(
                DenseDims[i], 
                activation=DenseActivations[i]
            )
        )
        model.add(
            Dropout(DropoutRates[i])
        )

    # Capa de salida
    model.add(
        Dense(
            num_labels, 
            activation="softmax"
        )
    )

    return model