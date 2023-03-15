########################################### MODELO ###########################################
# CNN-ReLU-MaxPooling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten

import tensorflow.keras as keras

def create_model_sign03 (
        
    #lote de datos
    input_shape = (100, 100, 3),
    num_labels = 10,                #Fijo! son 10 números del 0 al 9 (10 clases)
    
    #Capas adicionales
    DenseDims = [128, 256],
    DenseActivations = ['relu', 'relu'],
    DropoutRates = [0.4, 0.2]

):
    
    # Cargo el modelo sin la fase de clasificación y fijando el tamaño de la capa de 
    # entrada según el tamaño de las imágenes que usaremos:
    fase_features = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    fase_features.trainable = False # Freezo los parámetros!


    model = Sequential(fase_features)

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