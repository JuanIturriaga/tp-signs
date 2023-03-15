########################################### MODELO ###########################################
# CNN-ReLU-MaxPooling

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow.keras as keras

def create_model_sign01 (
        
    #lote de datos
    input_shape = (100, 100, 3),
    num_labels = 10,                #Fijo! son 10 números del 0 al 9 (10 clases)
    pool_size = 2,
    dropout = 0.2,
    
    #Filtros para 4 capas convolucionales (en orden)
    filters = [96, 128, 160, 192],
    kernel_sizes = [3, 3, 3, 3]  

):

    model = Sequential()
    model.add(
        Conv2D(
            filters=filters[0],
            kernel_size=kernel_sizes[0],
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        MaxPooling2D(pool_size)
    )
    model.add(
        Conv2D(
            filters=filters[1],
            kernel_size=kernel_sizes[1],
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(pool_size)
    )
    model.add(
        Conv2D(
            filters=filters[2],        
            kernel_size=kernel_sizes[2],
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=filters[3],
            kernel_size=kernel_sizes[3],
            activation='relu'
        )
    )
    model.add(
        Flatten()
    )
    model.add(
        Dropout(dropout)
    )
    model.add(
        Dense(
            num_labels, 
            activation="softmax"
        )
    )

    return model