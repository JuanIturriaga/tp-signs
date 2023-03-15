
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def plot_img (image, figsize = (0.7,0.7)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.axis('off') 
    plt.show()
    plt.close('all')

# Graficar la performance de entrenamiento:
def graficar_history(historial, metrica, filename = ""):
    h = historial.history
    epochs = range(1, len(h[metrica]) + 1)
    plt.figure(1, figsize=(20, 16))
    plt.clf()
    
    # Gráfico:
    plt.plot(epochs, h[metrica], "m", label= metrica + " del entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel(metrica)
    plt.legend()

    # Guardamos la figura:
    if filename != "":
        plt.savefig(filename + ".jpg")    


def draw_history(history, filename = ""):

    plt.title("Precisión de train y validation")
    
    acc      = history.history["accuracy"]
    epochs   = range(1, len(acc) + 1)
    plt.plot(epochs, acc, color="#D361FF", marker= '.', label="Training acc")

    if "val_accuracy" in history.history.keys():
        val_acc  = history.history["val_accuracy"]
        plt.plot(epochs, val_acc, color="#50E7E8", marker= '+', label="Validation acc")

    plt.legend()
    plt.figure()
    

    plt.title("Error de train y validation")
    
    loss     = history.history["loss"]
    plt.plot(epochs, loss, color="#FF618D", marker= '.', label="Training loss")

    if "val_loss" in history.history.keys():
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, color="#509BE8", marker= '+', label="Validation loss")
    
    plt.legend()
    
    
    plt.show()    

    # Guardamos la figura:
    if filename != "":
        plt.savefig(filename + ".jpg")  

        

# Graficar imágenes de datasets (un data set por cada columna)
# sets: lista de iteradores de datagen
def graficar_sets(sets, irows = 3, isize = 25):
    icols = len(sets) #una fila por cada set
    fig, ax = plt.subplots(nrows=irows, ncols=icols, figsize=(isize,isize))

    # Itero y voy mostrando las imágenes:
    for i in range(icols):
        if sets[i] is None:
            continue
        for j in range(irows):
            x, y = next(sets[i])
            image = x[0].astype('uint8')
            
            # Muestro:
            ax[j][i].imshow(image)
            ax[j][i].axis('off')    

# 'full' : rellena con espacios negros, no pierde información
def box_square_full(image):
    w = image.width
    h = image.height
    size = max(w,h)
    pad_w = (size-w)/2
    pad_h = (size-h)/2
    # box = (left, upper, right, lower) 
    return (0-pad_w, 0-pad_h, w+pad_w, h+pad_h)

# 'zoom' : recorta bordes, centrando la imágen
def box_square_zoom(image):
    w = image.width
    h = image.height
    size = min(w,h)
    pad_w = (w-size)/2
    pad_h = (h-size)/2
    # box = (left, upper, right, lower) 
    return ( pad_w , pad_h, w-pad_w, h-pad_h)

#recorta la imagen por tipo
def crop_scuare(image, btype='zoom'):
    box_func = {
        "zoom" : box_square_zoom,
        "full" : box_square_full,
    }
    box = box_func[btype](image)
    return image.crop(box)

#convierte una imagen a proporciones cuadradas    
def resize_scuare(image, size=150, btype='zoom'):
    result = crop_scuare(image, btype) 
    result = result.resize((size, size))
    return result