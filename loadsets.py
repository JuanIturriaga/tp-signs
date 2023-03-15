#Definición de paths y directorios
path = '.\\'
dir_train = 'dataset/train'
dir_validation = 'dataset/validation'
dir_test = 'dataset/test'
dir_mtest = 'dataset/meta_test'

from tensorflow.keras.preprocessing import image

def load_sets (
    
    #directoriros de origen de los datos
    path_train = path+dir_train,
    path_validation = path+dir_validation,
    path_test = path+dir_test,
    path_mtest = path+dir_mtest,

    #parámetros de augmentation
    image_size = 100, 
    rotation_range = 5,              #grados de rotacion aleatoria
    width_shift_range = 0.1,         #desplazamiento horizontal
    height_shift_range = 0.1,        #desplazamiento vertical
    zoom_range = 0.1,                #[1-zoom_range, 1+zoom_range]
    brightness_range = [0.5, 1.0],
    fill_mode = 'nearest', 
    cval = 0.0,
    horizontal_flip = True,
    vertical_flip = False
):
    # creo generador de augmentation teniendo en cuenta que
    # son imágenes de manos.
    datagen = image.ImageDataGenerator(
        rotation_range = rotation_range ,
        width_shift_range = width_shift_range ,
        height_shift_range = height_shift_range ,
        zoom_range = zoom_range ,
        brightness_range = brightness_range ,
        fill_mode = fill_mode ,
        cval = cval ,
        horizontal_flip = horizontal_flip ,
        vertical_flip = vertical_flip 
    )

    it_train = None
    it_validation = None
    it_test = None
    it_mtest = None

    if (path_train != ''):
        print ("Train:")
        it_train = datagen.flow_from_directory(
            path_train, 
            shuffle = True,
            target_size = (image_size,image_size),
            batch_size = 1,
            class_mode = 'categorical',
        )

    if (path_validation != ''):
        print ("Validation:")
        it_validation = datagen.flow_from_directory(
            path_validation, 
            shuffle = True,
            target_size = (image_size,image_size),
            batch_size = 1,
            class_mode = 'categorical'
        )


    datagen_test = image.ImageDataGenerator(
        rotation_range = 0 ,
        width_shift_range = 0 ,
        height_shift_range = 0 ,
        zoom_range = 0 ,
        fill_mode = fill_mode ,
        cval = cval ,
        horizontal_flip = False,
        vertical_flip = False 
    )        

    if (path_test != ''):
        print ("Test:")
        it_test = datagen_test.flow_from_directory(
            path_test, 
            shuffle = True,
            target_size = (image_size,image_size),
            batch_size = 1,
            class_mode = 'categorical'
        )

    if (path_mtest != ''):
        print ("Meta-test:")
        it_mtest = datagen_test.flow_from_directory(
            path_mtest, 
            shuffle = True,
            target_size = (image_size,image_size),
            class_mode = 'categorical'
        )

    return [it_train, it_validation, it_test, it_mtest]
        


