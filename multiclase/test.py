from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

path_dir = 'C:/Users/Samuel EA/Proyects/multiclase/'  #Dirección de donde se encuentra la carpeta con las imagenes

# Hacemos predicciones en nuevas imágenes
prediction_path = path_dir + "data/test/predict"

# Cargamos el modelo
loaded_model = load_model("model.h5")

for img in os.listdir(prediction_path):
    img_path = prediction_path + "/" + img
    img = image.load_img(img_path, target_size=(130, 130))
    plt.imshow(img)
    plt.show()
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = loaded_model.predict(img)
    class_index = np.argmax(pred)

    if class_index == 0:
        print("Agave")
    elif class_index == 1:
        print("Brocoli")
    elif class_index == 2:
        print("Cebolla")
    elif class_index == 3:
        print("Jitomate")
    elif class_index == 4:
        print("Lechuga")
    elif class_index == 5:
        print("Maiz")
    elif class_index == 6:
        print("Trigo")
    elif class_index == 7:
        print("Zanahoria")
    else:
        print("Desconocido")