from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

path_dir = 'C:/Users/Samuel EA/Proyects/multiclase/'  #Dirección de donde se encuentra la carpeta con las imagenes
train_path = path_dir + "data/train"
test_path = path_dir + "data/test"
val_path = path_dir + "data/train"

# Generadores de imágenes
train_datagen = ImageDataGenerator(
    rescale=1. / 255, #normalizado
    zoom_range=0.2,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 2

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(260, 260),
    batch_size=batch_size,
    class_mode='sparse'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(260, 260),
    batch_size=batch_size,
    class_mode='sparse'
)

val_set = val_datagen.flow_from_directory(
    val_path,
    target_size=(260, 260),
    batch_size=batch_size,
    class_mode='sparse'
)

# Usamos VGG-19 para el pre-entrenamiento
vgg = VGG19(input_shape=(260, 260, 3), weights='imagenet', include_top=False)

# Congelamos las capas pre-entrenadas
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(training_set.num_classes, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

# Compilamos el modelo
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

# Añadimos detenimiento temprano
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Entrenamos el modelo utilizando los generadores de imágenes
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=10,
    validation_data=val_set,
    validation_steps=len(val_set),
    callbacks=[early_stop]
)

#graficamos el valor de exactitud del modelo

plt.plot(model.history['accuracy'], label='train acc')

plt.plot(model.history['val_accuracy'], label='val acc')

plt.legend()

#plt.savefig('vgg-acc-rps-1.png')

plt.show()

# graficamos el valor de perdida del modelo
plt.plot(model.history['loss'], label='train loss')
plt.plot(model.history['val_loss'], label='val loss')
plt.legend()
#plt.savefig('vgg-loss-rps-1.png')
plt.show()