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
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

val_set = val_datagen.flow_from_directory(
    val_path,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

# Usamos VGG-19 para el pre-entrenamiento
vgg = VGG19(input_shape=(150, 150, 3), weights='imagenet', include_top=False)

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

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('vgg-acc-rps-1.png')
plt.show()

# graficamos el valor de perdida del modelo
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('vgg-loss-rps-1.png')
plt.show()

# Guardamos el modelo
model.save('model.h5')

# Evaluamos el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_set, steps=len(test_set))

# Hacemos predicciones en nuevas imágenes
prediction_path = path_dir + "data/test/predict"

for img in os.listdir(prediction_path):
    img_path = prediction_path + "/" + img
    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    plt.show()
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img)
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
