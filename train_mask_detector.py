#!/usr/bin/env python
# Uso Heimdall-EYE
# python train_mask_detector.py --dataset dataset

# Importamos los paquetes necesarios y los modulos
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Contruimos el analizador de argumentos y analizamos esos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Inicializamos la tasa de aprendizaje inicial , la cantidad de epocas
# Para entrenar , y tamaño del lote
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Tomamos la lista de imagenes de nuestra base de datos , luego inicilizamos la lista
# La lista de datos ( es la base de datos con las imagenes)
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Recorremos las rutas de imagen
for imagePath in imagePaths:
	# Extraemos la etiqueta de clase del nombre del archivo
	label = imagePath.split(os.path.sep)[-2]

	# Cargamos la imagen de entrada (224x224) y la preprocesamos
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# Actualizamos las listas de datos y las etiquetas respectivamente
	data.append(image)
	labels.append(label)

# Convertimos los datos y las etiquetas a  Arrays (NumPy) 
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Realizamos una codificacion en caliente en las etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Particionamos los datos en divisiones de entrenamiento y prueba utilizando
# Utilizando el 75% de los datos para la capacitacion y el 25 % para las pruebas restantes
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#  Contruimos el genereador de imagenes de entrenamiento para el aumento de datos
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Cargamos la red MobileNetV2 asegurando que los cojunto de capas FC principales
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Contruimos la cabeza del modleo que se colora encima de el modelo base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Colocamos el modelo de Cabeza FC en la parte superior del modelo base ( esto se convertira en el modelo real que entrenamos)
model = Model(inputs=baseModel.input, outputs=headModel)

# Repitimos todas las Capas en el modelo base y congelamos para que no se actualize
# Durante el primer proceso de capacitacion
for layer in baseModel.layers:
	layer.trainable = False

# Compilamos el modelo
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Entrenamos 
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Hacemos predicciones sobre el conjunto de pruebas
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Para cada imagen en el cojunto de pruebas necesitamos encontrar el indice de
# Etiqueta con la mayor probabilidad pronosticada correspodiente
predIdxs = np.argmax(predIdxs, axis=1)

# Mostramos un informe de clasificacion bien formateado
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# Trazamos la perdida de entrenamiento y la precision
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])