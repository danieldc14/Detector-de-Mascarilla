# Uso Heimdall-EYE:
# python detect_mask_image.py --image examples/imagen.png

# Importamos los Modulos y paquetes necesarios
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Contruimos los argumentos y analizamos los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Cargamos nuestro modelo detector facial que tenemos en el disco
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos el modelo detector de rostro facial desde el disco
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Cargamos la imagen desde entrada desde el disco , la clonamos y tomamos la nueva imagen
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# Contruimos un blob a partir de la imagen
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Pasamos el blob a traves de red y obtenemos la detecciones faciales
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# Hacemos un bucle con la deteccion facial
for i in range(0, detections.shape[2]):
	# Extraemos la confianza ( es decir , la probabilidad asociada con la deteccion)
	confidence = detections[0, 0, i, 2]

	# Filtramos las detecciones debiles asegurando que la confianza es mayor que la confianza minima
	if confidence > args["confidence"]:
		# Calculamos las coordenadas (x,y) del cuadro delimitador para el objeto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Aseguramos de que los cuadros delimitadores caigan dentro de las dimensiones del marco
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# Extraemos el ROI de la cara y pasamos de BGR a RGB y cambiamos el tamaño a 224x224 para procesarlo
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		#  Pasamos la cara a traves de modelo para determinar si la persona tiene mascarilla o no 
		(mask, withoutMask) = model.predict(face)[0]

		# Determinamos la clase etiqueta clase y el color que usaremos para dibujar el cuadro delimitador y el texto
		label = "Con Mascara" if mask > withoutMask else "Sin Mascara"
		color = (0, 255, 0) if label == "Con Mascara" else (0, 0, 255)

		# Incluimos la probabilidad de que la persona tenga la mascara
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Mostramos el frame con el resultado
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


cv2.imshow("Output", image)
cv2.waitKey(0)