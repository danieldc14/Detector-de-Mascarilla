from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')


def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grabamos las dimensiones del marco y luego contruimos un blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Pasamos el blob a traves de la red y obtenemos las detecciones faciales
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Inicializemos nuestra lista de caras sus ubicaciones correspondientes y lista de predicciones en nuestra red de mascarillas
	faces = []
	locs = []
	preds = []

	# Hacemos un bucle para la deteccion facial
	for i in range(0, detections.shape[2]):
		# Extraemos la confianza ( es decir la probabilidad ) asociada con la deteccion facial
		confidence = detections[0, 0, i, 2]

		# Filtramos detecciones debiles asegurando que la confianza mayor siempre sera mayor que la confianza minima
		if confidence > args["confidence"]:
			# Calculamos las coordenadas (x,y) del cuadro delimitador para el objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Aseguramos de que los cuadros delimitadores caigan dentro de las dimensiones del marco 
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extraer el ROI de la cara y convertimos el BGR a RGB y cambiamos el tamaño a 224x224 y lo procesamos
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# Agregamos la cara y los cuadros delimitadores a sus respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Solo haga predicciones si se detecto al menos una cara
	if len(faces) > 0:
		# Para un inferencia mas rapida , haremos las predicciones en lotes
		# En todos , caras al mismo tiempo en lugar de predicciones de una en una
		preds = maskNet.predict(faces)

	# Devolver una tupla de 2 de las ubicaciones de la cara y sus respondientes ubicaciones
	return (locs, preds)

# Contruimos los argumentos y analizamos los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Cargamos nuestro modelo detector facial desde el disco/carpeta
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos el modelo detector de mascaras facial desde disco / carpeta
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Inicializamos el sensor de camara 
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Realizamos un bucle mientras sea verdadero nos muestre el frame
while True:
	# Toma el fotograma de la secuencia de video y cambia el tamaño a 400 pixeles el frame
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detectamos los rostros en el marco y determinamos si lleva o no mascarilla
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Recorremos las ubicaciones de los rostros faciales detectados y sus correspodientes ubicaciones
	for (box, pred) in zip(locs, preds):
		# Extraemos el cuadro delimitador y las predicciones
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Determinamos la etiqueta y el color que usaremos para dibujar el cuadro delimitador y el texto del frame
		label = "Con Mascarilla" if mask > withoutMask else "Sin Mascarilla" 
		color = (255, 255, 0) if label == "Con Mascarilla" else (0, 0, 255)

		if label== "Sin Mascarilla":
			sound.play()
			time.sleep(5)
			print("Sin Mascarilla")
		#else:
		#	print("Con Mascarilla")


		# Incluimos la probabilidad de que llevo mascara
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		

			#sound.play()

		# Mostramos el frame de la camara en pantalla
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Para salir del frame basta con pulsar Q y salimos 
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
