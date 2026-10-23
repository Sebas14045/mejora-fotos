import cv2 
import numpy as np

img = cv2.imread("data/foto.jpg")

if img is None:     
    raise SystemExit("No encontré data/foto.jpg, revisa el nombre y la carpeta")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_mejorado = clahe.apply(l)

lab_mejorado = cv2.merge((l_mejorado, a, b))
mejorada = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)

comparacion = np.hstack((img, mejorada))
cv2.imwrite("resultados/comparacion_clahe.jpg", comparacion)
print("Listo: revisa resultados/comparacion_clahe.jpg")