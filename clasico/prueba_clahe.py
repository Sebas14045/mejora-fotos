import cv2 
import numpy as np
import os
os.makedirs("resultados", exist_ok=True)

img = cv2.imread("data/foto.jpg")

if img is None:     
    raise SystemExit("No encontré data/foto.jpg, revisa el nombre y la carpeta")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))
l_mejorado = clahe.apply(l)

lab_mejorado = cv2.merge((l_mejorado, a, b))
mejorada = cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2BGR)

comparacion = np.hstack((img, mejorada))
cv2.imwrite("resultados/comparacion_clahe_v2.jpg", comparacion)
print("Listo: revisa resultados/comparacion_clahe_v2.jpg")