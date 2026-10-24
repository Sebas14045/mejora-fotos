import cv2
import numpy as np

img = cv2.imread("data/foto.jpg")
if img is None:
    raise SystemExit("No encontré data/foto.jpg")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clips = [1.0, 3.0, 8.0]
tiles = [2, 8, 16]

# Reducimos cada imagen para que la cuadrícula quepa en pantalla
def pequena(im, ancho=400):
    alto = int(im.shape[0] * ancho / im.shape[1])
    return cv2.resize(im, (ancho, alto))

filas = []
for clip in clips:
    fila = []
    for t in tiles:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(t, t))
        mejorada = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        mejorada = pequena(mejorada)
        cv2.putText(mejorada, f"clip={clip} tiles={t}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        fila.append(mejorada)
    filas.append(np.hstack(fila))

cv2.imwrite("resultados/exploracion_clahe.jpg", np.vstack(filas))
print("Listo: resultados/exploracion_clahe.jpg")