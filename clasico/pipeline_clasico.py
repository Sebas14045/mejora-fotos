import cv2
import numpy as np
import os
os.makedirs("resultados", exist_ok=True)

def reducir_ruido(img, fuerza=7):
    # fuerza: más alto = más suave, pero se pierde detalle fino
    return cv2.fastNlMeansDenoisingColored(img, None, fuerza, fuerza, 7, 21)


def clahe_lab(img, clip=2.0, tiles=8, color=1.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)

    # Refuerza el color alrededor de 128 (el "gris" en a y b)
    a = np.clip((a.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def etiqueta(im, texto):
    im = im.copy()
    cv2.putText(im, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


def pequena(im, ancho=400):
    alto = int(im.shape[0] * ancho / im.shape[1])
    return cv2.resize(im, (ancho, alto))


img = cv2.imread("data/foto.jpg")
if img is None:
    raise SystemExit("No encontré data/foto.jpg")

a_original = img
b_clahe = clahe_lab(img, clip=3.0)
c_clahe_color = clahe_lab(img, clip=3.0, color=1.4)
d_completo = clahe_lab(reducir_ruido(img), clip=3.0, color=1.4)

fila = np.hstack([
    pequena(etiqueta(a_original, "original")),
    pequena(etiqueta(b_clahe, "solo CLAHE")),
    pequena(etiqueta(c_clahe_color, "CLAHE + color")),
    pequena(etiqueta(d_completo, "ruido + CLAHE + color")),
])
cv2.imwrite("resultados/pipeline_clasico.jpg", fila)
print("Listo: resultados/pipeline_clasico.jpg")