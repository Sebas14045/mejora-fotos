import cv2
import numpy as np


def reducir_ruido(img, fuerza=5):
    return cv2.fastNlMeansDenoisingColored(img, None, fuerza, fuerza, 7, 21)


def gamma(img, g=0.5):
    # g < 1 aclara las sombras
    tabla = np.array([(i / 255.0) ** g * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, tabla)


def clahe_lab(img, clip=3.0, tiles=8, color=1.4):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    a = np.clip((a.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def mejorar_sin_ruido(img, g=0.5, clip=3.0, color=1.4):
    # Orden: gamma levanta el brillo, luego CLAHE y color
    return clahe_lab(gamma(img, g), clip=clip, color=color)


def mejora_clasica(img, g=0.3, clip=3.0, color=1.0, fuerza_ruido=5):
    # El ruido se reduce primero, para que gamma y CLAHE no lo amplifiquen
    return mejorar_sin_ruido(reducir_ruido(img, fuerza_ruido), g, clip, color)