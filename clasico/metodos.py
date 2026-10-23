import cv2
import numpy as np


def reducir_ruido(img, fuerza=5):
    return cv2.fastNlMeansDenoisingColored(img, None, fuerza, fuerza, 7, 21)


def clahe_lab(img, clip=3.0, tiles=8, color=1.4):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    a = np.clip((a.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * color + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def mejora_clasica(img):
    return clahe_lab(reducir_ruido(img))