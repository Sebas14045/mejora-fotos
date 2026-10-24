import glob
import itertools
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from metodos import mejorar_sin_ruido, reducir_ruido

BAJA = "data/LOL/our485/low"
BUENA = "data/LOL/our485/high"

rutas = sorted(glob.glob(os.path.join(BAJA, "*.png")))[:40]
if not rutas:
    raise SystemExit("No encontré imágenes en " + BAJA)

# El ruido se reduce una sola vez por imagen (es la parte lenta)
pares = []
for ruta in rutas:
    ref = cv2.imread(os.path.join(BUENA, os.path.basename(ruta)))
    pares.append((reducir_ruido(cv2.imread(ruta)), ref))

resultados = []
for g, clip, color in itertools.product([0.3, 0.4, 0.5, 0.6], [2.0, 3.0, 4.0], [1.0, 1.4]):
    ps, ss = [], []
    for limpia, ref in pares:
        m = mejorar_sin_ruido(limpia, g, clip, color)
        ps.append(psnr(ref, m))
        ss.append(ssim(ref, m, channel_axis=2))
    resultados.append((np.mean(ps), np.mean(ss), g, clip, color))

resultados.sort(reverse=True)
print("Mejores 5 combinaciones (ordenadas por PSNR):")
for p, s, g, clip, color in resultados[:5]:
    print(f"PSNR {p:.2f} | SSIM {s:.3f} | gamma={g} clip={clip} color={color}")