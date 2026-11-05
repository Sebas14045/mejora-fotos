import argparse
import glob
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from inferencia import cargar_modelo, mejorar

p = argparse.ArgumentParser()
p.add_argument("--pesos", default="modelos/unet_mejor.pth")
p.add_argument("--base", type=int, default=32)
p.add_argument("--salida", default="resultados/cnn_eval15")
a = p.parse_args()

BAJA = "data/LOL/eval15/low"
BUENA = "data/LOL/eval15/high"
os.makedirs(a.salida, exist_ok=True)

modelo, dispositivo = cargar_modelo(a.pesos, a.base)

rutas = sorted(glob.glob(os.path.join(BAJA, "*.png")))
if not rutas:
    raise SystemExit("No encontré imágenes en " + BAJA)

ps, ss = [], []
for ruta in rutas:
    nombre = os.path.basename(ruta)
    oscura = cv2.imread(ruta)
    ref = cv2.imread(os.path.join(BUENA, nombre))
    mejorada = mejorar(oscura, modelo, dispositivo)
    cv2.imwrite(os.path.join(a.salida, nombre), mejorada)

    p_i = psnr(ref, mejorada)
    s_i = ssim(ref, mejorada, channel_axis=2)
    ps.append(p_i)
    ss.append(s_i)
    print(f"{nombre:12s} PSNR {p_i:5.2f} | SSIM {s_i:.3f}")

print(f"\nPesos: {a.pesos}")
print(f"Imágenes evaluadas: {len(rutas)}")
print(f"CNN : PSNR {np.mean(ps):.2f} dB | SSIM {np.mean(ss):.3f}")