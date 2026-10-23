import glob
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from metodos import mejora_clasica

BAJA = "data/LOL/eval15/low"
BUENA = "data/LOL/eval15/high"
SALIDA = "resultados/clasico_eval15"
os.makedirs(SALIDA, exist_ok=True)

rutas = sorted(glob.glob(os.path.join(BAJA, "*.png")))
if not rutas:
    raise SystemExit("No encontré imágenes en " + BAJA)

p_base, s_base, p_cla, s_cla = [], [], [], []

for ruta in rutas:
    nombre = os.path.basename(ruta)
    oscura = cv2.imread(ruta)
    ref = cv2.imread(os.path.join(BUENA, nombre))
    mejorada = mejora_clasica(oscura)
    cv2.imwrite(os.path.join(SALIDA, nombre), mejorada)

    p_base.append(psnr(ref, oscura))
    s_base.append(ssim(ref, oscura, channel_axis=2))
    p_cla.append(psnr(ref, mejorada))
    s_cla.append(ssim(ref, mejorada, channel_axis=2))

print(f"Imágenes evaluadas: {len(rutas)}")
print(f"Sin mejora : PSNR {np.mean(p_base):.2f} dB | SSIM {np.mean(s_base):.3f}")
print(f"Clásico    : PSNR {np.mean(p_cla):.2f} dB | SSIM {np.mean(s_cla):.3f}")