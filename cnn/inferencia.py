import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import UNet


def cargar_modelo(ruta, base=32, dispositivo=None):
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = UNet(base=base).to(dispositivo)
    modelo.load_state_dict(torch.load(ruta, map_location=dispositivo))
    modelo.eval()
    return modelo, dispositivo


def mejorar(img_bgr, modelo, dispositivo):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
    x = x.to(dispositivo)

    # La U-Net reduce 3 veces a la mitad: alto y ancho deben ser múltiplos de 8
    h, w = x.shape[2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")

    with torch.no_grad():
        y = modelo(x).clamp(0, 1)

    y = y[:, :, :h, :w].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    y = (y * 255).round().astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)