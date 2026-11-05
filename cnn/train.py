import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import LOLDataset
from model import UNet


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epocas", type=int, default=100)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--parche", type=int, default=256)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--up", default="transpuesta", choices=["transpuesta", "bilineal"])
    p.add_argument("--nombre", default="unet")
    a = p.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Entrenando en:", dispositivo)
    print(f"Configuración: up={a.up} parche={a.parche} base={a.base} "
          f"lote={a.lote} épocas={a.epocas} nombre={a.nombre}")

    raiz = "data/LOL/our485"
    ds_train = LOLDataset(raiz, entrenamiento=True, parche=a.parche)
    ds_val = LOLDataset(raiz, entrenamiento=False)

    # Las últimas 25 imágenes de our485 se apartan para validar.
    # eval15 NO se toca: queda reservado para la evaluación final.
    n, n_val = len(ds_train), 25
    train = Subset(ds_train, range(0, n - n_val))
    val = Subset(ds_val, range(n - n_val, n))
    cargador = DataLoader(train, batch_size=a.lote, shuffle=True)
    cargador_val = DataLoader(val, batch_size=1)

    modelo = UNet(base=a.base, up=a.up).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    perdida = nn.L1Loss()
    # Baja el learning rate suavemente a lo largo del entrenamiento
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas)

    os.makedirs("modelos", exist_ok=True)
    ruta_mejor = f"modelos/{a.nombre}_mejor.pth"
    ruta_ultimo = f"modelos/{a.nombre}_ultimo.pth"
    mejor = 0.0
    inicio = time.time()

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        total = 0.0
        for low, high in cargador:
            low, high = low.to(dispositivo), high.to(dispositivo)
            salida = modelo(low)
            error = perdida(salida, high)
            optimizador.zero_grad()
            error.backward()