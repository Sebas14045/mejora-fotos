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
    a = p.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Entrenando en:", dispositivo)

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

    modelo = UNet(base=a.base).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    perdida = nn.L1Loss()
    # Baja el learning rate suavemente a lo largo del entrenamiento
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=a.epocas)

    os.makedirs("modelos", exist_ok=True)
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
            optimizador.step()
            total += error.item()

        programador.step()

        if epoca % 5 == 0 or epoca == a.epocas:
            modelo.eval()
            valores = []
            with torch.no_grad():
                for low, high in cargador_val:
                    low, high = low.to(dispositivo), high.to(dispositivo)
                    salida = modelo(low).clamp(0, 1)
                    valores.append(psnr(salida, high))
            p_val = sum(valores) / len(valores)
            marca = ""
            if p_val > mejor:
                mejor = p_val
                torch.save(modelo.state_dict(), "modelos/unet_mejor.pth")
                marca = "  <- mejor"
            lr_actual = optimizador.param_groups[0]["lr"]
            print(f"Época {epoca:3d} | pérdida {total / len(cargador):.4f} | "
                  f"PSNR val {p_val:.2f} dB | lr {lr_actual:.2e}{marca} | "
                  f"{time.time() - inicio:.0f}s")

    torch.save(modelo.state_dict(), "modelos/unet_ultimo.pth")
    print("Listo. Mejor PSNR de validación:", round(mejor, 2))


if __name__ == "__main__":
    main()