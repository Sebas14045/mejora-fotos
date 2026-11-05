import torch
import torch.nn as nn


def bloque(entrada, salida):
    # Dos convoluciones 3x3 seguidas de ReLU
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(salida, salida, 3, padding=1), nn.ReLU(inplace=True),
    )


def subir(entrada, salida, modo):
    if modo == "bilineal":
        # Agranda por interpolación y luego convoluciona: evita el patrón de rejilla
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(entrada, salida, 3, padding=1),
        )
    # Versión original (v1)
    return nn.ConvTranspose2d(entrada, salida, 2, stride=2)


class UNet(nn.Module):
    def __init__(self, base=32, up="transpuesta"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Codificador: reduce la imagen y gana contexto
        self.e1 = bloque(3, base)
        self.e2 = bloque(base, base * 2)
        self.e3 = bloque(base * 2, base * 4)
        self.fondo = bloque(base * 4, base * 8)

        # Decodificador: vuelve a agrandar, recibiendo los detalles del codificador
        self.up3 = subir(base * 8, base * 4, up)
        self.d3 = bloque(base * 8, base * 4)
        self.up2 = subir(base * 4, base * 2, up)
        self.d2 = bloque(base * 4, base * 2)
        self.up1 = subir(base * 2, base, up)
        self.d1 = bloque(base * 2, base)

        self.salida = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        f = self.fondo(self.pool(e3))

        # torch.cat une el detalle fino del codificador (conexión de salto)
        d3 = self.d3(torch.cat([self.up3(f), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        # Aprendizaje residual: la red predice la corrección, no la imagen completa
        return x + self.salida(d1)