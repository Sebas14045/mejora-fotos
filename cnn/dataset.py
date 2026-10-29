import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LOLDataset(Dataset):
    def __init__(self, raiz, entrenamiento=True, parche=256):
        self.bajas = sorted(glob.glob(os.path.join(raiz, "low", "*.png")))
        self.altas_dir = os.path.join(raiz, "high")
        self.entrenamiento = entrenamiento
        self.parche = parche

    def __len__(self):
        return len(self.bajas)

    def __getitem__(self, i):
        ruta = self.bajas[i]
        low = cv2.imread(ruta)
        high = cv2.imread(os.path.join(self.altas_dir, os.path.basename(ruta)))
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)

        if self.entrenamiento:
            h, w = low.shape[:2]
            y = random.randint(0, h - self.parche)
            x = random.randint(0, w - self.parche)
            low = low[y:y + self.parche, x:x + self.parche]
            high = high[y:y + self.parche, x:x + self.parche]
            if random.random() < 0.5:
                low, high = np.fliplr(low), np.fliplr(high)
            if random.random() < 0.5:
                low, high = np.flipud(low), np.flipud(high)

        def a_tensor(im):
            return torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255.0

        return a_tensor(low), a_tensor(high)


if __name__ == "__main__":
    ds = LOLDataset("data/LOL/our485", entrenamiento=True)
    low, high = ds[0]
    print("Pares:", len(ds))
    print("Forma:", tuple(low.shape), "| rango:", float(low.min()), "a", float(low.max()))