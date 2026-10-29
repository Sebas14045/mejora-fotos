# mejora-fotos

Mejora de fotografías en condiciones de poca luz. Este repositorio arranca con un
**baseline clásico** (sin aprendizaje automático) que combina técnicas de procesamiento
de imágenes con OpenCV y se evalúa de forma objetiva sobre el dataset **LOL** usando las
métricas PSNR y SSIM. La siguiente fase compara este baseline contra una CNN.

## ¿Qué hace?

A partir de una imagen oscura y ruidosa, el pipeline clásico:

1. **Reduce el ruido** con `fastNlMeansDenoisingColored` (antes de amplificar la señal,
   para no realzar el grano).
2. **Levanta el brillo** con una corrección **gamma** (`g < 1` aclara las sombras).
3. **Mejora el contraste local** con **CLAHE** aplicado solo al canal de luminancia `L`
   en el espacio de color LAB.
4. **Refuerza el color** escalando los canales `a` y `b` alrededor del gris (128).

El orden importa: primero se quita el ruido, luego se aclara y por último se ajustan
contraste y color, para que gamma y CLAHE no amplifiquen el ruido.

## Estructura

```
mejora-fotos/
├── clasico/
│   ├── metodos.py            # Funciones del pipeline (gamma, CLAHE, denoise, mejora_clasica)
│   ├── evaluar_clasico.py    # Evalúa el baseline sobre LOL/eval15 con PSNR y SSIM
│   ├── ajustar_clasico.py    # Búsqueda de parámetros (gamma/clip/color) sobre LOL/our485
│   ├── explorar_clahe.py     # Rejilla visual de combinaciones clip × tiles
│   ├── pipeline_clasico.py   # Comparación visual paso a paso del pipeline
│   └── prueba_clahe.py       # Prueba inicial mínima de CLAHE
├── resultados/               # Imágenes y comparaciones generadas (no versionadas)
├── data/                     # Dataset e imágenes de entrada (no versionado)
├── requirements.txt
└── README.md
```

> `data/`, `resultados/`, el entorno virtual y los pesos de modelos (`*.pt`, `*.pth`)
> están excluidos del control de versiones.

## Requisitos

- Python 3.10+
- Las dependencias de [`requirements.txt`](requirements.txt) (OpenCV, NumPy, scikit-image,
  Matplotlib, entre otras).

## Instalación

```bash
git clone https://github.com/Sebas14045/mejora-fotos.git
cd mejora-fotos

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

El entorno virtual no se versiona: cada equipo lo recrea con estos comandos.

## Datos

El proyecto espera los datos en la carpeta `data/` (que debes crear). Para la evaluación
y el ajuste de parámetros se usa el dataset **LOL** (Low-Light): 500 pares de imágenes
con poca luz y con luz normal, divididos en 485 para entrenamiento y 15 para prueba,
introducido en el paper *Deep Retinex Decomposition for Low-Light Enhancement*
(Wei et al., 2018).

Una copia está disponible en Google Drive, enlazada desde el ejemplo de Keras sobre mejora
de baja luz: https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6

Debe quedar con esta estructura:

```
data/
├── foto.jpg                  # Imagen suelta para las pruebas visuales
└── LOL/
    ├── eval15/
    │   ├── low/              # Imágenes oscuras
    │   └── high/             # Referencia bien iluminada
    └── our485/
        ├── low/
        └── high/
```

Las imágenes `low/` y `high/` deben compartir el mismo nombre de archivo, ya que las
métricas comparan cada resultado con su referencia.

## Uso

Ejecuta los scripts desde la **raíz del proyecto** (las rutas a `data/` y `resultados/`
son relativas a ella):

```bash
# Comparación visual del pipeline completo (necesita data/foto.jpg)
python clasico/pipeline_clasico.py

# Rejilla de combinaciones clip × tiles de CLAHE
python clasico/explorar_clahe.py

# Evaluar el baseline clásico sobre LOL/eval15 (PSNR + SSIM)
python clasico/evaluar_clasico.py

# Buscar los mejores parámetros (gamma/clip/color) sobre LOL/our485
python clasico/ajustar_clasico.py
```

`evaluar_clasico.py` guarda las imágenes mejoradas en `resultados/clasico_eval15/` e
imprime el PSNR/SSIM medio con y sin mejora, para medir cuánto aporta el pipeline frente
a la imagen oscura original.

## Funciones principales (`clasico/metodos.py`)

| Función | Descripción |
|---|---|
| `reducir_ruido(img, fuerza)` | Denoising non-local means a color. |
| `gamma(img, g)` | Corrección gamma; `g < 1` aclara sombras. |
| `clahe_lab(img, clip, tiles, color)` | CLAHE sobre `L` en LAB + refuerzo de color en `a`/`b`. |
| `mejorar_sin_ruido(img, g, clip, color)` | Gamma → CLAHE + color. |
| `mejora_clasica(img, ...)` | Pipeline completo: ruido → gamma → CLAHE + color. |

## Métricas

- **PSNR** (dB): error píxel a píxel frente a la referencia. Más alto es mejor.
- **SSIM** (0 a 1): similitud de estructura, contraste y luminancia. Más cerca de 1 es mejor.

Ambas miden parecido con la referencia, no calidad estética, y pueden no coincidir entre sí
ni con la percepción visual.

## Parámetros del baseline

Los parámetros se eligieron con una búsqueda sobre 40 imágenes de `LOL/our485`
(`ajustar_clasico.py`) y se evaluaron **una sola vez** sobre `LOL/eval15`, para evitar
ajustar con el conjunto de prueba.

| Parámetro | Valor |
|---|---|
| `g` (gamma) | 0.3 |
| `clip` (CLAHE) | 3.0 |
| `color` | 1.0 |
| `fuerza_ruido` | 5 |

## Resultados (LOL/eval15)

| | PSNR (dB) | SSIM |
|---|---|---|
| Sin mejora | 7.77 | 0.191 |
| Baseline clásico | 17.10 | 0.728 |
| CNN | pendiente | pendiente |

## Estado y próximos pasos

- [x] Baseline clásico con ajuste de parámetros y evaluación con PSNR y SSIM
- [ ] CNN entrenada sobre `LOL/our485` y evaluada sobre `LOL/eval15`
- [ ] Comparación clásico vs. CNN
- [ ] Aplicación web (Flask) para subir una foto y ver original, clásico y CNN lado a lado

El entrenamiento de la CNN requerirá **PyTorch**, que no está en `requirements.txt` porque
la versión depende del hardware de cada equipo. Se instala aparte siguiendo las
instrucciones de https://pytorch.org. Los pesos entrenados (`*.pth`) no se versionan.