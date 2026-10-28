# mejora-fotos

Mejora de fotografías en condiciones de poca luz. Este repositorio arranca con un
**baseline clásico** (sin aprendizaje automático) que combina técnicas de procesamiento
de imágenes con OpenCV y se evalúa de forma objetiva sobre el dataset **LOL** usando las
métricas PSNR y SSIM.

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
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Datos

El proyecto espera los datos en la carpeta `data/` (que debes crear). Para la evaluación
y el ajuste de parámetros se usa el dataset **LOL** (Low-Light) con esta estructura:

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

## Estado y próximos pasos

La carpeta `clasico/` establece el baseline de referencia. El `.gitignore` ya contempla
pesos de modelos (`*.pt`, `*.pth`), anticipando una siguiente fase con métodos basados en
aprendizaje profundo para comparar contra este baseline.
