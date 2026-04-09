# Aplicación HAR — Clasificador de Actividades Humanas

## Descripción general

Aplicación de escritorio con interfaz gráfica (Tkinter) que clasifica imágenes de personas en **15 actividades humanas** utilizando un modelo **ConvNeXt-Base** entrenado con transfer learning.

---

## Modelo utilizado

| Aspecto | Detalle |
|---|---|
| **Arquitectura** | ConvNeXt-Base (torchvision) |
| **Parámetros** | ~88 millones |
| **Pesos** | `har_cnn_swa.pth` (SWA) o `har_cnn_best.pth` (fallback) |
| **Head personalizado** | Flatten → LayerNorm(1024) → Linear(1024→512) → BN → ReLU → Dropout(0.3) → Linear(512→256) → BN → ReLU → Dropout(0.2) → Linear(256→15) |
| **Entrada** | Imagen RGB de 384×288 px (H×W), normalizada con media/std de ImageNet |
| **Salida** | 15 probabilidades (softmax) — una por clase |
| **Precisión (test)** | 80.26% (TTA, 5 pasadas) — 79.82% (mejor época sin TTA) |

---

## Clases reconocidas (15)

| # | Clase | Emoji |
|---|---|---|
| 1 | Calling | 📞 |
| 2 | Clapping | 👏 |
| 3 | Cycling | 🚴 |
| 4 | Dancing | 💃 |
| 5 | Drinking | 🥤 |
| 6 | Eating | 🍽️ |
| 7 | Fighting | 🤼 |
| 8 | Hugging | 🤗 |
| 9 | Laughing | 😂 |
| 10 | Listening to Music | 🎧 |
| 11 | Running | 🏃 |
| 12 | Sitting | 🪑 |
| 13 | Sleeping | 😴 |
| 14 | Texting | 📱 |
| 15 | Using Laptop | 💻 |

---

## Pipeline de preprocesamiento

La app replica el mismo pipeline que se usó durante el entrenamiento (`data_adecuate.py`):

1. **Lectura**: `cv2.imread()` (BGR)
2. **Resize**: a 288×384 (W×H) con interpolación adaptativa:
   - `INTER_CUBIC` si la imagen es más chica que el target (upscale)
   - `INTER_AREA` si es más grande (downscale)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization):
   - Conversión a espacio LAB
   - Ecualización del canal L con `clipLimit=2.0`, `tileGridSize=(8, 8)`
   - Reconversión a BGR
4. **Conversión a RGB** para el modelo
5. **Normalización ImageNet**: mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`

---

## Estructura del código (`app.py`)

### Constantes y configuración (líneas 1-70)

- `BASE_DIR`, `MODEL_DIR`, `MODEL_PATH`: rutas relativas al proyecto
- `IMG_H`, `IMG_W`: dimensiones de la imagen de entrada (384×288)
- `CLASSES`: lista de las 15 clases ordenadas alfabéticamente
- `CLASS_EMOJIS`: emoji representativo de cada clase
- Paleta de colores *warm neutral* (fondo claro, acentos violeta/verde)

### `preprocess_image(img_path)` — Preprocesamiento

```python
def preprocess_image(img_path: str) -> tuple[np.ndarray, np.ndarray]:
```

- **Entrada**: ruta a un archivo de imagen
- **Salida**: tupla `(img_clahe_rgb, img_original_rgb)`
- Aplica resize + CLAHE como se describe arriba
- Devuelve la imagen procesada con CLAHE (para clasificar) y la **imagen original** en RGB (para mostrar en la GUI)

### `build_model()` — Construcción y carga del modelo

```python
def build_model() -> nn.Module:
```

- Crea `ConvNeXt-Base` sin pesos preentrenados (`weights=None`)
- Reemplaza el classifier con el head custom de 3 capas lineales
- Carga los pesos desde el archivo `.pth`
- **Limpieza de claves**: elimina prefijos `_orig_mod.` (de `torch.compile()`) y `module.` (de `DataParallel`/`SWA`)
- Pone el modelo en modo evaluación (`model.eval()`)

### `classify(model, img_rgb)` — Inferencia

```python
def classify(model: nn.Module, img_rgb: np.ndarray) -> list[tuple[str, float]]:
```

- **Entrada**: modelo cargado + imagen RGB preprocesada (numpy array)
- **Proceso**: transform → tensor → forward pass → softmax
- **Salida**: lista de 15 tuplas `(clase, probabilidad)` ordenadas de mayor a menor

### `HARApp` — Interfaz gráfica

```python
class HARApp:
    def __init__(self, root: tk.Tk):
```

**Layout de dos paneles:**

- **Panel izquierdo**: imagen **original** (sin procesar) + clase predicha (con emoji) + porcentaje de confianza en tarjeta destacada
- **Panel derecho**: 15 barras de progreso (`ttk.Progressbar`) mostrando el porcentaje real (0–100%) de cada clase, ordenadas de mayor a menor, con la clase ganadora resaltada en violeta

**Componentes:**

| Widget | Descripción |
|---|---|
| Header | Título "Human Activity Recognition" + nombre del modelo y archivo de pesos |
| Toolbar | Botón "Seleccionar imagen" + nombre del archivo |
| Panel imagen | Muestra la **imagen original** escalada proporcionalmente (máx 340px alto) |
| Resultado | Tarjeta con emoji + nombre de la clase predicha en grande |
| Confianza | Porcentaje de la clase top en verde |
| 15 barras | `ttk.Progressbar` con porcentaje real (0–100%) para cada clase |
| Status bar | Mensajes de estado (modelo cargado, procesando, resultado) |

**Tema visual:** paleta *warm neutral* con fondo claro `#faf8f5`, acentos violeta `#6c63ff` / `#a29bfe` y verde `#00b894`. Estilo `clam` de ttk para las barras de progreso.

---

## Cómo ejecutar

```bash
# Desde la raíz del proyecto, con el venv activado:
python app/app.py
```

### Requisitos

- Python 3.10+
- Virtual environment con las dependencias del proyecto instaladas
- Archivo de pesos del modelo en `data_train/output/` (`har_cnn_swa.pth` o `har_cnn_best.pth`)

### Dependencias principales

| Paquete | Uso |
|---|---|
| `torch` / `torchvision` | Modelo ConvNeXt-Base, transforms, inferencia |
| `opencv-python-headless` | Lectura de imagen, resize, CLAHE |
| `Pillow` | Conversión para display en Tkinter |
| `numpy` | Manipulación de arrays de imagen |
| `tkinter` | Interfaz gráfica (incluido con Python) |

---

## Flujo de uso

1. Ejecutar `python app/app.py`
2. Esperar a que cargue el modelo (status bar indica "Listo")
3. Clic en **"📂 Seleccionar imagen"**
4. Elegir una imagen (`.jpg`, `.png`, `.bmp`, `.webp`)
5. Se muestra la **imagen original** a la izquierda
6. A la derecha aparecen las 15 barras de progreso con el porcentaje real
7. La clase predicha se muestra en grande con su emoji y porcentaje de confianza

---

## Notas técnicas

- El modelo se carga **una sola vez** al iniciar la app y permanece en memoria
- La inferencia se ejecuta en **CPU** (`map_location="cpu"`) — no requiere GPU
- Las barras son `ttk.Progressbar` con **escala absoluta** (0–100%), mostrando el porcentaje real de cada clase
- La GUI muestra la **imagen original** sin procesar; el preprocesamiento (CLAHE) se usa internamente para la clasificación
- El pipeline de preprocesamiento es **idéntico** al usado durante el entrenamiento, garantizando consistencia
