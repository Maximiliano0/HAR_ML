# 🏃 Human Action Recognition — Clasificación de Actividades Humanas

Clasificación de imágenes de actividades humanas utilizando el dataset [Human Action Recognition (HAR)](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) de Kaggle. Se implementa un pipeline completo de Machine Learning: exploración de datos, preprocesamiento (CLAHE + oversampling), entrenamiento de una CNN **ConvNeXt-Base** con **transfer learning** desde ImageNet, y una aplicación de escritorio para inferencia en tiempo real.

## 📁 Estructura del proyecto

```
tp_ml/
├── app/                        # Aplicación de escritorio (GUI Tkinter)
│   ├── app.py                  # Clasificador con interfaz gráfica
│   ├── app_description.md      # Documentación de la aplicación
│   └── app_description.html
├── auxiliares/                  # Archivos auxiliares y configuración
│   ├── pyrightconfig.json      # Configuración pyright
│   ├── requirements.txt        # Dependencias del proyecto
│   ├── teoria_proyecto.md      # Fundamentos teóricos
│   └── teoria_proyecto.html
├── data_prep/                   # Exploración y preparación de datos
│   ├── data_explore.py         # Script de exploración del dataset
│   ├── dataset_description.md  # Documentación detallada del dataset
│   ├── dataset_description.html
│   └── output/                 # Gráficos generados
├── data_trans/                  # Adecuación de imágenes
│   ├── data_adecuate.py        # Resize + CLAHE + RGB + Oversampling
│   ├── data_transform_description.md
│   ├── data_transform_description.html
│   └── output/                 # Ejemplos visuales del pipeline
├── data_train/                  # Entrenamiento de la CNN
│   ├── train_cnn.py            # ConvNeXt-Base Transfer Learning, RGB 384×288
│   ├── train_cnn_colab.ipynb   # Notebook Colab (GPU T4, AMP)
│   ├── train_results.md        # Resultados del entrenamiento
│   ├── train_results.html
│   └── output/                 # Modelo (.pth), métricas (JSON), gráficos
├── datos_har/                   # Dataset (no versionado)
│   ├── dataset/                # 12,600 imágenes originales etiquetadas
│   ├── dataset_tr/             # Imágenes procesadas (RGB, CLAHE)
│   ├── new_data/               # 5,410 imágenes sin etiqueta
│   ├── dataset.csv             # Etiquetas: filename → label
│   └── new_data.csv            # Lista de filenames sin etiqueta
└── .gitignore
```

## 📊 Dataset

| Propiedad | Valor |
|-----------|-------|
| **Fuente** | [Kaggle — HAR Dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) |
| **Clases** | 15 actividades humanas |
| **Imágenes etiquetadas** | 12,600 (840/clase, perfectamente balanceado) |
| **Imágenes procesadas** | 12,570 en dataset_tr/ (838/clase, tras filtrado + oversampling) |
| **Imágenes en entrenamiento** | 12,510 (10,008 train + 2,502 test) |
| **Formato original** | JPEG, RGB, resoluciones variadas |
| **Formato procesado** | RGB, 288×384 px (W×H) |
| **Preprocesamiento** | Resize adaptativo → CLAHE (canal L, espacio LAB) → RGB |

Las 15 clases: `calling`, `clapping`, `cycling`, `dancing`, `drinking`, `eating`, `fighting`, `hugging`, `laughing`, `listening_to_music`, `running`, `sitting`, `sleeping`, `texting`, `using_laptop`.

## 🧠 Modelo (ConvNeXt-Base — Transfer Learning, 2 fases + SWA)

| Propiedad | Valor |
|-----------|-------|
| **Arquitectura** | ConvNeXt-Base (pretrained ImageNet1K_V1) + head custom |
| **Entrada** | 3×384×288 (RGB, normalización ImageNet) |
| **Parámetros** | ~88M (backbone congelable + head entrenable) |
| **Head** | Flatten→LN(1024)→Linear(1024→512)→BN→ReLU→Drop(0.3)→Linear(512→256)→BN→ReLU→Drop(0.2)→Linear(256→15) |
| **Entrenamiento** | 2 fases: head (15 ep) → progressive unfreezing (100 ep máx) + SWA |
| **Progressive unfreezing** | F2 épocas 1-10: features[5:7]+classifier; época 11+: todo |
| **Optimizador** | AdamW (head lr=1e-3/1e-4, backbone lr=3e-5, wd=5e-4) |
| **Scheduler** | SequentialLR (LinearLR warmup 10ep + CosineAnnealingLR) |
| **Early stopping** | Paciencia 15 épocas (por test loss, fase 2) |
| **Loss** | Focal Loss (γ=2.0, label_smoothing=0.02) |
| **Gradient clipping** | max_norm=1.0 |
| **SWA** | Stochastic Weight Averaging desde época 20 de F2 (lr=1e-5) |
| **TTA** | 5 pasadas en evaluación |
| **Augmentation** | HFlip, Rotation(20°), ColorJitter, TrivialAugmentWide, Erasing(0.10) |
| **Accuracy** | **80.26%** (TTA, 5 pasadas) — 79.82% (mejor época sin TTA) |
| **Reproducibilidad** | SEED=42 en torch, numpy, random, CUDA, DataLoader, pandas |

### 🔒 Reproducibilidad

Todos los procesos estocásticos usan semilla fija `SEED=42`:

| Componente | Mecanismo |
|------------|-----------|
| Python stdlib | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` |
| cuDNN | `deterministic=True`, `benchmark=False` |
| DataLoader shuffle | `torch.Generator().manual_seed(42)` |
| DataLoader workers | `worker_init_fn` con semilla `42 + worker_id` |
| Preprocesamiento | `pandas.sample(random_state=42)` |

> **Nota Colab:** El notebook prioriza velocidad GPU activando TF32, `cudnn.benchmark=True`,
> `cudnn.deterministic=False`, `torch.compile` y `NUM_WORKERS=2`. Esto puede producir
> resultados ligeramente distintos entre ejecuciones. El script local (`train_cnn.py`) usa
> `deterministic=True`, `benchmark=False` y `NUM_WORKERS=0` para reproducibilidad exacta.

## 🖥️ Aplicación de escritorio

La aplicación (`app/app.py`) permite clasificar imágenes de forma interactiva mediante una GUI Tkinter:

1. Seleccionar una imagen desde el explorador de archivos
2. Se aplica automáticamente el pipeline de preprocesamiento (resize + CLAHE + normalización ImageNet)
3. El modelo ConvNeXt-Base clasifica la imagen en una de las 15 actividades
4. Se muestra la imagen original, la clase predicha con emoji y las probabilidades de todas las clases con barras de progreso

**Requisitos:** Modelo entrenado en `data_train/output/` (`har_cnn_swa.pth` o `har_cnn_best.pth`).

## 🚀 Cómo ejecutar

### Requisitos previos

- **Python 3.10+**
- **Credenciales de Kaggle** (`~/.kaggle/kaggle.json`) para la descarga automática del dataset

### Instalación del entorno

```powershell
# Crear entorno virtual
python -m venv har_ml_env
.\har_ml_env\Scripts\Activate.ps1

# Instalar dependencias
pip install -r auxiliares/requirements.txt
```

### Pipeline completo

```powershell
# 1. Exploración del dataset (descarga automática desde Kaggle en la primera ejecución)
python data_prep/data_explore.py

# 2. Adecuación de imágenes (CLAHE + RGB + oversampling)
python data_trans/data_adecuate.py

# 3. Entrenamiento de la CNN (requiere GPU para tiempos razonables)
python data_train/train_cnn.py

# 4. Ejecutar la aplicación de escritorio
python app/app.py
```

> La primera ejecución de `data_explore.py` descarga el dataset desde Kaggle (~328 MB).

### Solo la aplicación

Si ya se cuenta con el modelo entrenado (`data_train/output/har_cnn_swa.pth`):

```powershell
.\har_ml_env\Scripts\Activate.ps1
python app/app.py
```

## 📄 Documentación

| Documento | Descripción |
|-----------|-------------|
| [`data_prep/dataset_description.md`](data_prep/dataset_description.md) | Análisis exploratorio del dataset |
| [`data_trans/data_transform_description.md`](data_trans/data_transform_description.md) | Pipeline de preprocesamiento |
| [`data_train/train_results.md`](data_train/train_results.md) | Resultados del entrenamiento |
| [`app/app_description.md`](app/app_description.md) | Documentación de la aplicación |
| [`auxiliares/teoria_proyecto.md`](auxiliares/teoria_proyecto.md) | Fundamentos teóricos |
| [`data_train/train_cnn_colab.ipynb`](data_train/train_cnn_colab.ipynb) | Notebook Colab (GPU T4) |
