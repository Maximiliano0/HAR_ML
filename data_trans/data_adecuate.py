"""
data_adecuate.py
================
Script de adecuación de imágenes del dataset HAR.
Lee las imágenes de datos_har/dataset/, las redimensiona a un
tamaño estándar, descarta las más chicas (umbral mínimo 128x128),
aplica oversampling de clases minoritarias para balance,
aplica CLAHE (sobre canal L en espacio LAB) y guarda en RGB.

Pasos
-----
1. Calcular tamano promedio y redondear al estandar mas cercano
2. Filtrar imagenes menores al umbral minimo (128x128)
3. Oversampling de clases minoritarias para balance
4. Redimensionar -> CLAHE (LAB) -> RGB
5. Guardar resultados en datos_har/dataset_tr/
"""

# pylint: disable=no-member

import shutil
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

# ──────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
datos_har = project_root / "datos_har"
dataset_dir = datos_har / "dataset"
dataset_csv = datos_har / "dataset.csv"
dataset_tr = datos_har / "dataset_tr"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Resolución fija de salida (debe coincidir con la entrada del modelo)
TARGET_W, TARGET_H = 288, 384  # ancho x alto (W x H) — misma resolución que IMG_W, IMG_H en train

# Umbral mínimo para incluir imágenes (permite upscale)
MIN_SIZE = 128


# ──────────────────────────────────────────────
# 1. CALCULAR TAMAÑO PROMEDIO → TAMAÑO OBJETIVO
# ──────────────────────────────────────────────
print("=" * 60)
print("1. RESOLUCIÓN DE SALIDA (FIJA)")
print("=" * 60)

df = pd.read_csv(dataset_csv)
print(f"Imágenes etiquetadas en dataset.csv: {len(df)}")

# Leer dimensiones de todas las imágenes etiquetadas (para estadísticas)
widths, heights = [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Leyendo dimensiones"):
    img_path = dataset_dir / row["filename"]
    if img_path.exists():
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except OSError:
            pass

avg_w = np.mean(widths)
avg_h = np.mean(heights)

print(f"\nPromedio real:  {avg_w:.1f} x {avg_h:.1f}")
print(f"Tamaño objetivo (fijo para modelo): {TARGET_W} x {TARGET_H}")

# ──────────────────────────────────────────────
# 2. FILTRAR IMÁGENES MENORES AL TAMAÑO OBJETIVO
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FILTRADO DE IMÁGENES POR TAMAÑO")
print("=" * 60)

# Agregar dimensiones al dataframe
dim_map = {}
for _, row in df.iterrows():
    img_path = dataset_dir / row["filename"]
    if img_path.exists():
        try:
            with Image.open(img_path) as img:
                dim_map[row["filename"]] = img.size
        except OSError:
            pass

df["width"] = df["filename"].map(lambda f: dim_map.get(f, (0, 0))[0])
df["height"] = df["filename"].map(lambda f: dim_map.get(f, (0, 0))[1])

total_before = len(df)
df_valid = df[(df["width"] >= MIN_SIZE) & (df["height"] >= MIN_SIZE)].copy()
discarded = total_before - len(df_valid)

print(f"Imágenes totales:     {total_before}")
print(f"Imágenes válidas (≥{MIN_SIZE}x{MIN_SIZE}): {len(df_valid)}")
print(f"Imágenes descartadas: {discarded} ({discarded/total_before*100:.1f}%)")

# Distribución por clase después del filtrado
class_counts_filtered = df_valid["label"].value_counts()
print("\nDistribucion por clase (post-filtrado):")
for cls, cnt in class_counts_filtered.sort_index().items():
    print(f"  {cls:<25} {cnt:>6}")

# ──────────────────────────────────────────────
# 3. OVERSAMPLING PARA BALANCE DE CLASES
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. OVERSAMPLING PARA BALANCE")
print("=" * 60)

max_count = class_counts_filtered.max()
max_class = class_counts_filtered.idxmax()
min_count = class_counts_filtered.min()
min_class = class_counts_filtered.idxmin()
print(f"Clase con más imágenes:   {max_class} ({max_count})")
print(f"Clase con menos imágenes: {min_class} ({min_count})")
print(f"Oversampling todas las clases a {max_count} imágenes...")

rng = np.random.default_rng(42)  # noqa: F841  # semilla fija para reproducibilidad
balanced_parts = []
for label, group in df_valid.groupby("label"):
    n = len(group)
    if n >= max_count:
        balanced_parts.append(group.sample(n=max_count, random_state=42))
    else:
        # Tomar todos + muestrear con reemplazo para llegar a max_count
        extra = group.sample(n=max_count - n, replace=True, random_state=42)
        balanced_parts.append(pd.concat([group, extra], ignore_index=True))

df_balanced = pd.concat(balanced_parts, ignore_index=True)

print(f"Total imágenes después del balance: {len(df_balanced)}")
print(f"Imágenes por clase: {max_count}")
print(f"Clases: {df_balanced['label'].nunique()}")

# ──────────────────────────────────────────────
# 4. PROCESAR: RESIZE + CLAHE + RGB
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. PROCESAMIENTO DE IMAGENES (CLAHE + RGB)")
print("=" * 60)
print(f"   Resize -> {TARGET_W}x{TARGET_H}")
print("   CLAHE (clipLimit=2.0, tileGridSize=8x8) sobre canal L (LAB)")
print("   Salida en RGB (3 canales)")
print("   Interpolacion adaptativa (INTER_CUBIC upscale / INTER_AREA downscale)")

# Crear CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Crear carpeta destino (limpiar si existe)
if dataset_tr.exists():
    shutil.rmtree(dataset_tr)
dataset_tr.mkdir(parents=True)

# Guardar ejemplos para el .md (antes/después)
examples = {}  # {clase: [(original_path, processed_path), ...]}
examples_per_class = 1  # pylint: disable=invalid-name

processed = 0  # pylint: disable=invalid-name
duplicates = 0  # pylint: disable=invalid-name
for idx, row in tqdm(df_balanced.iterrows(), total=len(df_balanced), desc="Procesando"):
    src_path = dataset_dir / row["filename"]
    label = row["label"]
    # Para oversampled (duplicados), generar nombre único
    base_name = Path(row["filename"]).stem
    ext = Path(row["filename"]).suffix
    dst_name = row["filename"]  # pylint: disable=invalid-name
    dst_path = dataset_tr / dst_name
    if dst_path.exists():
        duplicates += 1
        dst_name = f"{base_name}_dup{duplicates}{ext}"  # pylint: disable=invalid-name
        dst_path = dataset_tr / dst_name

    # Leer imagen con OpenCV (BGR)
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        continue

    # 4a. Redimensionar al tamano objetivo (interpolacion adaptativa)
    h_orig, w_orig = img_bgr.shape[:2]
    needs_upscale = (w_orig < TARGET_W) or (h_orig < TARGET_H)
    interp = cv2.INTER_CUBIC if needs_upscale else cv2.INTER_AREA
    img_resized = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=interp)

    # 4b. CLAHE en canal L (LAB color space)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    l_channel = clahe.apply(l_channel)
    img_lab = cv2.merge([l_channel, a_channel, b_channel])
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # Guardar imagen procesada (RGB)
    cv2.imwrite(str(dst_path), img_clahe)
    processed += 1

    # Guardar ejemplo para documentación
    if label not in examples:
        examples[label] = (src_path, dst_path)

print(f"\nImágenes procesadas y guardadas: {processed}")
print(f"Directorio de salida: {dataset_tr}")

# ──────────────────────────────────────────────
# 5. GENERAR IMÁGENES DE EJEMPLO (ANTES/DESPUÉS)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. GENERACIÓN DE EJEMPLOS VISUALES")
print("=" * 60)

# Grilla de ejemplos: 5 clases, antes vs después
sample_classes = sorted(examples.keys())[:5]
fig, axes = plt.subplots(len(sample_classes), 2, figsize=(8, len(sample_classes) * 3))

for i, cls in enumerate(sample_classes):
    orig_path, proc_path = examples[cls]

    # Original
    img_orig = cv2.imread(str(orig_path))
    img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    axes[i][0].imshow(img_orig_rgb)
    axes[i][0].set_title("Original" if i == 0 else "")
    if i == 0:
        axes[i][0].set_title("Original", fontsize=12, fontweight="bold")
    axes[i][0].set_ylabel(cls, fontsize=9, rotation=0, labelpad=90, va="center")
    axes[i][0].set_xticks([])
    axes[i][0].set_yticks([])

    # Procesada (RGB)
    img_proc = cv2.imread(str(proc_path))
    img_proc_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    axes[i][1].imshow(img_proc_rgb)
    if i == 0:
        axes[i][1].set_title("Procesada (RGB+CLAHE)", fontsize=12, fontweight="bold")
    axes[i][1].set_xticks([])
    axes[i][1].set_yticks([])

plt.suptitle(
    "Comparación: Original vs Procesada (5 clases)",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(output_dir / "ejemplos_antes_despues.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Gráfico guardado en: {output_dir / 'ejemplos_antes_despues.png'}")

# Ejemplo detallado del pipeline para una imagen
print("\nGenerando ejemplo detallado del pipeline...")
sample_cls = sample_classes[0]
orig_path, _ = examples[sample_cls]

img_bgr = cv2.imread(str(orig_path))
img_resized = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
l_ch, a_ch, b_ch = cv2.split(img_lab)
l_ch = clahe.apply(l_ch)
img_lab = cv2.merge([l_ch, a_ch, b_ch])
img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = ["1. Original", "2. Resize", "3. CLAHE (RGB)"]
images_plot = [
    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB),
]

for ax, title, img in zip(axes, titles, images_plot):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"Pipeline de procesamiento — clase: {sample_cls}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "pipeline_ejemplo.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Gráfico guardado en: {output_dir / 'pipeline_ejemplo.png'}")

# ──────────────────────────────────────────────
# RESUMEN
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  Tamaño objetivo:         {TARGET_W} x {TARGET_H}")
print(f"  Umbral mínimo:           {MIN_SIZE} x {MIN_SIZE}")
print(f"  Imágenes originales:     {total_before}")
print(f"  Descartadas (muy chicas): {discarded}")
print(f"  Post-oversampling/clase: {max_count}")
print(f"  Total procesadas:        {processed}")
print(f"  Duplicados oversampled:  {duplicates}")
print(f"  Carpeta de salida:       {dataset_tr}")
print(f"  Ejemplos en:             {output_dir}")
print("=" * 60)
print("Procesamiento completado.")
