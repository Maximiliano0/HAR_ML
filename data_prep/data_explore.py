"""
data_explore.py
===============
Script de exploración del dataset Human Action Recognition (HAR).
Si la carpeta datos_har/ no existe, descarga el dataset mediante
kagglehub y renombra las carpetas y CSVs originales:
  - train/  → dataset/      (imágenes etiquetadas)
  - test/   → new_data/     (imágenes sin etiqueta)
  - Training_set.csv → dataset.csv
  - Testing_set.csv  → new_data.csv
Luego unifica new_data en dataset, analiza estructura,
distribución de clases, tamaños y características generales.

Secciones
---------
1. Carga del dataset (descarga si no existe)
2. Estructura de directorios
3. Unificación de imágenes (new_data → dataset)
4. Distribución de clases + métricas de balance
5. Análisis de tamaños de imagen
6. Estadísticas de canales (media / desviación estándar)
7. Visualización de muestras por clase
"""

import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import matplotlib.image as mpimg  # pylint: disable=wrong-import-position

# ──────────────────────────────────────────────
# RUTAS DEL PROYECTO
# ──────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
datos_har = project_root / "datos_har"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def main():  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Ejecuta la exploración completa del dataset HAR."""
    # ──────────────────────────────────────────────
    # 1. CARGA DEL DATASET (descarga si no existe)
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("1. CARGA DEL DATASET")
    print("=" * 60)

    if not datos_har.exists() or not any(datos_har.iterdir()):
        import kagglehub  # pylint: disable=import-outside-toplevel
        print("Descargando dataset desde Kaggle...")
        kagglehub.dataset_download(
            "meetnagadia/human-action-recognition-har-dataset",
            output_dir=str(datos_har),
        )

        # Renombrar carpetas y CSVs originales
        rename_dirs = {"train": "dataset", "test": "new_data"}
        rename_csvs = {"Training_set.csv": "dataset.csv", "Testing_set.csv": "new_data.csv"}

        # Si queda una subcarpeta intermedia (ej. "Human Action Recognition"), aplanar
        subdirs = [d for d in datos_har.iterdir() if d.is_dir() and d.name != ".complete"]
        if len(subdirs) == 1 and not (datos_har / "dataset").exists():
            intermediate = subdirs[0]
            for item in intermediate.iterdir():
                shutil.move(str(item), str(datos_har / item.name))
            intermediate.rmdir()
            print(f"  Aplanado: {intermediate.name}/ → datos_har/")

        for old_name, new_name in rename_dirs.items():
            old_p, new_p = datos_har / old_name, datos_har / new_name
            if old_p.exists() and not new_p.exists():
                old_p.rename(new_p)
                print(f"  Renombrado: {old_name}/ → {new_name}/")

        for old_name, new_name in rename_csvs.items():
            old_p, new_p = datos_har / old_name, datos_har / new_name
            if old_p.exists() and not new_p.exists():
                old_p.rename(new_p)
                print(f"  Renombrado: {old_name} → {new_name}")

    print(f"Ruta del dataset: {datos_har}\n")

    # Carpetas y CSVs esperados
    dataset_dir = datos_har / "dataset"
    new_data_dir = datos_har / "new_data"
    dataset_csv = datos_har / "dataset.csv"
    new_data_csv = datos_har / "new_data.csv"

    # ──────────────────────────────────────────────
    # 2. ESTRUCTURA DE DIRECTORIOS
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("2. ESTRUCTURA DE DIRECTORIOS")
    print("=" * 60)

    # Listar contenido de datos_har/
    print(f"\nContenido de datos_har/ ({datos_har}):")
    for item in sorted(datos_har.iterdir()):
        tipo = "DIR " if item.is_dir() else "FILE"
        print(f"  [{tipo}] {item.name}")

    print(f"\nDirectorio principal (dataset): {dataset_dir}")
    print(f"Directorio nuevos datos (new_data): {new_data_dir}")

    # Leer CSV de etiquetas
    df_dataset = pd.read_csv(dataset_csv) if dataset_csv.exists() else pd.DataFrame()
    print(f"\n  dataset.csv: {len(df_dataset)} filas, columnas = {list(df_dataset.columns)}")
    if new_data_csv.exists():
        df_new_data = pd.read_csv(new_data_csv)
        print(f"  new_data.csv: {len(df_new_data)} filas, columnas = {list(df_new_data.columns)}")

    # Determinar estructura (plana con CSV o subcarpetas por clase)
    dataset_subdirs = (
        sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        if dataset_dir.exists() else []
    )

    if dataset_subdirs:
        clases = dataset_subdirs
        use_csv = False
    else:
        clases = (
            sorted(df_dataset["label"].unique().tolist())
            if "label" in df_dataset.columns else []
        )
        use_csv = True

    print(f"\nNúmero de clases: {len(clases)}")
    print(f"Clases: {clases}")
    print(f"Estructura: {'CSV (imágenes planas)' if use_csv else 'Subcarpetas por clase'}")

    # Contar imágenes originales en cada carpeta
    if dataset_dir.exists():
        dataset_count_original = len(
            [f for f in dataset_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        )
    else:
        dataset_count_original = 0
    new_data_count = 0
    if new_data_dir.exists():
        new_data_subdirs_list = [d for d in new_data_dir.iterdir() if d.is_dir()]
        if new_data_subdirs_list:
            for d in new_data_subdirs_list:
                new_data_count += len([f for f in d.iterdir() if f.is_file()])
        else:
            new_data_count = len([f for f in new_data_dir.iterdir() if f.is_file()])

    print(f"\nImágenes en dataset/ (original): {dataset_count_original}")
    print(f"Imágenes en new_data/:           {new_data_count}")

    # ──────────────────────────────────────────────
    # 3. UNIFICACIÓN DE IMÁGENES (new_data → dataset)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. UNIFICACIÓN DE IMÁGENES (new_data → dataset)")
    print("=" * 60)

    copied = 0
    skipped = 0
    if new_data_dir.exists() and dataset_dir.exists():
        new_data_images = [
            f for f in new_data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        print(f"\nCopiando {len(new_data_images)} imágenes de new_data/ a dataset/...")
        for img in tqdm(new_data_images, desc="Copiando"):
            dest = dataset_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
                copied += 1
            else:
                skipped += 1
        print(f"Copiadas: {copied}  |  Ya existían (omitidas): {skipped}")
    else:
        print("No se encontraron ambos directorios dataset/ y new_data/.")

    if dataset_dir.exists():
        total_unified = len(
            [f for f in dataset_dir.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        )
    else:
        total_unified = 0
    print(f"Total imágenes unificadas en dataset/: {total_unified}")

    # ──────────────────────────────────────────────
    # 4. DISTRIBUCIÓN DE CLASES + MÉTRICAS DE BALANCE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. DISTRIBUCIÓN DE CLASES")
    print("=" * 60)

    class_counts = {}
    if use_csv and not df_dataset.empty:
        class_counts = df_dataset["label"].value_counts().sort_index().to_dict()
    elif dataset_dir.exists():
        for clase_dir in sorted(dataset_dir.iterdir()):
            if clase_dir.is_dir():
                n = len([f for f in clase_dir.iterdir() if f.is_file()])
                class_counts[clase_dir.name] = n

    total_labeled = sum(class_counts.values())
    counts_array = np.array(list(class_counts.values()))

    print(f"\nTotal imágenes etiquetadas: {total_labeled}")
    print(f"Total imágenes sin etiqueta (new_data): {new_data_count}")
    print(f"Total unificado en dataset/: {total_unified}\n")

    print(f"{'Clase':<25} {'Cantidad':>10} {'Proporción':>12} {'Barra':>5}")
    print("-" * 65)
    max_count = max(class_counts.values()) if class_counts else 1
    for cls, cnt in sorted(class_counts.items()):
        bar_str = "█" * int(cnt / max_count * 20)
        pct = cnt / total_labeled * 100
        print(f"{cls:<25} {cnt:>10} {pct:>11.2f}% {bar_str}")

    # Métricas de balance
    if len(counts_array) > 0:
        ratio_max_min = (
            counts_array.max() / counts_array.min()
            if counts_array.min() > 0 else float("inf")
        )
        std_counts = counts_array.std()
        cv = std_counts / counts_array.mean() if counts_array.mean() > 0 else 0
        is_balanced = ratio_max_min <= 1.5

        print("\n" + "─" * 40)
        print("  MÉTRICAS DE BALANCE")
        print("─" * 40)
        print(f"  Imágenes por clase (min): {counts_array.min()}")
        print(f"  Imágenes por clase (max): {counts_array.max()}")
        print(f"  Imágenes por clase (media): {counts_array.mean():.1f}")
        print(f"  Desviación estándar:  {std_counts:.2f}")
        print(f"  Coef. de variación:   {cv:.4f}")
        print(f"  Ratio max/min:        {ratio_max_min:.2f}")
        print(f"  Balance: {'✅ Balanceado' if is_balanced else '⚠️ Desbalanceado'}")

    # Gráfico mejorado: barras horizontales con valores
    _, ax = plt.subplots(figsize=(10, 8))
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes_sorted = [item[0] for item in sorted_items]
    counts_sorted = [item[1] for item in sorted_items]

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(classes_sorted)))
    bars = ax.barh(classes_sorted, counts_sorted, color=colors, edgecolor="white", height=0.7)

    for rect, cnt in zip(bars, counts_sorted):
        ax.text(rect.get_width() + 5, rect.get_y() + rect.get_height() / 2,
                f"{cnt} ({cnt / total_labeled * 100:.1f}%)",
                ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Cantidad de imágenes", fontsize=12)
    ax.set_title("Distribución de clases (imágenes etiquetadas)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts_sorted) * 1.25)

    # Línea de media
    mean_count = counts_array.mean()
    ax.axvline(
        mean_count, color="red", linestyle="--",
        linewidth=1.2, alpha=0.7, label=f"Media: {mean_count:.0f}"
    )
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "distribucion_clases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico guardado en: {output_dir / 'distribucion_clases.png'}")

    # ──────────────────────────────────────────────
    # 5. ANÁLISIS DE TAMAÑOS DE IMAGEN
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. ANÁLISIS DE TAMAÑOS DE IMAGEN")
    print("=" * 60)

    widths = []
    heights = []
    modes = []
    formats_list = []
    errores = 0

    # Recopilar todas las imágenes de dataset/ (ya unificado)
    all_images = []
    if dataset_dir.exists():
        if use_csv or not dataset_subdirs:
            for img_path in dataset_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    all_images.append(img_path)
        else:
            for clase_dir in sorted(dataset_dir.iterdir()):
                if clase_dir.is_dir():
                    for img_path in clase_dir.iterdir():
                        if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                            all_images.append(img_path)

    print(f"\nAnalizando {len(all_images)} imágenes...")

    for img_path in tqdm(all_images, desc="Leyendo imágenes"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                modes.append(img.mode)
                formats_list.append(img.format)
        except (OSError, SyntaxError):
            errores += 1

    print(f"\nImágenes leídas correctamente: {len(widths)}")
    print(f"Imágenes con error: {errores}")

    if widths:
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        print(f"\nAncho  -> min: {min(widths)}, max: {max(widths)},"
              f" promedio: {avg_w:.1f}")
        print(f"Alto   -> min: {min(heights)}, max: {max(heights)},"
              f" promedio: {avg_h:.1f}")

        size_counter = Counter(zip(widths, heights))
        print(f"\nResoluciones únicas: {len(size_counter)}")
        print("Top 10 resoluciones más frecuentes:")
        for (w, h), cnt in size_counter.most_common(10):
            print(f"  {w}x{h}: {cnt} imágenes ({cnt/len(widths)*100:.1f}%)")

        mode_counter = Counter(modes)
        print("\nModos de color:")
        for mode, cnt in mode_counter.most_common():
            print(f"  {mode}: {cnt} imágenes")

        format_counter = Counter(formats_list)
        print("\nFormatos de archivo:")
        for fmt, cnt in format_counter.most_common():
            print(f"  {fmt}: {cnt} imágenes")

        # Gráfico de distribución de tamaños
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(widths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].set_title("Distribución de anchos")
        axes[0].set_xlabel("Ancho (px)")
        axes[0].set_ylabel("Frecuencia")
        axes[1].hist(heights, bins=50, color="coral", edgecolor="black", alpha=0.7)
        axes[1].set_title("Distribución de altos")
        axes[1].set_xlabel("Alto (px)")
        axes[1].set_ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(output_dir / "distribucion_tamanos.png", dpi=150)
        plt.close()
        print(f"\nGráfico guardado en: {output_dir / 'distribucion_tamanos.png'}")

        # Scatter de ancho vs alto
        _fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(widths, heights, alpha=0.15, s=5, color="steelblue")
        ax.set_xlabel("Ancho (px)")
        ax.set_ylabel("Alto (px)")
        ax.set_title("Ancho vs Alto de las imágenes")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_ancho_alto.png", dpi=150)
        plt.close()
        print(f"Gráfico guardado en: {output_dir / 'scatter_ancho_alto.png'}")

    # ──────────────────────────────────────────────
    # 6. ESTADÍSTICAS DE CANALES (RGB)
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. ESTADÍSTICAS DE CANALES (RGB)")
    print("=" * 60)

    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sq_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    sample_size = min(2000, len(all_images))

    if sample_size == 0:
        print("\nNo hay imágenes para calcular estadísticas de canales.")
        sample_indices = np.array([], dtype=int)
    else:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(all_images), size=sample_size, replace=False)

    print(f"\nCalculando media y desviación estándar sobre {sample_size} imágenes...")

    for idx in tqdm(sample_indices, desc="Estadísticas de canales"):
        img_path = all_images[idx]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                arr = np.array(img_rgb, dtype=np.float64) / 255.0
                npixels = arr.shape[0] * arr.shape[1]
                channel_sums += arr.reshape(-1, 3).sum(axis=0)
                channel_sq_sums += (arr.reshape(-1, 3) ** 2).sum(axis=0)
                pixel_count += npixels
        except (OSError, SyntaxError):
            pass

    if pixel_count > 0:
        mean_rgb = channel_sums / pixel_count
        std_rgb = np.sqrt(channel_sq_sums / pixel_count - mean_rgb ** 2)
        print(f"\nMedia  (R, G, B): ({mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f})")
        print(f"Std    (R, G, B): ({std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f})")

    # ──────────────────────────────────────────────
    # 7. VISUALIZACIÓN DE MUESTRAS POR CLASE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. VISUALIZACIÓN DE MUESTRAS POR CLASE")
    print("=" * 60)

    n_samples = 3

    if clases:
        class_to_images = {c: [] for c in clases}
        if use_csv and not df_dataset.empty:
            for _, row in df_dataset.iterrows():
                label = row["label"]
                img_path = dataset_dir / row["filename"]
                if label in class_to_images and img_path.exists():
                    class_to_images[label].append(img_path)
        else:
            for clase in clases:
                clase_path = dataset_dir / clase
                if clase_path.is_dir():
                    class_to_images[clase] = sorted(
                        [f for f in clase_path.iterdir()
                         if f.suffix.lower() in IMAGE_EXTENSIONS]
                    )

        _, axes = plt.subplots(len(clases), n_samples, figsize=(n_samples * 4, len(clases) * 3))
        if len(clases) == 1:
            axes = [axes]

        for i, clase in enumerate(clases):
            imgs = class_to_images[clase][:n_samples]
            for j in range(n_samples):
                ax = axes[i][j]
                if j < len(imgs):
                    img = mpimg.imread(str(imgs[j]))
                    ax.imshow(img)
                    if j == 0:
                        ax.set_ylabel(clase, fontsize=10, rotation=0, labelpad=80, va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle("Muestras de cada clase (dataset unificado)", fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "muestras_por_clase.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Gráfico guardado en: {output_dir / 'muestras_por_clase.png'}")
    else:
        print("No se encontraron clases. No se genera la visualización.")

    # ──────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET")
    print("=" * 60)
    print("  Nombre:             Human Action Recognition (HAR) Dataset")
    print("  Fuente:             kaggle.com/meetnagadia/human-action-recognition-har-dataset")
    print(f"  Clases:             {len(clases)}")
    print(f"  Imágenes dataset/:  {dataset_count_original}")
    print(f"  Imágenes new_data/: {new_data_count}")
    print(f"  Total unificado:    {total_unified}")
    print("  Resoluciones:       Variadas (ver análisis arriba)")
    if pixel_count > 0:
        print(f"  Media RGB:          ({mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f})")
        print(f"  Std RGB:            ({std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f})")
    if len(counts_array) > 0:
        bal = 'Balanceado' if is_balanced else 'Desbalanceado'
        print(f"  Balance:            {bal}"
              f" (ratio max/min: {ratio_max_min:.2f})")
    print(f"\nGráficos exportados en: {output_dir}")
    print("=" * 60)
    print("Exploración completada.")


if __name__ == "__main__":
    main()
