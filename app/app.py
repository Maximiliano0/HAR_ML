"""
app.py
======
Aplicación de clasificación de actividades humanas (HAR) con GUI Tkinter.

Flujo:
1. El usuario selecciona una imagen desde el explorador de archivos.
2. Se aplica el pipeline de preprocesamiento (resize + CLAHE + normalización ImageNet).
3. Se clasifica con el modelo ConvNeXt-Base entrenado (har_cnn_best.pth o har_cnn_swa.pth).
4. Se muestra la imagen original y las probabilidades de las 15 clases con barras de porcentaje.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import transforms, models

# ───────────────────────────────────────────────────────
# Configuración
# ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "data_train" / "output"

MODEL_PATH = MODEL_DIR / "har_cnn_swa.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / "har_cnn_best.pth"

IMG_H, IMG_W = 384, 288
TARGET_W, TARGET_H = 288, 384

CLASSES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "listening_to_music",
    "running", "sitting", "sleeping", "texting", "using_laptop",
]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_EMOJIS = {
    "calling": "📞", "clapping": "👏", "cycling": "🚴", "dancing": "💃",
    "drinking": "🥤", "eating": "🍽️", "fighting": "🤼", "hugging": "🤗",
    "laughing": "😂", "listening_to_music": "🎧", "running": "🏃",
    "sitting": "🪑", "sleeping": "😴", "texting": "📱", "using_laptop": "💻",
}

# ── Paleta suave (warm neutral) ──
BG = "#faf8f5"
BG_HEADER = "#3b3a52"
BG_CARD = "#ffffff"
ACCENT = "#6c63ff"
ACCENT_LIGHT = "#a29bfe"
GREEN = "#00b894"
TEXT = "#2d3436"
TEXT_SEC = "#636e72"
TEXT_LIGHT = "#b2bec3"
BAR_TRACK = "#dfe6e9"
BAR_TOP = "#6c63ff"
BAR_REST = "#a29bfe"
HIGHLIGHT_BG = "#f0edff"
BORDER = "#dfe6e9"


# ───────────────────────────────────────────────────────
# Pipeline de preprocesamiento (igual que data_adecuate.py)
# ───────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Lee imagen, aplica resize + CLAHE. Devuelve (img_clahe_rgb, img_original_rgb)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {img_path}")

    # Imagen original en RGB para display
    img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h_orig, w_orig = img_bgr.shape[:2]
    needs_upscale = (w_orig < TARGET_W) or (h_orig < TARGET_H)
    interp = cv2.INTER_CUBIC if needs_upscale else cv2.INTER_AREA
    img_resized = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    l_ch = clahe.apply(l_ch)
    img_lab = cv2.merge([l_ch, a_ch, b_ch])
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    return img_rgb, img_original_rgb


# ───────────────────────────────────────────────────────
# Modelo
# ───────────────────────────────────────────────────────
def build_model() -> nn.Module:
    """Construye ConvNeXt-Base con el head custom y carga los pesos entrenados."""
    model = models.convnext_base(weights=None)
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(1024),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}.\n"
            "Ejecutá primero el entrenamiento (train_cnn.py o el notebook de Colab)."
        )

    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ("_orig_mod.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = v
    model.load_state_dict(cleaned)
    model.eval()
    return model


def classify(model: nn.Module, img_rgb: np.ndarray) -> list[tuple[str, float]]:
    """Clasifica una imagen RGB preprocesada. Devuelve las 15 clases ordenadas por probabilidad."""
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = img_transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    sorted_idx = probs.argsort(descending=True)
    results = []
    for idx in sorted_idx.tolist():
        results.append((CLASSES[idx], probs[idx].item()))
    return results


# ───────────────────────────────────────────────────────
# GUI Tkinter
# ───────────────────────────────────────────────────────
def _configure_styles():
    """Configura los estilos ttk para las barras de progreso."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Top.Horizontal.TProgressbar",
                     troughcolor=BAR_TRACK, background=BAR_TOP,
                     thickness=20, borderwidth=0)
    style.configure("Rest.Horizontal.TProgressbar",
                     troughcolor=BAR_TRACK, background=BAR_REST,
                     thickness=18, borderwidth=0)


class HARApp:
    """Interfaz gráfica para clasificación de actividades humanas."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HAR — Clasificador de Actividades Humanas")
        self.root.geometry("1120x760")
        self.root.minsize(950, 680)
        self.root.configure(bg=BG)

        self.model = None
        self._photo = None
        _configure_styles()
        self._build_ui()
        self._load_model()

    def _build_ui(self):
        """Construye los widgets de la interfaz."""
        # ── Header ──
        header = tk.Frame(self.root, bg=BG_HEADER, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="🏃  Human Activity Recognition",
            font=("Segoe UI", 16, "bold"), fg="#ffffff", bg=BG_HEADER,
        ).pack(side=tk.LEFT, padx=24, pady=10)
        self.model_label = tk.Label(
            header, text="",
            font=("Segoe UI", 9), fg="#b2bec3", bg=BG_HEADER,
        )
        self.model_label.pack(side=tk.RIGHT, padx=24)

        # ── Toolbar ──
        toolbar = tk.Frame(self.root, bg=BG)
        toolbar.pack(fill=tk.X, padx=24, pady=(14, 0))
        self.btn_load = tk.Button(
            toolbar, text="📂  Seleccionar imagen",
            font=("Segoe UI", 11), padx=20, pady=6,
            command=self._on_load_image, bg=ACCENT, fg="#ffffff",
            activebackground=ACCENT_LIGHT, relief=tk.FLAT, cursor="hand2",
        )
        self.btn_load.pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value="")
        tk.Label(
            toolbar, textvariable=self.path_var,
            font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG, anchor="w",
        ).pack(side=tk.LEFT, padx=(16, 0), fill=tk.X, expand=True)

        # ── Contenido principal ──
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=14)

        # -- Panel izquierdo: imagen original + predicción --
        left = tk.Frame(body, bg=BG, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        left.pack_propagate(False)

        self.img_frame = tk.Frame(left, bg=BG_CARD, bd=0,
                                  highlightthickness=1, highlightbackground=BORDER)
        self.img_frame.pack(fill=tk.X)
        self.img_label = tk.Label(
            self.img_frame, text="Seleccioná una imagen\npara clasificar",
            font=("Segoe UI", 10), fg=TEXT_LIGHT, bg=BG_CARD,
            height=20,
        )
        self.img_label.pack(padx=6, pady=6, fill=tk.X)

        # Resultado principal
        result_card = tk.Frame(left, bg=HIGHLIGHT_BG, bd=0,
                               highlightthickness=1, highlightbackground=ACCENT_LIGHT)
        result_card.pack(fill=tk.X, pady=(12, 0))

        self.result_var = tk.StringVar(value="")
        tk.Label(
            result_card, textvariable=self.result_var,
            font=("Segoe UI", 17, "bold"), fg=ACCENT, bg=HIGHLIGHT_BG,
        ).pack(pady=(10, 2))

        self.confidence_var = tk.StringVar(value="")
        tk.Label(
            result_card, textvariable=self.confidence_var,
            font=("Segoe UI", 11), fg=GREEN, bg=HIGHLIGHT_BG,
        ).pack(pady=(0, 10))

        # -- Panel derecho: barras de progreso de las 15 clases --
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right, text="Probabilidades por clase",
            font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG, anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))

        # Frame scrollable para las 15 barras
        bars_outer = tk.Frame(right, bg=BG)
        bars_outer.pack(fill=tk.BOTH, expand=True)

        self.bars_canvas = tk.Canvas(bars_outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(bars_outer, orient=tk.VERTICAL,
                                  command=self.bars_canvas.yview)
        self.bars_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bars_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.bars_inner = tk.Frame(self.bars_canvas, bg=BG)
        self.bars_canvas.create_window((0, 0), window=self.bars_inner, anchor="nw")
        self.bars_inner.bind(
            "<Configure>",
            lambda e: self.bars_canvas.configure(
                scrollregion=self.bars_canvas.bbox("all")))

        self.prob_rows: list[tuple[tk.Label, ttk.Progressbar, tk.Label]] = []
        for _ in range(NUM_CLASSES):
            row = tk.Frame(self.bars_inner, bg=BG)
            row.pack(fill=tk.X, pady=3)

            lbl_name = tk.Label(
                row, text="", font=("Segoe UI", 9), fg=TEXT_SEC,
                bg=BG, width=22, anchor="e",
            )
            lbl_name.pack(side=tk.LEFT)

            progress = ttk.Progressbar(
                row, orient=tk.HORIZONTAL, length=100, mode="determinate",
                maximum=100, style="Rest.Horizontal.TProgressbar",
            )
            progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

            lbl_pct = tk.Label(
                row, text="", font=("Segoe UI", 9), fg=TEXT_SEC,
                bg=BG, width=7, anchor="w",
            )
            lbl_pct.pack(side=tk.LEFT, padx=(8, 0))

            self.prob_rows.append((lbl_name, progress, lbl_pct))

        # ── Status bar ──
        status_bar = tk.Frame(self.root, bg=BG_HEADER, height=26)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Cargando modelo...")
        tk.Label(
            status_bar, textvariable=self.status_var,
            font=("Segoe UI", 8), fg="#b2bec3", bg=BG_HEADER, anchor="w",
        ).pack(fill=tk.X, padx=14, pady=4)

    def _load_model(self):
        """Carga el modelo."""
        try:
            self.model = build_model()
            name = MODEL_PATH.name
            self.model_label.configure(text=f"ConvNeXt-Base  ·  {name}")
            self.status_var.set(f"Listo — modelo cargado ({name})")
        except FileNotFoundError as e:
            self.status_var.set("Error: modelo no encontrado")
            messagebox.showerror("Error", str(e))

    def _on_load_image(self):
        """Callback del botón de carga de imagen."""
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Todos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes,
            initialdir=str(BASE_DIR / "datos_har"),
        )
        if not path:
            return

        self.path_var.set(Path(path).name)
        self.status_var.set("Procesando imagen...")
        self.root.update_idletasks()

        try:
            img_clahe, img_original = preprocess_image(path)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error al leer la imagen")
            return

        # Mostrar imagen ORIGINAL (no la procesada)
        pil_img = Image.fromarray(img_original)
        display_h = 340
        ratio = display_h / pil_img.height
        display_w = int(pil_img.width * ratio)
        if display_w > 308:
            display_w = 308
            display_h = int(pil_img.height * (308 / pil_img.width))
        pil_display = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_display)
        self.img_label.configure(image=self._photo, text="",
                                 width=display_w, height=display_h)

        if self.model is None:
            messagebox.showwarning("Atención", "El modelo no está cargado.")
            return

        # Clasificar usando la imagen preprocesada (CLAHE)
        results = classify(self.model, img_clahe)
        top_class, top_prob = results[0]
        emoji = CLASS_EMOJIS.get(top_class, "")

        self.result_var.set(f"{emoji}  {top_class.replace('_', ' ').title()}")
        self.confidence_var.set(f"Confianza: {top_prob:.1%}")

        # Actualizar las 15 barras de progreso
        for i, (cls, prob) in enumerate(results):
            lbl_name, progress, lbl_pct = self.prob_rows[i]
            emoji_i = CLASS_EMOJIS.get(cls, "")
            is_top = i == 0

            lbl_name.configure(
                text=f"{emoji_i} {cls.replace('_', ' ').title()}",
                fg=ACCENT if is_top else TEXT,
                font=("Segoe UI", 9, "bold") if is_top else ("Segoe UI", 9),
            )

            progress.configure(
                value=prob * 100,
                style="Top.Horizontal.TProgressbar" if is_top
                else "Rest.Horizontal.TProgressbar",
            )

            lbl_pct.configure(
                text=f"{prob:.1%}",
                fg=ACCENT if is_top else TEXT_SEC,
                font=("Segoe UI", 9, "bold") if is_top else ("Segoe UI", 9),
            )

        self.status_var.set(
            f"Clasificación completada — {top_class.replace('_', ' ').title()} ({top_prob:.1%})"
        )


# ───────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app_root = tk.Tk()
    HARApp(app_root)
    app_root.mainloop()
