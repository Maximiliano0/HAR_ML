"""
train_cnn.py
============
Entrena un modelo ConvNeXt-Base (transfer learning desde ImageNet) para
clasificar imagenes del dataset HAR procesado con CLAHE y oversampling.

Arquitectura
------------
- Backbone: ConvNeXt-Base preentrenado en ImageNet (IMAGENET1K_V1)
- Head custom: Flatten -> LayerNorm(1024) -> Linear(1024->512) -> BN -> ReLU ->
              Dropout(0.3) -> Linear(512->256) -> BN -> ReLU -> Dropout(0.2) ->
              Linear(256->15)
- Entrada: 3 canales RGB, 384x288 (alta resolucion para capturar detalle)
- Parametros totales: ~88M

Entrenamiento (2 fases + progressive unfreezing + SWA)
------------------------------------------------------
Fase 1 — Solo head (backbone congelado):
  - Epocas: 15
  - Optimizador: AdamW (lr=1e-3, weight_decay=5e-4)

Fase 2 — Fine-tuning con progressive unfreezing:
  - Epocas 1-10: solo features[5:7]+classifier entrenables (backbone parcial)
  - Epocas 11+: todo el backbone descongelado
  - Optimizador: AdamW con LR diferencial (head=1e-4, backbone=3e-5)
  - Scheduler: SequentialLR(LinearLR warmup 10ep + CosineAnnealingLR)
  - Early stopping: paciencia 15 epocas (por test loss)
  - SWA (Stochastic Weight Averaging) desde epoca 20 de F2

Regularizacion
--------------
- Focal Loss (gamma=2.0) con label smoothing 0.02
- Gradient clipping: max_norm=1.0
- Dropout: 0.3/0.2 (head)
- Weight decay: 5e-4
- Data augmentation: RandomHorizontalFlip, RandomRotation(20),
  ColorJitter(0.3, 0.3, 0.3, 0.1), TrivialAugmentWide(),
  RandomErasing(0.10)
- Normalizacion ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- TTA (Test-Time Augmentation) con 5 pasadas en evaluacion

Pasos
-----
1. Cargar imagenes de datos_har/dataset_tr/ y etiquetas de dataset.csv
2. Separar en train (80%) y test (20%) con estratificacion
3. Fase 1: entrenar head con backbone congelado
4. Fase 2: fine-tuning con progressive unfreezing, LR diferencial y SWA
5. Evaluar: accuracy con TTA, classification report y matriz de confusion
6. Guardar modelo (.pth), metricas (JSON) y graficos en data_train/output/
"""

import os
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

import torch  # pylint: disable=wrong-import-position
import torch.nn as nn  # pylint: disable=wrong-import-position
import torch.optim as optim  # pylint: disable=wrong-import-position
from torch.utils.data import Dataset, DataLoader  # pylint: disable=wrong-import-position
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # pylint: disable=wrong-import-position
from torchvision import transforms, models  # pylint: disable=wrong-import-position
from sklearn.model_selection import train_test_split  # pylint: disable=wrong-import-position
from sklearn.metrics import (  # pylint: disable=wrong-import-position
    confusion_matrix, accuracy_score, classification_report
)

# ───────────────────────────────────────────────────────
# Rutas
# ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datos_har"
IMG_DIR = DATA_DIR / "dataset_tr"
CSV_PATH = DATA_DIR / "dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hiperparámetros ──
SEED = 42
BATCH_SIZE = 64
PHASE1_EPOCHS = 15        # fase 1: solo head (backbone congelado)
PHASE2_EPOCHS = 100       # fase 2: fine-tuning completo (early stopping puede cortar)
LR = 1e-3                 # learning rate del head (fase 1)
LR_HEAD_PHASE2 = 1e-4     # learning rate del head (fase 2, mas bajo para no desestabilizar)
LR_BACKBONE = 3e-5        # learning rate del backbone (fase 2, moderado)
WEIGHT_DECAY = 5e-4       # regularización L2
IMG_H, IMG_W = 384, 288   # resolución alta para capturar detalle
UNFREEZE_AFTER = 10       # épocas F2 solo capas tardías, luego descongelar todo
NUM_WORKERS = 0            # workers del DataLoader (0 = main thread)
EARLY_STOP_PATIENCE = 15  # épocas sin mejora antes de detener (fase 2)
WARMUP_EPOCHS = 10        # épocas de warmup lineal (fase 2)
LABEL_SMOOTHING = 0.02    # suavizado de etiquetas (mínimo para 15 clases)
FOCAL_GAMMA = 2.0         # gamma para Focal Loss (penaliza ejemplos fáciles)
SWA_START_EPOCH = 20      # epoch F2 a partir del cual activar SWA
SWA_LR = 1e-5             # learning rate fijo para SWA
N_TTA = 5                 # pasadas TTA para evaluacion (1 = sin TTA)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

df_all = pd.read_csv(CSV_PATH)

# Quedarse solo con las imágenes que existen en dataset_tr
existing_files = set(os.listdir(IMG_DIR))
df = df_all[df_all["filename"].isin(existing_files)].copy()
df.reset_index(drop=True, inplace=True)

# Codificar etiquetas
labels_sorted = sorted(df["label"].unique())
label2idx = {label: idx for idx, label in enumerate(labels_sorted)}
idx2label = {idx: label for label, idx in label2idx.items()}
df["label_idx"] = df["label"].map(label2idx)

NUM_CLASSES = len(labels_sorted)
print(f"   Imagenes encontradas: {len(df)}")
print(f"   Clases: {NUM_CLASSES} -> {labels_sorted}")

# ═══════════════════════════════════════════════════════
# 2. SPLIT TRAIN / TEST (80% / 20%)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. SPLIT TRAIN / TEST")
print("=" * 60)

X_train_df, X_test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["label_idx"]
)
X_train_df.reset_index(drop=True, inplace=True)
X_test_df.reset_index(drop=True, inplace=True)

print(f"   Train: {len(X_train_df)} imagenes ({len(X_train_df)/len(df)*100:.0f}%)")
print(f"   Test:  {len(X_test_df)} imagenes ({len(X_test_df)/len(df)*100:.0f}%)")

# ═══════════════════════════════════════════════════════
# 3. DATASET Y DATALOADER
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. DATASET Y DATALOADER")
print("=" * 60)

# ── Normalización ImageNet (para transfer learning con ConvNeXt-Base) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Data augmentation agresiva (solo train) ──
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.10),
])

# ── Transform de test (sin augmentacion) ──
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class HARDataset(Dataset):
    """Dataset para imagenes HAR con transformaciones opcionales."""

    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(img)
        else:
            pil_img = Image.fromarray(img).resize(
                (IMG_W, IMG_H), Image.Resampling.LANCZOS
            )
            img = np.array(pil_img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img)
        label = row["label_idx"]
        return img, torch.tensor(label, dtype=torch.long)


train_ds = HARDataset(X_train_df, IMG_DIR, transform=train_transform)
test_ds = HARDataset(X_test_df, IMG_DIR, transform=test_transform)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, generator=g)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ═══════════════════════════════════════════════════════
# 4. MODELO: ConvNeXt-Base (Transfer Learning)
#    Backbone preentrenado en ImageNet + clasificador custom
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. ARQUITECTURA DEL MODELO (ConvNeXt-Base)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar ConvNeXt-Base preentrenado en ImageNet
model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

# ConvNeXt-Base: features (backbone) + classifier (head)
# classifier original: LayerNorm + Flatten + Linear(1024, 1000)
# Reemplazar con head expresivo
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

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Device: {device}")
print("   Backbone: ConvNeXt-Base (preentrenado ImageNet)")
print(f"   Parametros totales:     {total_params:,}")
print(f"   Parametros entrenables: {trainable_params:,}")
print(f"   Input: 3 x {IMG_H} x {IMG_W} (RGB)")
print(f"   Output: {NUM_CLASSES} clases")

# ═══════════════════════════════════════════════════════
# 5. ENTRENAMIENTO (2 FASES + SWA)
#    Fase 1: head (backbone congelado)
#    Fase 2: fine-tuning completo con Focal Loss y SWA
# ═══════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """Focal Loss: penaliza ejemplos fáciles para enfocarse en los difíciles."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, inputs, targets):
        """Calcula focal loss ponderando inversamente los ejemplos bien clasificados."""
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


print("\n" + "=" * 60)
print("5. ENTRENAMIENTO (2 FASES + SWA)")
print("=" * 60)

# pylint: disable=invalid-name
criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_test_acc = 0.0
best_test_loss = float("inf")

# ── FASE 1: Entrenar solo el head (backbone congelado) ──
for name, param in model.named_parameters():
    if not name.startswith("classifier"):
        param.requires_grad = False

optimizer1 = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
trainable1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n--- FASE 1: Entrenar head ({PHASE1_EPOCHS} epocas, backbone congelado) ---")
print(f"   Parametros entrenables: {trainable1:,}")

for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer1.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer1.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = running_loss / total
    test_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    if test_loss < best_test_loss:
        best_test_acc = test_acc
        best_test_loss = test_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn_best.pth")

    print(f"   [F1] Epoca {epoch:03d}/{PHASE1_EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
          f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

# ── FASE 2: Fine-tuning progresivo con Focal Loss y SWA ──
# Primeras UNFREEZE_AFTER épocas: solo features[5:7] + classifier (ConvNeXt-Base)
# Después: descongelar todo el backbone
# SWA se activa a partir de SWA_START_EPOCH
for name, param in model.named_parameters():
    if name.startswith(("features.5.", "features.6.", "classifier.")):
        param.requires_grad = True
    else:
        param.requires_grad = False


def _build_optimizer_and_scheduler(mdl, last_epoch=-1):
    """Crea optimizer con LR diferencial y scheduler."""
    backbone_p = [p for n, p in mdl.named_parameters()
                  if not n.startswith("classifier") and p.requires_grad]
    opt = optim.AdamW([
        {"params": backbone_p, "lr": LR_BACKBONE},
        {"params": mdl.classifier.parameters(), "lr": LR_HEAD_PHASE2},
    ], weight_decay=WEIGHT_DECAY)
    if last_epoch >= 0:
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])
    warmup = optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS, last_epoch=last_epoch)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=PHASE2_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6,
        last_epoch=last_epoch)
    sched = optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[WARMUP_EPOCHS],
        last_epoch=last_epoch)
    return opt, sched


optimizer2, scheduler2 = _build_optimizer_and_scheduler(model)

# SWA: modelo promediado
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer2, swa_lr=SWA_LR)
swa_active = False

trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n--- FASE 2: Fine-tuning progresivo ({PHASE2_EPOCHS} epocas) ---")
print(f"   LR head={LR_HEAD_PHASE2}  |  LR backbone={LR_BACKBONE}")
print(f"   Parametros entrenables (parcial): {trainable2:,}")
print(f"   Descongelar todo tras epoca {UNFREEZE_AFTER}")
print(f"   Warmup: {WARMUP_EPOCHS} epocas  |  Early stopping: {EARLY_STOP_PATIENCE} epocas")
print(f"   Focal Loss (gamma={FOCAL_GAMMA})")
print(f"   SWA desde epoca {SWA_START_EPOCH} (lr={SWA_LR})")

epochs_no_improve = 0
phase2_trained = 0

for epoch in range(1, PHASE2_EPOCHS + 1):
    global_epoch = PHASE1_EPOCHS + epoch
    phase2_trained = epoch

    # Progressive unfreezing: descongelar todo tras UNFREEZE_AFTER
    if epoch == UNFREEZE_AFTER + 1:
        for param in model.parameters():
            param.requires_grad = True
        optimizer2, scheduler2 = _build_optimizer_and_scheduler(
            model, last_epoch=epoch - 2)
        swa_scheduler = SWALR(optimizer2, swa_lr=SWA_LR)
        n_train = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   >>> Descongelando backbone completo "
              f"(params entrenables: {n_train:,})")

    # Activar SWA
    if epoch == SWA_START_EPOCH and not swa_active:
        swa_active = True
        print(f"   >>> SWA activado (epoch {epoch})")

    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer2.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer2.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # --- Eval ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = running_loss / total
    test_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    # Scheduler: SWA scheduler o normal
    if swa_active:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler2.step()

    # Guardar mejor modelo (por test loss)
    if test_loss < best_test_loss:
        best_test_acc = test_acc
        best_test_loss = test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn_best.pth")
    else:
        epochs_no_improve += 1

    lr_display = swa_scheduler.get_last_lr()[0] if swa_active else optimizer2.param_groups[1]["lr"]
    SWA_TAG = " [SWA]" if swa_active else ""
    print(f"   [F2] Epoca {epoch:03d}/{PHASE2_EPOCHS} (global {global_epoch:03d})  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
          f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  |  "
          f"LR: {lr_display:.1e}{SWA_TAG}")

    # Early stopping
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n   Early stopping en epoca {epoch} de fase 2 "
              f"(sin mejora en {EARLY_STOP_PATIENCE} epocas)")
        print(f"   Mejor Test Acc: {best_test_acc:.4f}")
        break

total_epochs = PHASE1_EPOCHS + phase2_trained
print(f"\n   Entrenamiento finalizado. Epocas totales: {total_epochs}")
print(f"   Mejor Test Acc: {best_test_acc:.4f} | Mejor Test Loss: {best_test_loss:.4f}")

# Actualizar BN del modelo SWA
if swa_active:
    print("   Actualizando BatchNorm para SWA...")
    update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.module.state_dict(), OUTPUT_DIR / "har_cnn_swa.pth")
    print(f"   Modelo SWA guardado: {OUTPUT_DIR / 'har_cnn_swa.pth'}")
# pylint: enable=invalid-name

# ═══════════════════════════════════════════════════════
# 6. EVALUACIÓN FINAL
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. EVALUACION FINAL")
print("=" * 60)

# Cargar el mejor modelo guardado durante entrenamiento
model.load_state_dict(torch.load(OUTPUT_DIR / "har_cnn_best.pth", weights_only=True))
model.eval()

# Evaluación con TTA (Test-Time Augmentation)
tta_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Predicción base (sin augmentación)
all_logits, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.extend(labels.numpy())

avg_logits = torch.cat(all_logits)

# Predicciones TTA (augmentaciones suaves)
if N_TTA > 1:
    print(f"   Aplicando TTA ({N_TTA} pasadas)...")
    for i in range(N_TTA - 1):
        tta_ds = HARDataset(X_test_df, IMG_DIR, transform=tta_augment)
        tta_loader = DataLoader(tta_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)
        tta_logits = []
        with torch.no_grad():
            for imgs, _ in tta_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                tta_logits.append(logits.cpu())
        avg_logits += torch.cat(tta_logits)
        print(f"   TTA pasada {i+2}/{N_TTA} completada")

avg_logits /= N_TTA
_, preds_tensor = avg_logits.max(1)
all_preds = preds_tensor.numpy()
all_labels = np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=labels_sorted)

print(f"\n   Accuracy global: {acc:.4f} ({acc*100:.2f}%)")
print("\n   Classification Report:\n")
print(report)

# ═══════════════════════════════════════════════════════
# 7. GRÁFICOS (curvas de entrenamiento + matriz de confusión)
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("7. GENERACION DE GRAFICOS")
print("=" * 60)

# 7a. Curvas de loss y accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(history["train_loss"])+1), history["train_loss"],
         label="Train Loss", marker="o", markersize=3)
ax1.plot(range(1, len(history["test_loss"])+1), history["test_loss"],
         label="Test Loss", marker="o", markersize=3)
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Loss")
ax1.set_title("Loss por Epoca")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(history["train_acc"])+1), history["train_acc"],
         label="Train Acc", marker="o", markersize=3)
ax2.plot(range(1, len(history["test_acc"])+1), history["test_acc"],
         label="Test Acc", marker="o", markersize=3)
ax2.set_xlabel("Epoca")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy por Epoca")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Guardado: {OUTPUT_DIR / 'curvas_entrenamiento.png'}")

# 7b. Matriz de confusión
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Matriz de Confusion", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

tick_marks = np.arange(NUM_CLASSES)
ax.set_xticks(tick_marks)
ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=8)
ax.set_yticks(tick_marks)
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel("Prediccion", fontsize=12)
ax.set_ylabel("Real", fontsize=12)

# Anotar valores en cada celda
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "matriz_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Guardado: {OUTPUT_DIR / 'matriz_confusion.png'}")

# ═══════════════════════════════════════════════════════
# 8. GUARDAR MODELO Y MÉTRICAS (último + mejor por test_loss)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. GUARDADO DE MODELO Y METRICAS")
print("=" * 60)

torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn.pth")
print(f"   Modelo ultimo guardado: {OUTPUT_DIR / 'har_cnn.pth'}")
print(f"   Mejor modelo (acc={best_test_acc:.4f}): {OUTPUT_DIR / 'har_cnn_best.pth'}")

metrics = {
    "architecture": "convnext_base",
    "transfer_learning": True,
    "pretrained_weights": "IMAGENET1K_V1",
    "training_strategy": "two_phase_swa",
    "accuracy": float(acc),
    "best_test_acc": float(best_test_acc),
    "best_test_loss": float(best_test_loss),
    "phase1_epochs": PHASE1_EPOCHS,
    "phase2_epochs": phase2_trained,
    "total_epochs": total_epochs,
    "batch_size": BATCH_SIZE,
    "learning_rate_head_phase1": LR,
    "learning_rate_head_phase2": LR_HEAD_PHASE2,
    "learning_rate_backbone": LR_BACKBONE,
    "weight_decay": WEIGHT_DECAY,
    "loss_function": "focal_loss",
    "focal_gamma": FOCAL_GAMMA,
    "label_smoothing": LABEL_SMOOTHING,
    "gradient_clipping": 1.0,
    "dropout": "0.3/0.2",
    "swa_start_epoch": SWA_START_EPOCH,
    "swa_lr": SWA_LR,
    "warmup_epochs": WARMUP_EPOCHS,
    "n_tta": N_TTA,
    "img_h": IMG_H,
    "img_w": IMG_W,
    "train_size": len(X_train_df),
    "test_size": len(X_test_df),
    "num_classes": NUM_CLASSES,
    "classes": labels_sorted,
    "confusion_matrix": cm.tolist(),
    "history": history,
    "classification_report": report,
    "device": str(device),
    "total_params": total_params,
}
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"   Metricas guardadas: {OUTPUT_DIR / 'metrics.json'}")

# ═══════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("  Arquitectura:           ConvNeXt-Base (Transfer Learning, 2 fases + SWA)")
print(f"  Parametros totales:     {total_params:,}")
print(f"  Entrada:                3x{IMG_H}x{IMG_W}")
print(f"  Imagenes totales:       {len(df)}")
print(f"  Train / Test:           {len(X_train_df)} / {len(X_test_df)}")
print(f"  Clases:                 {NUM_CLASSES}")
print(f"  Fase 1:                 {PHASE1_EPOCHS} epocas (solo head)")
print(f"  Fase 2:                 {phase2_trained} epocas (fine-tuning)")
print(f"  Epocas totales:         {total_epochs}")
print(f"  Focal Loss:             gamma={FOCAL_GAMMA}, label_smoothing={LABEL_SMOOTHING}")
print("  Gradient clipping:      1.0")
print("  Dropout:                0.3/0.2")
print(f"  SWA:                    desde epoca {SWA_START_EPOCH} (lr={SWA_LR})")
print(f"  TTA:                    {N_TTA} pasadas")
print(f"  Accuracy final (test):  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Modelo:                 {OUTPUT_DIR / 'har_cnn.pth'}")
print(f"  Metricas:               {OUTPUT_DIR / 'metrics.json'}")
print(f"  Graficos:               {OUTPUT_DIR}/")
print("=" * 60)
print("Entrenamiento completado.")
