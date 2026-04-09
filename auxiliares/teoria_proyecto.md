<div align="center">

# 📚 Fundamentos Teóricos del Proyecto HAR

### Human Action Recognition — Clasificación de Imágenes con Deep Learning

[![Transfer Learning](https://img.shields.io/badge/Transfer_Learning-ConvNeXt--Base-blue)]()
[![Loss](https://img.shields.io/badge/Loss-Focal_Loss-red)]()
[![SWA](https://img.shields.io/badge/SWA-Stochastic_Weight_Averaging-green)]()
[![TTA](https://img.shields.io/badge/TTA-Test--Time_Augmentation-orange)]()

</div>

---

## 📑 Índice

| # | Sección | Descripción |
|:-:|---------|-------------|
| 1 | [Introducción](#1-introducción) | Contexto y objetivo del proyecto |
| 2 | [Exploración de Datos](#2-exploración-de-datos-data_prep) | Análisis exploratorio y estadísticas |
| 3 | [Pipeline de Preprocesamiento](#3-pipeline-de-preprocesamiento-data_trans) | CLAHE, resize, oversampling |
| 4 | [Arquitectura del Modelo](#4-arquitectura-del-modelo) | ConvNeXt-Base y head custom |
| 5 | [Transfer Learning](#5-transfer-learning) | Fundamentos y aplicación |
| 6 | [Progressive Unfreezing](#6-progressive-unfreezing) | Descongelamiento gradual |
| 7 | [Función de Pérdida: Focal Loss](#7-función-de-pérdida-focal-loss) | Penalización de ejemplos fáciles |
| 8 | [Regularización](#8-regularización) | Dropout, weight decay, label smoothing, gradient clipping |
| 9 | [Data Augmentation](#9-data-augmentation) | Augmentación de datos en entrenamiento |
| 10 | [Optimización: AdamW y Schedulers](#10-optimización-adamw-y-schedulers) | Optimizador y programación de LR |
| 11 | [Stochastic Weight Averaging (SWA)](#11-stochastic-weight-averaging-swa) | Promediado de pesos |
| 12 | [Mixed Precision Training (AMP)](#12-mixed-precision-training-amp) | Entrenamiento en precisión mixta |
| 13 | [Test-Time Augmentation (TTA)](#13-test-time-augmentation-tta) | Augmentación en inferencia |
| 14 | [Métricas de Evaluación](#14-métricas-de-evaluación) | Accuracy, precision, recall, F1, matriz de confusión |
| 15 | [Reproducibilidad](#15-reproducibilidad) | Semillas y determinismo |
| 16 | [Referencias](#16-referencias) | Bibliografía en formato APA |

---

## 1. Introducción

El presente proyecto aborda el problema de **Human Action Recognition (HAR)** mediante técnicas de **visión por computadora** y **deep learning**. El objetivo es clasificar imágenes de personas realizando 15 actividades humanas distintas (llamar, aplaudir, correr, dormir, etc.) utilizando una red neuronal convolucional basada en **transfer learning**.

El flujo del proyecto sigue tres etapas secuenciales:

1. **Exploración de datos** (`data_prep/`): Análisis estadístico del dataset, distribución de clases, resoluciones y canales de color.
2. **Preprocesamiento** (`data_trans/`): Normalización de resolución, mejora de contraste (CLAHE) y balanceo de clases.
3. **Entrenamiento** (`data_train/`): Transfer learning con ConvNeXt-Base, fine-tuning progresivo, SWA y evaluación con TTA.

La clasificación de acciones humanas a partir de imágenes estáticas es un problema desafiante porque requiere que el modelo capture patrones posturales, contexto espacial e interacción con objetos. A diferencia del reconocimiento de acciones en video (que aprovecha información temporal), las imágenes estáticas contienen una única instantánea, lo que exige representaciones visuales ricas (Poppe, 2010).

---

## 2. Exploración de Datos (`data_prep/`)

### 2.1 Análisis Exploratorio de Datos (EDA)

El Análisis Exploratorio de Datos (Exploratory Data Analysis, EDA) es una etapa fundamental en cualquier proyecto de Machine Learning. Fue formalizado por Tukey (1977) como un enfoque para analizar conjuntos de datos con el fin de resumir sus características principales, frecuentemente mediante métodos visuales.

En este proyecto, el EDA se realiza en `data_explore.py` y abarca:

- **Distribución de clases**: Verificar el balance entre las 15 actividades.
- **Análisis de resoluciones**: Identificar la variabilidad de tamaños para definir el resize objetivo.
- **Estadísticas de canales RGB**: Calcular media y desviación estándar para futuras normalizaciones.

```python
# data_prep/data_explore.py — Estadísticas de canales RGB
channel_sums += arr.reshape(-1, 3).sum(axis=0)
channel_sq_sums += (arr.reshape(-1, 3) ** 2).sum(axis=0)
pixel_count += npixels
# ...
mean_rgb = channel_sums / pixel_count
std_rgb = np.sqrt(channel_sq_sums / pixel_count - mean_rgb ** 2)
```

El cálculo de media y varianza en un solo pase (Welford, 1962) evita almacenar todas las imágenes en memoria, lo que es esencial para datasets grandes.

### 2.2 Balance de Clases

El dataset HAR está **perfectamente balanceado** (840 imágenes por clase). Un dataset balanceado permite usar accuracy como métrica principal sin sesgo hacia clases mayoritarias (He & Garcia, 2009). No obstante, tras el filtrado por tamaño mínimo, algunas clases pierden imágenes, lo que se corrige con oversampling en la etapa de preprocesamiento.

---

## 3. Pipeline de Preprocesamiento (`data_trans/`)

### 3.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE es una variante de la ecualización adaptativa de histograma propuesta por Zuiderveld (1994). A diferencia de la ecualización global, CLAHE opera sobre regiones locales (*tiles*) de la imagen y limita la amplificación del contraste mediante un parámetro `clipLimit`, evitando la amplificación excesiva de ruido.

**Principio de funcionamiento:**

1. La imagen se divide en regiones rectangulares (parámetro `tileGridSize`).
2. En cada región se calcula el histograma local.
3. Se aplica un límite de recorte (*clip limit*) que redistribuye los bins que exceden el umbral.
4. Se interpola bilinealmente entre regiones vecinas para evitar bordes visibles.

**Marco matemático:**

La ecualización de histograma clásica define una función de transferencia $T(r)$ que mapea los niveles de intensidad originales $r$ a nuevos valores uniformemente distribuidos:

$$T(r) = (L - 1) \int_0^r p_r(w) \, dw$$

donde $L$ es el número de niveles de gris (256 para imágenes de 8 bits) y $p_r(w)$ es la función de densidad de probabilidad de la intensidad $w$.

En CLAHE, el paso central es el **recorte del histograma** (*clipping*). Dado un histograma local $h(i)$ con $N_{\text{pixels}}$ píxeles y $N_{\text{bins}}$ bins, el límite de recorte $\beta$ se calcula como:

$$\beta = \frac{N_{\text{pixels}}}{N_{\text{bins}}} \cdot \alpha$$

donde $\alpha$ es el parámetro `clipLimit` (en este proyecto, $\alpha = 2.0$). Los bins que superan $\beta$ se recortan y los píxeles sobrantes se redistribuyen uniformemente:

$$h'(i) = \begin{cases} \beta & \text{si } h(i) > \beta \\ h(i) + \Delta & \text{en caso contrario} \end{cases} \quad \text{con } \Delta = \frac{\sum_{j:\, h(j)>\beta} \bigl(h(j) - \beta\bigr)}{N_{\text{bins}}}$$

**Ejemplo numérico:** Para una región (*tile*) de 32×32 = 1,024 píxeles con 256 bins y `clipLimit=2.0`:
- Promedio por bin: $1024 / 256 = 4$ píxeles
- Límite de recorte: $\beta = 4 \times 2.0 = 8$ píxeles
- Un bin con 12 píxeles se recorta a 8; los 4 sobrantes se redistribuyen entre los demás bins

En este proyecto, CLAHE se aplica sobre el **canal L** (luminancia) del espacio de color **LAB**, lo que mejora el contraste local sin alterar la información cromática:

```python
# data_trans/data_adecuate.py — Aplicación de CLAHE en espacio LAB
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(img_lab)
l_channel = clahe.apply(l_channel)
img_lab = cv2.merge([l_channel, a_channel, b_channel])
img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
```

**¿Por qué LAB y no RGB directo?** El espacio CIE L\*a\*b\* separa la luminancia (L) de la crominancia (a, b). La conversión desde RGB requiere un paso intermedio por el espacio XYZ:

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = M_{\text{sRGB}} \cdot \begin{bmatrix} R' \\ G' \\ B' \end{bmatrix}, \qquad L^* = 116 \cdot f\!\left(\frac{Y}{Y_n}\right) - 16$$

donde $f(t) = t^{1/3}$ si $t > \delta^3$, y $f(t) = \frac{t}{3\delta^2} + \frac{4}{29}$ si $t \leq \delta^3$ (con $\delta = 6/29$). Los canales $a^*$ y $b^*$ codifican la oposición verde-rojo y azul-amarillo respectivamente.

Aplicar CLAHE en RGB modificaría cada canal de color independientemente, generando **artefactos cromáticos** (desaturación, falsos colores). En cambio, operar solo sobre el canal L preserva los colores originales mientras mejora el detalle en zonas oscuras y claras (Reza, 2004).

### 3.2 Interpolación Adaptativa para Resize

El redimensionado utiliza interpolación adaptativa según la dirección del cambio:

```python
# data_trans/data_adecuate.py — Interpolación adaptativa
needs_upscale = (w_orig < TARGET_W) or (h_orig < TARGET_H)
interp = cv2.INTER_CUBIC if needs_upscale else cv2.INTER_AREA
img_resized = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=interp)
```

- **`INTER_AREA`** (downscale): Usa media ponderada de píxeles vecinos. Superior a bilinear para reducción porque evita aliasing (OpenCV, 2024).
- **`INTER_CUBIC`** (upscale): Interpolación bicúbica de 4×4 píxeles vecinos. Produce resultados más suaves que la bilinear para ampliación.

### 3.3 Oversampling con Reemplazo

Tras el filtrado por tamaño mínimo (128×128), las clases quedan ligeramente desbalanceadas. Se aplica **random oversampling con reemplazo** (Chawla et al., 2002) para igualar todas las clases a la más numerosa (838 imágenes):

```python
# data_trans/data_adecuate.py — Oversampling por clase
for label, group in df_valid.groupby("label"):
    n = len(group)
    if n >= max_count:
        balanced_parts.append(group.sample(n=max_count, random_state=42))
    else:
        extra = group.sample(n=max_count - n, replace=True, random_state=42)
        balanced_parts.append(pd.concat([group, extra], ignore_index=True))
```

El oversampling con reemplazo duplica imágenes existentes de las clases minoritarias. Aunque no genera datos nuevos (a diferencia de SMOTE), combinado con data augmentation en entrenamiento, las copias se transforman en variantes únicas en cada época.

---

## 4. Arquitectura del Modelo

Las redes neuronales convolucionales (CNN) son la base de la visión por computadora moderna. La operación fundamental es la **convolución discreta 2D**: dado un kernel $K$ de tamaño $k \times k$ y una imagen de entrada $I$, la salida (*feature map*) en la posición $(i, j)$ es:

$$S(i, j) = (I * K)(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m, n)$$

Una CNN típica apila múltiples capas convolucionales con funciones de activación no lineales y operaciones de pooling, aprendiendo representaciones jerárquicas: las primeras capas detectan bordes y texturas, las intermedias capturan partes de objetos, y las finales reconocen patrones semánticos de alto nivel (Zeiler & Fergus, 2014).

### 4.1 ConvNeXt: Modernizando las ConvNets

ConvNeXt (Liu et al., 2022) es una familia de redes convolucionales puras diseñada para competir directamente con los Vision Transformers (ViT). Los autores partieron de una ResNet-50 estándar y la "modernizaron" sistemáticamente incorporando principios de diseño de los Transformers:

1. **Macro design**: Ratio de stages 3:3:9:3 (similar a Swin Transformer).
2. **Patchify stem**: Conv 4×4 stride 4 en la entrada (reemplaza stem agresivo de ResNet).
3. **Depthwise separable convolutions**: Convs 7×7 depthwise (análogo a attention local de Swin).
4. **Inverted bottleneck**: Canal ancho → estrecho → ancho (como en MobileNetV2/Transformers).
5. **Layer normalization**: Reemplaza BatchNorm en el backbone.
6. **GELU activation**: Reemplaza ReLU. GELU (*Gaussian Error Linear Unit*): $\text{GELU}(x) = x \cdot \Phi(x)$, donde $\Phi$ es la CDF de la distribución normal estándar. A diferencia de ReLU, GELU es diferenciable en todo su dominio y pondera la activación por la probabilidad de que la entrada sea positiva.
7. **Fewer activation/norm layers**: Menos operaciones por bloque (como Transformers).

**ConvNeXt-Base** tiene ~88M parámetros y alcanza 83.8% top-1 accuracy en ImageNet-1K, comparable a Swin-B (83.5%) pero con la simplicidad arquitectural de una ConvNet pura (Liu et al., 2022).

### 4.2 Head Personalizado (Clasificador)

El clasificador original de ConvNeXt (`LayerNorm → Flatten → Linear(1024, 1000)`) se reemplaza por un head multicapa con mayor capacidad expresiva:

```python
# data_train/train_cnn.py — Head personalizado
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
    nn.Linear(256, 15),
)
```

**Diseño del head:**

- **LayerNorm(1024)**: Normaliza las features del backbone, estabilizando la distribución de entrada al head.
- **Reducción progresiva** (1024 → 512 → 256 → 15): Permite al modelo aprender representaciones intermedias entre las features generales de ImageNet y las clases específicas de HAR.
- **BatchNorm1d después de cada Linear**: Normaliza las activaciones por mini-batch, acelerando la convergencia (Ioffe & Szegedy, 2015). Para cada mini-batch $\mathcal{B}$: $\hat{x}_i = (x_i - \mu_{\mathcal{B}}) / \sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}$, seguido de una transformación afín aprendible $y_i = \gamma \hat{x}_i + \beta$ que preserva la capacidad representacional de la capa.
- **ReLU**: Función de activación no lineal $f(x) = \max(0, x)$, computacionalmente eficiente (Nair & Hinton, 2010).
- **Dropout decreciente** (0.3 → 0.2): Mayor regularización en capas anchas (más propensas a memorizar).

---

## 5. Transfer Learning

### 5.1 Fundamento Teórico

Transfer learning consiste en reutilizar un modelo entrenado en una tarea fuente (ej. ImageNet, 1.28M imágenes, 1000 clases) y adaptarlo a una tarea objetivo con menos datos (Yosinski et al., 2014). Las features aprendidas en capas tempranas (bordes, texturas, patrones) son **transferibles** entre dominios visuales, mientras que las capas tardías se especializan progresivamente.

**¿Por qué funciona?** Las capas convolucionales tempranas aprenden filtros genéricos (detectores de bordes tipo Gabor, blobs de color) que son universales. Las capas intermedias capturan texturas y parts. Solo las capas finales se especializan en las clases del dataset original (Zeiler & Fergus, 2014). Al transferir estos conocimientos, el modelo necesita muchos menos datos para converger.

### 5.2 Estrategia de Dos Fases

El entrenamiento se divide en dos fases para maximizar la estabilidad de transfer learning:

**Fase 1 — Entrenar solo el head (backbone congelado):**

```python
# data_train/train_cnn.py — Congelamiento del backbone
for name, param in model.named_parameters():
    if not name.startswith("classifier"):
        param.requires_grad = False
```

Se entrenan solo los ~660K parámetros del head durante 15 épocas. Esto permite que las capas de clasificación aprendan a mapear las features de ImageNet a las 15 clases de HAR sin perturbar las representaciones del backbone.

**Fase 2 — Fine-tuning progresivo:**

```python
# data_train/train_cnn.py — Descongelamiento progresivo
for name, param in model.named_parameters():
    if name.startswith(("features.5.", "features.6.", "classifier.")):
        param.requires_grad = True
    else:
        param.requires_grad = False
```

Se descongelan gradualmente más capas del backbone, permitiendo que las features se adapten al dominio HAR. Se usa un **learning rate diferencial**: menor para el backbone (3e-5) y mayor para el head (1e-4), evitando que el fine-tuning destruya las features pretrained.

---

## 6. Progressive Unfreezing

### 6.1 Concepto

Progressive unfreezing, propuesto por Howard & Ruder (2018) en el contexto de ULMFiT, consiste en descongelar las capas del modelo gradualmente, de las más externas (cercanas a la salida) a las más internas (cercanas a la entrada).

**Justificación:** Las capas tempranas contienen features más genéricas y transferibles. Si se descongelan de golpe con un learning rate alto, sus pesos se pueden degradar ("catastrophic forgetting"). El descongelamiento gradual permite a cada capa adaptarse antes de que la siguiente se active.

### 6.2 Implementación

En este proyecto, la progresión es:

| Épocas F2 | Capas entrenables | Parámetros |
|-----------|-------------------|-----------|
| 1–10 | `features[5:7]` + `classifier` | ~60M |
| 11+ | Todo el backbone + `classifier` | ~88M |

```python
# data_train/train_cnn.py — Descongelamiento total
if epoch == UNFREEZE_AFTER + 1:
    for param in model.parameters():
        param.requires_grad = True
    optimizer2, scheduler2 = _build_optimizer_and_scheduler(
        model, last_epoch=epoch - 2)
```

Las capas `features[5:7]` corresponden a los dos últimos stages de ConvNeXt-Base, que contienen features más especializadas y son las primeras candidatas para adaptación al dominio HAR.

---

## 7. Función de Pérdida: Focal Loss

### 7.1 Motivación

La Cross-Entropy estándar trata todos los ejemplos por igual. En problemas de clasificación, muchos ejemplos son "fáciles" (el modelo ya los clasifica correctamente con alta confianza). Estos ejemplos dominan el gradiente y limitan el aprendizaje de los ejemplos difíciles.

### 7.2 Definición Matemática

**Softmax y Cross-Entropy (preliminares):**

La función **softmax** convierte los logits $z_k$ (salida cruda de la red) en una distribución de probabilidad sobre las $K$ clases:

$$\sigma(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad \text{con } \sum_{k=1}^{K} \sigma(z_k) = 1$$

La **Cross-Entropy Loss** mide la discrepancia entre la distribución predicha $\hat{y}$ y la etiqueta real (one-hot vector $y$):

$$\text{CE}(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log(\hat{y}_k) = -\log(p_c)$$

donde $p_c$ es la probabilidad asignada a la clase correcta $c$. Definimos $p_t \equiv p_c$ como la **probabilidad de la clase verdadera** — es la cantidad central en Focal Loss.

**Focal Loss:**

Focal Loss fue propuesta por Lin et al. (2017) para detección de objetos desbalanceada, pero se generaliza a cualquier clasificación:

$$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

donde:

- $p_t$ es la probabilidad predicha para la clase correcta.
- $\gamma$ es el parámetro de *focusing* (en este proyecto, $\gamma = 2.0$).

Cuando $\gamma = 0$, Focal Loss se reduce a Cross-Entropy. A medida que $\gamma$ aumenta, la contribución de los ejemplos bien clasificados ($p_t \to 1$) se reduce exponencialmente:

| $p_t$ | CE Loss | Focal Loss ($\gamma=2$) | Reducción |
|-------|---------|------------------------|-----------|
| 0.9 | 0.105 | 0.001 | 99% |
| 0.5 | 0.693 | 0.173 | 75% |
| 0.1 | 2.303 | 1.867 | 19% |

**Ejemplo numérico detallado:** Supongamos un mini-batch de 3 imágenes con $\gamma = 2.0$:

| Imagen | Clase real | $p_t$ | CE: $-\log(p_t)$ | Factor focal: $(1-p_t)^2$ | Focal Loss |
|--------|-----------|-------|-------------------|---------------------------|------------|
| Img 1 | cycling | 0.95 | 0.051 | 0.0025 | 0.00013 |
| Img 2 | dancing | 0.60 | 0.511 | 0.1600 | 0.08170 |
| Img 3 | fighting | 0.15 | 1.897 | 0.7225 | 1.37058 |

La Img 1 (fácil, $p_t = 0.95$) contribuye **0.00013** al loss — prácticamente nada. La Img 3 (difícil, $p_t = 0.15$) contribuye **1.37** — unas 10,000 veces más. Esto fuerza al modelo a concentrar su capacidad de aprendizaje en los ejemplos que aún no clasifica bien, acelerando la convergencia en clases difíciles.

### 7.3 Implementación

```python
# data_train/train_cnn.py — Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)           # p_t = exp(-CE) = probabilidad de la clase correcta
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

Se usa `reduction="none"` en `CrossEntropyLoss` para obtener la pérdida por ejemplo y luego aplicar la modulación focal individualmente.

---

## 8. Regularización

La regularización previene el sobreajuste (overfitting), que ocurre cuando el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generalizables (Goodfellow et al., 2016, Cap. 7).

### 8.1 Dropout

Dropout (Srivastava et al., 2014) desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento, forzando al modelo a no depender de ninguna neurona individual:

$$\tilde{h}_i = m_i \cdot h_i, \quad m_i \sim \text{Bernoulli}(1 - p)$$

En el head se usan tasas decrecientes (0.3 → 0.2) porque las capas más anchas tienen mayor riesgo de memorización.

### 8.2 Weight Decay (Regularización L2)

Weight decay penaliza pesos grandes añadiendo un término a la función de pérdida (Krogh & Hertz, 1991):

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2} \sum_i w_i^2$$

Con AdamW, el weight decay se aplica de forma **desacoplada** del gradiente (Loshchilov & Hutter, 2019), lo que es crucial para la correcta regularización con optimizadores adaptativos:

```python
# data_train/train_cnn.py — AdamW con weight decay desacoplado
optimizer = optim.AdamW([
    {"params": backbone_p, "lr": LR_BACKBONE},
    {"params": model.classifier.parameters(), "lr": LR_HEAD_PHASE2},
], weight_decay=WEIGHT_DECAY)  # WEIGHT_DECAY = 5e-4
```

### 8.3 Label Smoothing

Label smoothing (Szegedy et al., 2016) suaviza las etiquetas one-hot, distribuyendo una pequeña probabilidad ($\epsilon$) entre todas las clases:

$$y'_k = (1 - \epsilon) \cdot y_k + \frac{\epsilon}{K}$$

Con $\epsilon = 0.02$ y $K = 15$ clases, la clase correcta recibe probabilidad 0.9813 en vez de 1.0, y cada clase incorrecta recibe 0.0013. Esto previene que el modelo se vuelva "sobreconfiado" y mejora la calibración de probabilidades (Müller et al., 2019).

### 8.4 Gradient Clipping

Gradient clipping limita la norma del gradiente para evitar explosiones que desestabilizan el entrenamiento (Pascanu et al., 2013), especialmente relevante con AMP donde los gradientes pueden tener mayor varianza:

```python
# data_train/train_cnn.py — Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Si $\|g\| > \text{max\_norm}$, el gradiente se escala: $g' = g \cdot \frac{\text{max\_norm}}{\|g\|}$

---

## 9. Data Augmentation

### 9.1 Fundamento

Data augmentation genera variantes artificiales de los datos de entrenamiento aplicando transformaciones que preservan la semántica de la etiqueta (Shorten & Khoshgoftaar, 2019). Esto actúa como regularización implícita y aumenta virtualmente el tamaño del dataset.

### 9.2 Transformaciones Aplicadas

```python
# data_train/train_cnn.py — Pipeline de augmentación
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
```

| Transformación | Descripción | Justificación |
|----------------|-------------|---------------|
| **RandomHorizontalFlip** | Espejo horizontal con p=0.5 | Las actividades humanas son simétricas lateralmente |
| **RandomRotation(20°)** | Rotación aleatoria ±20° | Simula variaciones de ángulo de cámara |
| **ColorJitter** | Variaciones de brillo, contraste, saturación y tonalidad | Robustez a condiciones de iluminación |
| **TrivialAugmentWide** | Una transformación aleatoria de un conjunto amplio (Müller & Hutter, 2021) | Diversidad sin hiperparámetros adicionales |
| **RandomErasing** | Borrado aleatorio de una región rectangular (p=0.10) | Simula oclusiones parciales (Zhong et al., 2020) |
| **Normalize (ImageNet)** | Normaliza con media y std de ImageNet | Alinea la distribución con los pesos pretrained |

### 9.3 TrivialAugmentWide

TrivialAugment (Müller & Hutter, 2021) se distingue de RandAugment y AutoAugment por su simplicidad: selecciona **una sola** transformación al azar (de ~31 posibles) con una magnitud uniforme aleatoria. No requiere búsqueda de hiperparámetros y supera consistentemente a métodos más complejos:

> "We propose TrivialAugment, the simplest possible baseline, which samples a single augmentation uniformly at random and applies it with a random strength." (Müller & Hutter, 2021, p. 1)

---

## 10. Optimización: AdamW y Schedulers

### 10.1 AdamW

Adam (Kingma & Ba, 2015) combina momentos de primer y segundo orden para adaptar el learning rate por parámetro. AdamW (Loshchilov & Hutter, 2019) corrige un error en la implementación original de weight decay en Adam, desacoplando la regularización del paso de actualización:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$w_t = w_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_{t-1} \right)$$

### 10.2 Learning Rate Diferencial

Se asignan learning rates distintos al backbone y al head:

```python
# data_train/train_cnn.py — LR diferencial
opt = optim.AdamW([
    {"params": backbone_p, "lr": LR_BACKBONE},     # 3e-5
    {"params": model.classifier.parameters(), "lr": LR_HEAD_PHASE2},  # 1e-4
], weight_decay=WEIGHT_DECAY)
```

El backbone ya tiene features bien calibradas de ImageNet y necesita ajustes finos (LR bajo). El head se inicializa aleatoriamente y necesita un LR más alto para converger rápido (Howard & Ruder, 2018).

### 10.3 SequentialLR: Warmup + Cosine Annealing

El scheduler combina dos fases:

1. **LinearLR (warmup)**: Escala el LR de 10% al 100% durante las primeras 10 épocas. Esto estabiliza el entrenamiento inicial cuando los gradientes del head recién inicializado son ruidosos (Goyal et al., 2017).

2. **CosineAnnealingLR**: Decae el LR siguiendo una curva coseno:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \pi}{T}\right)\right)$$

El decay coseno produce un decaimiento suave que tiende a encontrar mínimos más anchos (y generalizables) que el step decay (Loshchilov & Hutter, 2017).

```python
# data_train/train_cnn.py — SequentialLR
warmup = optim.lr_scheduler.LinearLR(
    opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=PHASE2_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
sched = optim.lr_scheduler.SequentialLR(
    opt, [warmup, cosine], milestones=[WARMUP_EPOCHS])
```

---

## 11. Stochastic Weight Averaging (SWA)

### 11.1 Fundamento

SWA (Izmailov et al., 2018) promedia los pesos del modelo a lo largo de múltiples épocas del entrenamiento, obteniendo una solución que tiende a estar en regiones más "planas" del landscape de pérdida. Estas regiones planas generalizan mejor que mínimos "estrechos" (Hochreiter & Schmidhuber, 1997):

$$w_{SWA} = \frac{1}{n} \sum_{i=1}^{n} w_i$$

### 11.2 ¿Por qué SWA mejora la generalización?

La intuición es geométrica: el promedio de dos puntos en el espacio de pesos que están en mínimos diferentes del landscape de pérdida tiende a caer en una región de bajo loss si los mínimos están conectados ("mode connectivity"). SWA explora esta propiedad promediando a lo largo de la trayectoria de entrenamiento.

Experimentalmente, SWA produce mejoras de **+0.5–1.5 pp** en accuracy sin costo computacional adicional significativo (Izmailov et al., 2018).

### 11.3 Implementación

```python
# data_train/train_cnn.py — SWA
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer2, swa_lr=SWA_LR)  # SWA_LR = 1e-5

# Durante entrenamiento (a partir de época 20 de F2):
if swa_active:
    swa_model.update_parameters(model)
    swa_scheduler.step()
```

**Parámetros:**

- `SWA_START_EPOCH = 20`: Se activa después de que el modelo haya convergido en las métricas regulares.
- `SWA_LR = 1e-5`: Learning rate constante y bajo para que los pesos oscilen cerca del mínimo.

### 11.4 Actualización de BatchNorm

Después de promediar los pesos, las estadísticas de BatchNorm (running_mean, running_var) quedan desactualizadas. Se hace un forward pass completo sobre el train set para recalcularlas:

```python
# data_train/train_cnn.py — Actualizar BN para SWA
if swa_active:
    update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.module.state_dict(), OUTPUT_DIR / "har_cnn_swa.pth")
```

---

## 12. Mixed Precision Training (AMP)

### 12.1 Concepto

Automatic Mixed Precision (AMP) alterna entre FP32 y FP16 durante el entrenamiento para aprovechar los Tensor Cores de las GPUs NVIDIA modernas (Micikevicius et al., 2018):

- **Forward pass**: FP16 (más rápido, menor memoria).
- **Loss y gradientes**: FP32 (para estabilidad numérica).
- **Actualización de pesos**: FP32 (máxima precisión para acumulación).

### 12.2 Beneficios

| Aspecto | FP32 | FP16 (AMP) |
|---------|------|------------|
| Memoria por parámetro | 4 bytes | 2 bytes |
| Velocidad (Tensor Cores) | 1× | 2-3× |
| Rango dinámico | $\pm 3.4 \times 10^{38}$ | $\pm 65,504$ |

### 12.3 GradScaler

Para compensar el rango reducido de FP16, se usa un `GradScaler` que escala la loss antes del backward para evitar underflow de gradientes:

```python
# data_train/train_cnn_colab.ipynb — AMP con GradScaler
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    outputs = model(imgs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Nota:** AMP se usa en el notebook de Colab (`train_cnn_colab.ipynb`) con GPU T4. El script local (`train_cnn.py`) opera en FP32 por simplicidad y reproducibilidad.

---

## 13. Test-Time Augmentation (TTA)

### 13.1 Concepto

TTA aplica múltiples transformaciones aleatorias a cada imagen de test y promedia las predicciones (Wang et al., 2019). Esto reduce la varianza de la predicción y mejora la robustez:

$$\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f(T_i(x))$$

donde $T_i$ son transformaciones aleatorias y $f$ es el modelo.

### 13.2 Implementación

```python
# data_train/train_cnn.py — TTA
tta_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Promedio de 5 pasadas (1 base + 4 con augmentación)
for i in range(N_TTA - 1):
    tta_ds = HARDataset(X_test_df, IMG_DIR, transform=tta_augment)
    # ... acumular logits
avg_logits /= N_TTA
```

**Notas de diseño:**

- Las augmentaciones de TTA son **más suaves** que las de entrenamiento (sin `TrivialAugmentWide` ni `RandomErasing`) para no distorsionar excesivamente la imagen.
- Se promedian los **logits** (pre-softmax) en vez de las probabilidades, lo que es equivalente al ensemble de modelos por promedio de predicciones.
- Con 5 pasadas se obtiene típicamente **+0.5–1.0 pp** de accuracy (Shanmugam et al., 2021).

---

## 14. Métricas de Evaluación

### 14.1 Accuracy

$$\text{Accuracy} = \frac{\text{Predicciones correctas}}{\text{Total de predicciones}}$$

Es la métrica principal dado que el dataset está balanceado (840 imágenes/clase). En datasets desbalanceados, accuracy puede ser engañosa (Japkowicz & Stephen, 2002).

### 14.2 Precision, Recall y F1-Score

Para cada clase $c$:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c} \qquad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

El `classification_report` de scikit-learn calcula estas métricas por clase y sus promedios macro (no ponderado) y weighted (ponderado por soporte):

```python
# data_train/train_cnn.py — Classification report
report = classification_report(all_labels, all_preds, target_names=labels_sorted)
```

### 14.3 Matriz de Confusión

La matriz de confusión $C$ de $K \times K$ permite visualizar los errores de clasificación. $C_{ij}$ indica cuántas instancias de la clase $i$ fueron predichas como clase $j$. Los elementos diagonales representan las predicciones correctas; los fuera de la diagonal representan confusiones entre clases.

---

## 15. Reproducibilidad

### 15.1 Importancia

La reproducibilidad es un pilar fundamental de la ciencia y la ingeniería de ML (Pineau et al., 2021). Un experimento reproducible permite:

- Verificar resultados por terceros.
- Comparar modelos bajo las mismas condiciones.
- Depurar problemas de entrenamiento.

### 15.2 Mecanismos Implementados

```python
# data_train/train_cnn.py — Semillas para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

| Componente | Mecanismo | Propósito |
|------------|-----------|-----------|
| `random.seed(42)` | Python stdlib | Reproducir operaciones internas de torchvision |
| `np.random.seed(42)` | NumPy | Reproducir operaciones de scikit-learn y pandas |
| `torch.manual_seed(42)` | PyTorch CPU | Inicialización de pesos y operaciones aleatorias |
| `torch.cuda.manual_seed_all(42)` | PyTorch GPU | Semilla en todas las GPUs |
| `cudnn.deterministic = True` | cuDNN | Forzar algoritmos deterministas |
| `cudnn.benchmark = False` | cuDNN | Desactivar autotuning no determinista |
| `Generator().manual_seed(42)` | DataLoader | Shuffle de batches reproducible |
| `worker_init_fn` | DataLoader workers | Semilla por worker = 42 + worker_id |

> **Trade-off Colab:** El notebook de Colab prioriza velocidad con `deterministic=False`, `benchmark=True`, TF32 y `torch.compile`, lo que puede producir variaciones entre ejecuciones (NVIDIA, 2024).

---

## 16. Referencias

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. *arXiv preprint arXiv:1706.02677*. https://arxiv.org/abs/1706.02677

He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Hochreiter, S., & Schmidhuber, J. (1997). Flat Minima. *Neural Computation*, *9*(1), 1–42. https://doi.org/10.1162/neco.1997.9.1.1

Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 328–339. https://doi.org/10.18653/v1/P18-1031

Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456. https://proceedings.mlr.press/v37/ioffe15.html

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI)*, 876–885. https://arxiv.org/abs/1803.05407

Japkowicz, N., & Stephen, S. (2002). The Class Imbalance Problem: A Systematic Study. *Intelligent Data Analysis*, *6*(5), 429–449. https://doi.org/10.3233/IDA-2002-6504

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

Krogh, A., & Hertz, J. A. (1991). A Simple Weight Decay Can Improve Generalization. *Advances in Neural Information Processing Systems (NeurIPS)*, *4*, 950–957.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11976–11986. https://doi.org/10.1109/CVPR52688.2022.01167

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1608.03983

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1711.05101

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed Precision Training. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1710.03740

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When Does Label Smoothing Help? *Advances in Neural Information Processing Systems (NeurIPS)*, *32*, 4694–4703. https://arxiv.org/abs/1906.02629

Müller, S. G., & Hutter, F. (2021). TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 774–782. https://doi.org/10.1109/ICCV48922.2021.00081

Nair, V., & Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 807–814.

NVIDIA. (2024). *CUDA Toolkit Documentation: Reproducibility*. https://docs.nvidia.com/cuda/cublas/index.html#reproducibility

OpenCV. (2024). *Geometric Image Transformations*. https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 1310–1318. https://proceedings.mlr.press/v28/pascanu13.html

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving Reproducibility in Machine Learning Research. *Journal of Machine Learning Research*, *22*(164), 1–20. https://jmlr.org/papers/v22/20-303.html

Poppe, R. (2010). A Survey on Vision-Based Human Action Recognition. *Image and Vision Computing*, *28*(6), 976–990. https://doi.org/10.1016/j.imavis.2009.11.014

Reza, A. M. (2004). Realization of the Contrast Limited Adaptive Histogram Equalization (CLAHE) for Real-Time Image Enhancement. *Journal of VLSI Signal Processing Systems*, *38*(1), 35–44. https://doi.org/10.1023/B:VLSI.0000028532.53893.82

Shanmugam, D., Blalock, D., Balber, G., & Guttag, J. (2021). Better Aggregation in Test-Time Augmentation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1214–1223. https://doi.org/10.1109/ICCV48922.2021.00125

Shorten, C., & Khoshgoftaar, T. M. (2019). A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, *6*, 60. https://doi.org/10.1186/s40537-019-0197-0

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, *15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2818–2826. https://doi.org/10.1109/CVPR.2016.308

Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.

Wang, G., Li, W., Aertsen, M., Deprest, J., Ourselin, S., & Vercauteren, T. (2019). Aleatoric Uncertainty Estimation with Test-Time Augmentation for Medical Image Segmentation with Convolutional Neural Networks. *Neurocomputing*, *338*, 34–45. https://doi.org/10.1016/j.neucom.2019.01.103

Welford, B. P. (1962). Note on a Method for Calculating Corrected Sums of Squares and Products. *Technometrics*, *4*(3), 419–420. https://doi.org/10.1080/00401706.1962.10490022

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How Transferable Are Features in Deep Neural Networks? *Advances in Neural Information Processing Systems (NeurIPS)*, *27*, 3320–3328. https://arxiv.org/abs/1411.1792

Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. *Proceedings of the European Conference on Computer Vision (ECCV)*, 818–833. https://doi.org/10.1007/978-3-319-10590-1_53

Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random Erasing Data Augmentation. *Proceedings of the AAAI Conference on Artificial Intelligence*, *34*(7), 13001–13008. https://doi.org/10.1609/aaai.v34i07.7000

Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In P. S. Heckbert (Ed.), *Graphics Gems IV* (pp. 474–485). Academic Press. https://doi.org/10.1016/B978-0-12-336156-1.50061-6

---

<div align="center">

*Documento generado como parte del TP de Machine Learning — Fundamentos teóricos del proyecto HAR*

</div>
