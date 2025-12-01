---
title: Monitoreo de actividades con Inteligencia artificial
theme:
  name: terminal-dark
---

## Red Neuronal
### Librerias    

```python +line_numbers
from datasets import load_dataset
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
```

<!-- end_slide -->

## Red Neuronal
### Dataset

```python +line_numbers
dataset = load_dataset("guillherms/human-activity-pose_v4")
```

### Split

```python +line_numbers
train_ds = dataset["train"]
test_ds = dataset["validation"]
print(train_ds, "\n", test_ds)
```

<!-- end_slide -->

## Red Neuronal
### Exploracion del dataset

<!-- column_layout: [2,2] -->

<!-- column: 0 -->
> train_ds
```sh
Dataset({
    features: ['x0', 'y0', 'z0', 'v0', 'x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4', 'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7', 'z7', 'v7', 'x8', 'y8', 'z8', 'v8', 'x9', 'y9', 'z9', 'v9', 'x10', 'y10', 'z10', 'v10', 'x11', 'y11', 'z11', 'v11', 'x12', 'y12', 'z12', 'v12', 'x13', 'y13', 'z13', 'v13', 'x14', 'y14', 'z14', 'v14', 'x15', 'y15', 'z15', 'v15', 'x16', 'y16', 'z16', 'v16', 'x17', 'y17', 'z17', 'v17', 'x18', 'y18', 'z18', 'v18', 'x19', 'y19', 'z19', 'v19', 'x20', 'y20', 'z20', 'v20', 'x21', 'y21', 'z21', 'v21', 'x22', 'y22', 'z22', 'v22', 'x23', 'y23', 'z23', 'v23', 'x24', 'y24', 'z24', 'v24', 'x25', 'y25', 'z25', 'v25', 'x26', 'y26', 'z26', 'v26', 'x27', 'y27', 'z27', 'v27', 'x28', 'y28', 'z28', 'v28', 'x29', 'y29', 'z29', 'v29', 'x30', 'y30', 'z30', 'v30', 'x31', 'y31', 'z31', 'v31', 'x32', 'y32', 'z32', 'v32', 'label'],
    num_rows: 736
}) 
```

<!-- column: 1 -->
> test_ds
```sh
 Dataset({
    features: ['x0', 'y0', 'z0', 'v0', 'x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4', 'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7', 'z7', 'v7', 'x8', 'y8', 'z8', 'v8', 'x9', 'y9', 'z9', 'v9', 'x10', 'y10', 'z10', 'v10', 'x11', 'y11', 'z11', 'v11', 'x12', 'y12', 'z12', 'v12', 'x13', 'y13', 'z13', 'v13', 'x14', 'y14', 'z14', 'v14', 'x15', 'y15', 'z15', 'v15', 'x16', 'y16', 'z16', 'v16', 'x17', 'y17', 'z17', 'v17', 'x18', 'y18', 'z18', 'v18', 'x19', 'y19', 'z19', 'v19', 'x20', 'y20', 'z20', 'v20', 'x21', 'y21', 'z21', 'v21', 'x22', 'y22', 'z22', 'v22', 'x23', 'y23', 'z23', 'v23', 'x24', 'y24', 'z24', 'v24', 'x25', 'y25', 'z25', 'v25', 'x26', 'y26', 'z26', 'v26', 'x27', 'y27', 'z27', 'v27', 'x28', 'y28', 'z28', 'v28', 'x29', 'y29', 'z29', 'v29', 'x30', 'y30', 'z30', 'v30', 'x31', 'y31', 'z31', 'v31', 'x32', 'y32', 'z32', 'v32', 'label'],
    num_rows: 184
})
```

<!-- end_slide -->

## Red Neuronal
### Exploracion y preprocesamiento del dataset

```python +line_numbers
df_train = train_ds.to_pandas()
df_test = test_ds.to_pandas()
x_train = df_train.drop("label", axis=1).values

le = LabelEncoder()
y_train = le.fit_transform(df_train["label"].values)
x_test = df_test.drop("label", axis=1).values
y_test = le.transform(df_test["label"].values)

num_classes = len(np.unique(y_train))
print(f"Detected {num_classes} classes.")
print(f"Input shape: {x_train.shape[1]}")
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
```

```sh
Detected 9 classes.
Input shape: 132
Class mapping: {'agreeing': 0, 'dancing': 1, 'handshake': 2, 'lying_down': 3, 'medical_observation': 4, 'medical_procedure': 5, 'office_work': 6, 'reading': 7, 'waving': 8}
```

<!-- end_slide -->

## Red Neuronal
### Estructura del modelo

```python +line_numbers
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(66, activation="relu", input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(132, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
```
<!-- column_layout: [2,2] -->

<!-- column: 0 -->

### Shape

```python +line_numbers
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
```
<!-- column: 1 -->

```sh
x_train shape: (736, 132)
y_train shape: (736,)
x_test shape: (184, 132)
y_test shape: (184,)
```

<!-- end_slide -->

## Red neuronal
### Estructura del modelo

```sh
model.summary()
Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 66)             │         8,778 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 132)            │         8,844 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 64)             │         8,512 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 128)            │         8,320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 9)              │         1,161 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 35,615 (139.12 KB)
 Trainable params: 35,615 (139.12 KB)
 Non-trainable params: 0 (0.00 B)
```

<!-- end_slide -->

## Red Neuronal
### Entrenamiento

```python +line_numbers
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
```

```sh
Epoch 45/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9914 - loss: 0.0217 - val_accuracy: 0.9946 - val_loss: 0.0097
Epoch 46/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9836 - loss: 0.0587 - val_accuracy: 0.9837 - val_loss: 0.0531
Epoch 47/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9922 - loss: 0.0350 - val_accuracy: 0.9565 - val_loss: 0.1898
Epoch 48/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9851 - loss: 0.0545 - val_accuracy: 0.9946 - val_loss: 0.0249
Epoch 49/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9959 - loss: 0.0209 - val_accuracy: 1.0000 - val_loss: 0.0016
Epoch 50/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.9968 - loss: 0.0155 - val_accuracy: 1.0000 - val_loss: 0.0029
```

<!-- end_slide -->

## Red Neuronal
### Guardado y exportacion

```python +line_numbers
drive.mount('/content/drive/', force_remount=True)
file = "pose_clasification_model"

model.save(f"{file}.h5")
labels_list = list(le.classes_)
with open(f"{file}_labels.json", "w") as f:
    json.dump(labels_list, f)
print(f"Saved model and labels for: {labels_list}")

!cp "{file}.h5" /content/drive/MyDrive/Models/
!cp "{file}_labels.json" /content/drive/MyDrive/Models/
```

```python +line_numbers
tfjs_target_dir = f"{file}_tfjs"
!mkdir -p "{tfjs_target_dir}"
!tensorflowjs_converter --input_format=keras "{file}.h5" "{tfjs_target_dir}"
!cp -r "{tfjs_target_dir}" /content/drive/MyDrive/Models/
print("Carpeta exportada a Drive.")
```

<!-- end_slide -->

## Demo

> Monitoreo de poses en tiempo real usando OpenCV y MediaPipe

![image:width:100%](./assets/opencv.png)

<!-- end_slide -->

## Implementacion

![image:width:100%](./assets/camera.png)

<!-- end_slide -->

## Implementacion

![image:width:100%](./assets/gallery.png)
<!-- end_slide -->
## Implementacion
### Estructura de archivos

```sh
server.js
index.html
logica
├── camera.js
├── gallery.js
├── main.js
├── mediapipe.js
└── model.js
pose_clasification_model_tfjs
├── group1-shard1of1.bin
├── model.json
├── pose_clasification_model.h5
└── pose_clasification_model_labels.json
```
<!-- end_slide -->
## Implementacion
### Tegnologias

| Tecnología      | Descripción                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Node.js         | Plataforma para ejecutar JavaScript en el servidor y servir archivos estáticos. |
| TensorFlow.js   | Librería para ejecutar modelos de machine learning en el navegador.          |
| MediaPipe       | Framework de Google para detección de poses y landmarks en tiempo real.      |
| Bootstrap       | Framework CSS para diseño responsivo y componentes visuales modernos.        |

### Flujo de la aplicación

| Archivo / Carpeta                        | Descripción                                                                                 |
|------------------------------------------|---------------------------------------------------------------------------------------------|
| `server.js`                              | Servidor Node.js que entrega los archivos estáticos de la aplicación web.                   |
| `index.html`                             | Página principal de la app, integra los scripts y la interfaz de usuario.                   |
| `logica/camera.js`                       | Módulo para gestionar el acceso, activación y desactivación de la cámara web.               |
| `logica/gallery.js`                      | Módulo para mostrar y actualizar la galería de imágenes clasificadas.                       |
| `logica/main.js`                         | Controlador principal: orquesta cámara, modelo, predicción y UI.                            |
| `logica/mediapipe.js`                    | Funciones para usar MediaPipe Pose y procesar landmarks y bounding box.                     |
| `logica/model.js`                        | Carga el modelo TensorFlow.js y realiza la predicción de la pose.                           |
| `pose_clasification_model_tfjs/`         | Carpeta con el modelo de clasificación exportado a TensorFlow.js y sus etiquetas.           |
| `assets/`                                | Imágenes y recursos gráficos usados en la presentación y la app.    
<!-- end_slide -->