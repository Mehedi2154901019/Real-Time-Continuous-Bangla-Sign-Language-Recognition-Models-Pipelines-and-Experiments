# Dynamic Bangla Sign Language Recognition

A student-independent deep learning framework for **dynamic Bangla Sign Language (BdSL) recognition** using hand and upper-body landmarks extracted from video sequences.

The main experiment uses **130 features per frame × 120 frames per video** and a hybrid **Conv1D + Bidirectional LSTM + Attention** architecture for recognition of 40 dynamic sign classes.

---

## Overview

This repository contains the code, trained model, preprocessing artifacts, and experimental notebooks for a dynamic Bangla Sign Language recognition system.

The complete pipeline is:

```text
Video
  ↓
MediaPipe Hand + Pose Landmark Extraction
  ↓
130 Features / Frame
  ↓
120 Frames / Video
  ↓
Standardization
  ↓
Conv1D
  ↓
Bidirectional LSTM
  ↓
Bidirectional LSTM
  ↓
Attention
  ↓
Temporal Global Average Pooling
  ↓
Fully Connected Layers
  ↓
40-Class Prediction
```

The system is designed using a **student-independent split**, meaning that the students appearing in the training set are different from those in the validation and test sets.

---

## Repository Contents

The repository contains the following main files:

```text
.
├── 120frames_130values.ipynb
├── final_dynamic.ipynb
├── numpy_extraction.ipynb
├── app.py
│
├── best_model.keras
├── classes.json
├── label_encoder.pkl
├── scaler.pkl
│
├── 60frames_130values.ipynb
├── 80frames_130values.ipynb
└── 120frames_ablation_study.ipynb
```

### Main files

| File                        | Description                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `numpy_extraction.ipynb`    | Extracts 130 landmark-based features from every video frame and saves them as NumPy arrays.  |
| `120frames_130values.ipynb` | Main 120-frame × 130-feature training and evaluation experiment.                             |
| `final_dynamic.ipynb`       | Final training pipeline used to generate deployment artifacts for the Streamlit application. |
| `app.py`                    | Real-time webcam-based sign recognition application.                                         |
| `best_model.keras`          | Saved trained Keras model.                                                                   |
| `classes.json`              | Class names used by the deployed model.                                                      |
| `label_encoder.pkl`         | Saved label encoder.                                                                         |
| `scaler.pkl`                | Training-fitted `StandardScaler` used during inference.                                      |

Additional notebooks are provided for users interested in the ablation experiments.

---

# Dataset

The dynamic video dataset used in this work was originally collected under the **Dhaka University Centennial Research Grant**.

The original dataset contains:

* **50 students**
* **40 dynamic sign classes**
* Each student performs all 40 classes.
* Each video is approximately **4 seconds**
* **30 FPS**
* Exactly **120 frames per video**

For the student-independent experiment, the 50 student folders were divided into:

```text
Training   : 35 students
Validation : 7 students
Testing    : 8 students
```

Each split contains the same 40 sign classes.

The student folders were subsequently merged class-wise to obtain the following structure:

```text
BdSL_dynamic_merged_video/
│
├── train/
│   ├── class_01/
│   ├── class_02/
│   ├── ...
│   └── class_40/
│
├── validation/
│   ├── class_01/
│   ├── class_02/
│   ├── ...
│   └── class_40/
│
└── test/
    ├── class_01/
    ├── class_02/
    ├── ...
    └── class_40/
```

The original dataset contains Bengali class names. For the repository and experiments, the class folder names were represented using **Romanized Bangla** for easier visual accessibility.

### Dataset availability

The original video dataset is **not currently distributed with this repository**.

Following the guidance associated with the dataset and the research supervision, we are not publicly uploading the dataset to Zenodo or another public repository at this time.

Therefore, the repository provides the complete processing, training, evaluation, and deployment pipeline, while the original video data remains unavailable for direct public download.

**When the dataset becomes publicly available, the corresponding dataset link and access instructions will be added to this README.**

---

# Sign Classes

The dataset contains 40 dynamic sign classes.

Examples include:

| Romanized Bangla                   | English                   |
| ---------------------------------- | ------------------------- |
| `office`                           | Office                    |
| `apnake amar valo legeche`         | I seem to like you        |
| `apnake kivabe sahajjo korte pari` | How can I help you        |
| `apnar nam ki`                     | What is your name         |
| `apni ki kaj koren`                | What do you do for living |
| `apni kemon achen`                 | How are you               |
| `apni valo thakben`                | May you stay fine         |
| `abar dekha hbe`                   | See you again             |
| `ami dukkhito`                     | I am sorry                |
| `ami valo achi`                    | I am fine                 |
| `asha`                             | Hope                      |
| `internet`                         | Internet                  |
| `computer`                         | Computer                  |
| `kolom`                            | Pen                       |
| `college`                          | College                   |
| `kaj`                              | Work                      |
| `ghumano`                          | Sleep                     |
| `dekha`                            | Meet                      |
| `dhonnobad`                        | Thank you                 |
| `fan`                              | Fan                       |
| `boi`                              | Book                      |
| `baba`                             | Father                    |
| `bon`                              | Sister                    |
| `vai`                              | Brother                   |
| `ma`                               | Mother                    |
| `mobile`                           | Mobile                    |
| `light`                            | Light                     |
| `shuvo oporanho`                   | Good afternoon            |
| `shuvo jonmodin`                   | Happy birthday            |
| `shuvo sokal`                      | Good morning              |
| `shuvo ratri`                      | Good night                |
| `stri`                             | Wife                      |
| `shami`                            | Husband                   |
| `hata`                             | Walk                      |
| `hello`                            | Hello                     |
| `Good to See You`                  | Good to See You           |
| `Leave`                            | Leave                     |
| `Please`                           | Please                    |
| `What Time is it`                  | What Time is it           |
| `Where Do You Live`                | Where Do You Live         |

The complete class mapping is available through the project files and `classes.json`.

---

# 🔬 Feature Extraction

Each video contains exactly **120 frames**.

For every frame, MediaPipe Hands and MediaPipe Pose are used to extract a fixed-length feature vector.

### Hand landmarks

Each hand contains:

```text
21 landmarks × 3 coordinates = 63 features
```

For two hands:

```text
Left hand  = 63
Right hand = 63
```

### Pose-based features

Four additional 3D Euclidean distances are calculated:

```text
Nose → Left Wrist
Nose → Right Wrist
Nose → Left Elbow
Nose → Right Elbow
```

Therefore:

```text
63 + 63 + 4 = 130 features / frame
```

The final representation of one video is:

```text
120 frames × 130 features
```

If a hand is not detected in a frame, its corresponding landmark vector is filled with zeros.

---

# NumPy Dataset Representation

After feature extraction, the videos are represented using NumPy arrays.

The generated structure is:

```text
BdSL_dynamic_npy/
│
├── train_csv/
│   ├── class_01/
│   │   ├── Student 1_1668322905/
│   │   │   ├── 001.npy
│   │   │   ├── 002.npy
│   │   │   ├── ...
│   │   │   └── 120.npy
│   │   └── ...
│   └── ...
│
├── validation_csv/
│   └── ...
│
└── test_csv/
    └── ...
```

Each video folder contains exactly:

```text
001.npy
002.npy
...
120.npy
```

Each NumPy file contains:

```text
130 values
```

Thus, every video becomes:

```text
(120, 130)
```

---

# Model Architecture

The main model receives an input sequence of:

```text
120 × 130
```

The architecture is:

```text
Input
(120, 130)
      │
      ▼
Conv1D
64 filters
kernel size = 3
      │
      ▼
Batch Normalization
      │
      ▼
Bidirectional LSTM
128 units
      │
      ▼
Batch Normalization
      │
      ▼
Dropout
      │
      ▼
Bidirectional LSTM
64 units
      │
      ▼
Batch Normalization
      │
      ▼
Dropout
      │
      ▼
Attention
Query = 64
Value = 64
      │
      ▼
GlobalAveragePooling1D
      │
      ▼
Dense
128 units
      │
      ▼
Batch Normalization
      │
      ▼
Dropout
      │
      ▼
Dense
64 units
      │
      ▼
Batch Normalization
      │
      ▼
Dropout
      │
      ▼
Softmax
40 classes
```

The dense layers use L2 regularization with:

```text
1e-5
```

---

# Training Configuration

The main experiment uses:

```text
Sequence length : 120 frames
Features/frame  : 130
Classes         : 40

Batch size      : 64
Maximum epochs  : 200
Learning rate   : 0.0005
Optimizer       : Adam
Loss            : Categorical Crossentropy
```

Training uses:

* Early stopping
* Model checkpointing
* Learning-rate reduction on plateau

### Callbacks

```text
EarlyStopping
    monitor = val_loss
    patience = 15

ModelCheckpoint
    monitor = val_accuracy
    save_best_only = True

ReduceLROnPlateau
    monitor = val_loss
    factor = 0.5
    patience = 10
    minimum learning rate = 1e-6
```

---

# Data Leakage Prevention

Feature scaling is performed using `StandardScaler`.

Importantly, the scaler is:

```text
Fit only on the training data
```

and then applied to:

```text
Training
Validation
Testing
```

This prevents information from the validation or test sets from being used during scaler fitting.

The fitted scaler is saved as:

```text
scaler.pkl
```

and reused by the real-time application.

---

# Reproducibility

Because the original video dataset cannot currently be distributed publicly, reproducibility is provided through the complete processing and training pipeline.

To reproduce the experiments when the dataset is available:

### Step 1 — Obtain the dataset

Obtain the original dataset through the appropriate dataset access procedure.

The dataset should be organized into student folders containing the 40 gesture classes.

---

### Step 2 — Create the student-independent split

Create:

```text
35 students → training
7 students  → validation
8 students  → testing
```

Each split should contain all 40 gesture classes.

---

### Step 3 — Merge the classes

Organize the videos as:

```text
BdSL_dynamic_merged_video/
├── train/
├── validation/
└── test/
```

with 40 class folders inside each split.

---

### Step 4 — Extract NumPy features

Run:

```text
numpy_extraction.ipynb
```

This extracts:

```text
63 left-hand features
63 right-hand features
4 pose-distance features
--------------------------------
130 features/frame
```

and produces 120 NumPy files per video.

---

### Step 5 — Train the main model

Open:

```text
120frames_130values.ipynb
```

The notebook loads the extracted sequences, standardizes the features using the training set, trains the model, evaluates it on the held-out test students, and saves the experiment outputs.

---

### Step 6 — Generate deployment artifacts

Run:

```text
final_dynamic.ipynb
```

The final pipeline saves the artifacts required by the Streamlit application:

```text
best_model.keras
classes.json
label_encoder.pkl
scaler.pkl
```

---

# Real-Time Recognition

The repository includes a Streamlit application:

```text
app.py
```

The application uses:

* Streamlit
* Streamlit-WebRTC
* OpenCV
* MediaPipe
* TensorFlow/Keras
* NumPy
* Scikit-learn

The application processes webcam frames and constructs the same 130-dimensional feature representation used during training.

The model operates on a temporal window of:

```text
120 frames
```

The processing resolution is:

```text
640 × 480
```

while the displayed camera view is kept visually smaller.

The application also reports runtime pipeline statistics such as:

* Total frames
* Average FPS
* Average latency
* Peak RAM usage

---

# Running the Application

After placing the deployment artifacts in the same directory as `app.py`:

```text
best_model.keras
classes.json
label_encoder.pkl
scaler.pkl
app.py
```

install the required Python packages and run:

```bash
streamlit run app.py
```

Then open the Streamlit URL provided in the terminal and allow webcam access.

---

# Ablation Studies

Additional experiments are included for researchers interested in examining the effect of temporal length and feature selection.

### Temporal-length experiments

```text
60frames_130values.ipynb
80frames_130values.ipynb
120frames_130values.ipynb
```

These investigate different temporal sequence lengths while retaining the 130-feature representation.

### Landmark ablation

```text
120frames_ablation_study.ipynb
```

This notebook contains experiments involving different landmark configurations, including:

* All landmarks + wrist information
* Wrist-only
* Elbow-only
* Hand-landmark-only
* Elbow + wrist

These experiments are provided for researchers who want to investigate the contribution of different body/hand landmark groups.

---

# Recommended Project Structure

A complete working directory can be organized as:

```text
dynamic-bangla-sign-language/
│
├── README.md
├── app.py
│
├── numpy_extraction.ipynb
├── 120frames_130values.ipynb
├── final_dynamic.ipynb
│
├── 60frames_130values.ipynb
├── 80frames_130values.ipynb
├── 120frames_ablation_study.ipynb
│
├── best_model.keras
├── classes.json
├── label_encoder.pkl
└── scaler.pkl
```

The original dataset itself is **not included** in the repository.

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for the complete license text.

---

# Contact

For questions, discussions, or research-related inquiries:

**Md Mehedi Hasan**

Email: `mehedi3128.mhd@gmail.com`

---

# Acknowledgement

We acknowledge the **Dhaka University Centennial Research Grant** associated with the original dynamic video dataset used in this research.

The dataset is currently not publicly redistributed in this repository. Dataset availability information will be updated here if and when public access becomes possible.

---

# Citation

If you use the code or methodology from this repository in your research, please cite the corresponding research work associated with this project.

A formal citation entry can be added here once the associated publication details are finalized.

