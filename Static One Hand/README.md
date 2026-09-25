# BdSL47_Updated_2026 — Dataset, MLP Training, Ablation Study & Real-Time Pipeline

This repository contains the training, evaluation, ablation-study, and real-time inference materials for **BdSL47_Updated_2026**, a static one-handed Bangla Sign Language (BdSL) dataset based on **MediaPipe hand landmarks** and a **Multi-Layer Perceptron (MLP)** classifier.

The materials are intended to support **reproducibility, baseline comparison, and future research** on static one-handed Bangla Sign Language recognition.

---

## 1. Dataset

The complete dataset is available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.22937669

The Zenodo archive contains:

- `BdSL47_Updated_2026/` — original dataset organization, including the images used to generate the landmark CSV files.
- `BdSL47_Updated_2026_csv/` — ready-to-use CSV dataset containing:
  - `train.csv`
  - `validation.csv`
  - `test.csv`

### Which one should I download?

If you only want to reproduce the model training, use:

```text
BdSL47_Updated_2026_csv/
````

If you want to inspect the original images and understand the data before landmark extraction, use:

```text
BdSL47_Updated_2026/
```

The CSV files contain **21 MediaPipe hand landmarks × 3 coordinates (X, Y, Z) = 63 features** per sample.

---

## 2. Repository Contents

```text
.
├── model.ipynb
├── ablation.ipynb
├── app.py
├── final_model.keras
├── scaler.pkl
├── pipeline_config.json
├── NotoSansBengali-Regular.ttf
└── README.md
```

### `model.ipynb`

Production training notebook for the final baseline model.

It:

1. Loads the train/validation/test CSV files.
2. Extracts the 63 MediaPipe landmark features.
3. Applies landmark normalization.
4. Applies `StandardScaler` fitted only on the training data.
5. Trains the MLP classifier.
6. Evaluates the model on the test set.
7. Saves the model and preprocessing artifacts.

### `ablation.ipynb`

Contains the ablation experiments used to investigate different feature-processing and model configurations, including landmark normalization, rotation handling, scaling, translation, augmentation, and related configurations.

It is provided primarily for researchers who want to inspect or extend the experimental analysis.

### `app.py`

Real-time one-handed BdSL recognition pipeline using:

```text
Webcam
  ↓
MediaPipe Hands
  ↓
21 × XYZ landmarks
  ↓
Translation normalization
  ↓
Scale normalization
  ↓
63-dimensional feature vector
  ↓
StandardScaler
  ↓
MLP
  ↓
BdSL47 prediction
```

The application also reports:

* Prediction confidence
* FPS
* End-to-end latency
* Peak RAM usage
* Number of detected hands

Two-hand input is treated as unsupported because the dataset contains static **one-handed** signs.

### Saved artifacts

The repository also includes the artifacts required by the inference pipeline:

```text
final_model.keras
scaler.pkl
pipeline_config.json
NotoSansBengali-Regular.ttf
```

`pipeline_config.json` stores the preprocessing configuration so that the same normalization settings used during training can be reproduced during inference.

---

## 3. Model Architecture

The final baseline is an MLP operating on 63 normalized and standardized landmark features.

```text
Input: 63 features

Dense(128)
Batch Normalization
LeakyReLU

Dense(128)
Batch Normalization
LeakyReLU

Dense(64)
Batch Normalization
LeakyReLU
Dropout(0.33)

Dense(32)
Batch Normalization
LeakyReLU
Dropout(0.25)

Dense(47, Softmax)
```

Training uses:

* Adam optimizer
* Learning rate: `0.001`
* Sparse categorical cross-entropy
* Batch size: `64`
* Maximum epochs: `600`
* Early stopping on validation loss
* Random seed: `42`

The model predicts **47 classes**.

---

## 4. Class Mapping

The labels in the CSV files are encoded as integers `0–46`.

* `0–9` correspond to the ten digit classes.
* `10–46` correspond to the remaining 37 Bangla Sign Language classes.

The exact mapping used by the real-time application is defined in `app.py` and should be kept consistent with the training labels.

---

## 5. Reproducing the Training

### Step 1 — Download the dataset

Download the Zenodo dataset:

[https://doi.org/10.5281/zenodo.22937669](https://doi.org/10.5281/zenodo.22937669)

Extract the archive.

For model training, locate:

```text
BdSL47_Updated_2026_csv/
```

containing:

```text
train.csv
validation.csv
test.csv
```

### Step 2 — Clone/download this repository

Place the repository and dataset so that the training notebook can access the CSV directory.

For example:

```text
project/
├── model.ipynb
├── ablation.ipynb
├── app.py
├── final_model.keras
├── scaler.pkl
├── pipeline_config.json
├── NotoSansBengali-Regular.ttf
│
└── BdSL47_Updated_2026/
    └── bdsl47_csv/
        ├── train.csv
        ├── validation.csv
        └── test.csv
```

If your directory structure is different, update `DATA_DIR` in `model.ipynb`.

### Step 3 — Install dependencies

A Python environment containing the following packages is required:

```bash
pip install numpy pandas matplotlib scikit-learn joblib tensorflow opencv-python mediapipe pillow psutil
```

### Step 4 — Run the training notebook

Open:

```text
model.ipynb
```

Update `DATA_DIR` if necessary and run the notebook from beginning to end.

The notebook trains the MLP and saves:

```text
final_model.keras
scaler.pkl
pipeline_config.json
```

The model filename may differ from the final artifact name depending on the version of the notebook. For `app.py`, the expected model filename is:

```text
final_model.keras
```

---

## 6. Running Real-Time Recognition

For the simplest setup, keep the following files in the same directory:

```text
.
├── app.py
├── final_model.keras
├── scaler.pkl
├── pipeline_config.json
└── NotoSansBengali-Regular.ttf
```

Update the `DATA_DIR` variable near the beginning of `app.py` so that it points to this directory.

For example:

```python
DATA_DIR = r"C:\path\to\BdSL47_project"
```

Then run:

```bash
python app.py
```

A webcam window will open.

### Controls

Press:

```text
Q
```

to exit the application.

When one hand is detected, the system extracts the 21 MediaPipe landmarks, applies the same preprocessing used during training, and performs classification using the saved MLP model.

---

## 7. Important Reproducibility Note

The preprocessing used during inference must match the preprocessing used during training.

The current final configuration is:

```text
Translation normalization: ON
Scale normalization:       ON
Rotation normalization:    OFF
StandardScaler:             ON
Features:                  21 × 3 = 63
Classes:                   47
```

`pipeline_config.json` is used by `app.py` to verify these settings.

Do not replace `scaler.pkl` or modify the normalization configuration unless the model is retrained with the corresponding preprocessing.

---

## 8. Dataset Split and Signer Consideration

The original BdSL47 dataset contained a limited number of signers. The updated dataset and experimental setup are provided with the goal of improving the usefulness of signer-independent evaluation and providing a reproducible baseline.

When extending this work, researchers are encouraged to report:

* Number of signers
* Signer-independent train/validation/test protocol
* Class distribution
* Feature preprocessing
* Model architecture
* Test-set performance
* Real-time performance where applicable

The ablation notebook is included to make the feature-processing and architecture experiments inspectable and extensible.

---

## 9. Using This Work

If you use the **code or materials from this repository**, please cite/reference this repository.

If you use the **BdSL47_Updated_2026 dataset**, please cite the corresponding Zenodo dataset:

[https://doi.org/10.5281/zenodo.22937669](https://doi.org/10.5281/zenodo.22937669)

Please distinguish between the licenses of the repository code and the dataset when redistributing or building upon this work.

### License

The code and materials in this repository are released under the **MIT License**.

The dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, subject to the terms specified in the Zenodo dataset record.

Please refer to the respective license terms before redistributing the code or dataset.

---

## 10. Contact

For questions, reproducibility issues, or problems running the pipeline:

Email: [mehedi3128.mhd@gmail.com](mailto:mehedi3128.mhd@gmail.com)

If you encounter an issue, please include your operating system, Python/TensorFlow version, error message, and the step at which the problem occurred.

---

## Acknowledgement

This repository is provided as a reproducible research baseline for **static one-handed Bangla Sign Language recognition** using MediaPipe hand landmarks and an MLP classifier. Researchers are welcome to use the provided training, ablation, and inference materials for further experimentation and comparison, subject to the applicable licenses.

```
```
