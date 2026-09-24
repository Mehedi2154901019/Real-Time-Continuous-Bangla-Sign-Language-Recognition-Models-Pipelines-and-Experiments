# BdSL49_Updated_2026 Bangla Sign Language Recognition with MobileNetV2_first50LayersFrozen_CrossValidation

A reproducible deep-learning pipeline for **Bangla Sign Language (BdSL) static two-hand sign classification** using **MobileNetV2**, stratified cross-validation, data augmentation, transfer learning, and a real-time Streamlit inference pipeline.

This repository contains the code and configuration required to:

1. Download and prepare the **BdSL49_Updated_2026** dataset.
2. Train a MobileNetV2-based Bangla Sign Language classifier.
3. Perform **3-fold stratified cross-validation**.
4. Train a final model using the combined training and validation data.
5. Evaluate the final model on an independent test set.
6. Automatically generate model artifacts, metrics, class mappings, and a confusion matrix.
7. Run the trained model through a **real-time webcam-based Streamlit application**.
8. Monitor real-time inference performance, including FPS, latency, and RAM consumption.

The repository is intended both for **research reproducibility** and for **future researchers who wish to extend, compare, or deploy the proposed approach**.

---

# Overview

This project investigates **static Bangla Sign Language recognition** using a transfer-learning-based convolutional neural network.

The classification architecture is based on **MobileNetV2 pretrained on ImageNet**. The first 50 layers of the MobileNetV2 backbone are frozen, while the remaining layers are fine-tuned together with a lightweight custom classification head.

The model receives RGB images resized to:

```text
224 × 224 × 3
```

The classification head consists of:

```text
MobileNetV2 backbone
        ↓
Global Average Pooling
        ↓
Dense(256, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(49, Softmax)
```

The training pipeline additionally performs lightweight online augmentation using:

* Random horizontal flipping
* Random brightness adjustment

The final model is evaluated on an independent test split, while the training/validation data are pooled for stratified cross-validation.

A separate Streamlit application provides real-time webcam inference using:

* OpenCV
* MediaPipe Hands
* TensorFlow/Keras
* Streamlit
* psutil

The application detects one or two hands, constructs a combined bounding box, crops the detected hand region, performs classification, and displays the predicted sign and confidence together with runtime telemetry.

---

# Repository Structure

The recommended repository layout is:

```text
BdSL49-MobileNetV2/
│
├── README.md
├── requirements.txt
│
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
├── app.py
│
├── bdsl49_updated_2026/
│   ├── train/
│   │   ├── class_01/
│   │   ├── class_02/
│   │   ├── ...
│   │   └── class_49/
│   │
│   ├── val/
│   │   ├── class_01/
│   │   ├── class_02/
│   │   ├── ...
│   │   └── class_49/
│   │
│   └── test/
│       ├── class_01/
│       ├── class_02/
│       ├── ...
│       └── class_49/
│
└── bdsl49_results/
    ├── classes.json
    ├── class_indices.json
    ├── config.json
    ├── cv_results.json
    ├── test_results.json
    ├── final_model.keras
    ├── best_fold_1.keras
    ├── best_fold_2.keras
    ├── best_fold_3.keras
    ├── model_summary_fold1.txt
    ├── model_summary_fold2.txt
    ├── model_summary_fold3.txt
    ├── final_model_summary.txt
    └── confusion_matrix.png
```

The dataset and generated model artifacts do not have to be committed to GitHub if their size exceeds GitHub's recommended limits. The dataset should instead be obtained from Zenodo, while trained artifacts can be provided separately when appropriate.

---

# Dataset

## BdSL49_Updated_2026

The experiments use the **BdSL49_Updated_2026** dataset containing **49 Bangla Sign Language classes**.

The dataset is publicly archived on Zenodo.

**DOI:** `10.5281/zenodo.22930407`

Dataset page:

https://doi.org/10.5281/zenodo.22930407

Please download the dataset from Zenodo before running the training notebook.

## Download Procedure

1. Open the Zenodo dataset page.
2. Download the dataset ZIP archive.
3. Extract the ZIP archive.
4. Locate the directory:

```text
BdSL49_Updated_2026
```

5. Rename it to:

```text
bdsl49_updated_2026
```

if necessary, so that it matches the directory expected by the training notebook.
6. Place the directory in the same directory as the training notebook.

The resulting structure should be:

```text
project/
│
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
├── requirements.txt
│
└── bdsl49_updated_2026/
    ├── train/
    ├── val/
    └── test/
```

Each split contains the 49 sign classes.

---

# Dataset Organization

The training code automatically discovers the class names from:

```text
bdsl49_updated_2026/train/
```

The expected structure is:

```text
bdsl49_updated_2026/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
│   └── class_49/
│
├── val/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
│   └── class_49/
│
└── test/
    ├── class_1/
    ├── class_2/
    ├── ...
    └── class_49/
```

The exact directory names inside the splits should remain consistent.

The notebook sorts the discovered class names and generates a corresponding integer mapping.

---

# Environment and Installation

## 1. Clone the repository

```bash
git clone <GITHUB-REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>
```

## 2. Create a virtual environment

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The supplied `requirements.txt` contains the major dependencies required by both the training and real-time inference pipelines.

---

# Requirements

The primary software dependencies are:

```text
tensorflow>=2.12.0
keras>=2.12.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.20.0
opencv-python>=4.7.0
mediapipe>=0.10.0
psutil>=5.9.0
```

For research reproducibility, future users should record the exact Python version, TensorFlow version, CUDA version, GPU model, and operating system used for each experiment.

---

# Recommended Hardware

The training notebook is computationally intensive because it performs transfer learning and 3-fold cross-validation.

For the reported experiments, a GPU environment is recommended.

```text
Platform: NVIDIA GPU environment
GPU: NVIDIA T4 or better
Recommended: Kaggle GPU environment
Preferred: T4 × 2 when available
```

A CPU-only environment may work but will require substantially longer training time.

---

# Training

The main training file is:

```text
mobilentv2_First50FrozenLayers_cross_validation.ipynb
```

The filename refers to the configuration in which the **first 50 MobileNetV2 layers are frozen**.

Before running the notebook, ensure that:

```text
requirements.txt
mobilentv2_First50FrozenLayers_cross_validation.ipynb
bdsl49_updated_2026/
```

are available in the same working directory.

## Running on Kaggle

Kaggle can be used for convenient GPU-based reproduction.

Upload:

```text
mobilentv2_First50FrozenLayers_cross_validation.ipynb
```

to a Kaggle notebook and make the BdSL49 dataset available.

The notebook expects:

```text
bdsl49_updated_2026/
    train/
    val/
    test/
```

Select a GPU accelerator and run the notebook from beginning to end.

The notebook automatically:

1. Discovers the 49 classes.
2. Creates class-to-index mappings.
3. Loads training and validation image paths.
4. Combines the training and validation paths for cross-validation.
5. Creates the MobileNetV2 architecture.
6. Freezes the first 50 backbone layers.
7. Performs 3-fold stratified cross-validation.
8. Saves the best model from each fold.
9. Calculates cross-validation metrics.
10. Trains a final model.
11. Saves the final model as `final_model.keras`.
12. Evaluates the final model on the independent test set.
13. Generates the confusion matrix.
14. Saves the final evaluation metrics and configuration files.

---

# Model Architecture

The model uses ImageNet-pretrained MobileNetV2 without its original classification head.

```python
MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)
```

The first 50 layers are frozen:

```python
for layer in base_model.layers[:50]:
    layer.trainable = False
```

The custom classification head is:

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense(256, ReLU)
     ↓
Dropout(0.3)
     ↓
Dense(49, Softmax)
```

The optimizer is:

```text
Adam
```

with:

```text
Learning rate = 1 × 10⁻⁴
```

The loss function is:

```text
Sparse Categorical Crossentropy
```

and the primary training metric is:

```text
Accuracy
```

---

# Image Preprocessing

The training pipeline explicitly uses the official MobileNetV2 preprocessing function:

```python
tensorflow.keras.applications.mobilenet_v2.preprocess_input
```

Images are resized to:

```text
224 × 224
```

The MobileNetV2 preprocessing convention scales the input pixels to the range expected by the pretrained MobileNetV2 network.

Before preprocessing, training images may receive lightweight online augmentation:

```text
Random horizontal flip
Random brightness adjustment
```

The test pipeline does not apply augmentation.

For reproducible deployment, the real-time Streamlit application uses the **same MobileNetV2 preprocessing convention as the training pipeline**.

---

# Cross-Validation

The training notebook uses:

```text
3-fold Stratified Cross-Validation
```

implemented with:

```python
StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

The stratification preserves the class distribution across the folds.

For each fold:

1. A new MobileNetV2 model is initialized.
2. The first 50 backbone layers are frozen.
3. Training augmentation is enabled.
4. Validation augmentation is disabled.
5. The model is trained for up to 20 epochs.
6. Early stopping monitors validation accuracy.
7. The best model weights are saved.

The relevant callbacks are:

```text
ModelCheckpoint
EarlyStopping
```

with validation accuracy used as the monitoring criterion.

---

# Generated Training Artifacts

The notebook creates:

```text
bdsl49_results/
```

automatically.

## Class metadata

```text
classes.json
class_indices.json
config.json
```

These files contain:

* Discovered class names
* Class-to-index mappings
* Index-to-class mappings
* Image size
* Number of classes
* Preprocessing configuration

## Cross-validation results

```text
cv_results.json
```

contains aggregated cross-validation metrics.

## Fold models

```text
best_fold_1.keras
best_fold_2.keras
best_fold_3.keras
```

contain the best checkpoint from each cross-validation fold.

## Final deployment model

```text
final_model.keras
```

contains the model trained using the complete pooled training/validation data and is the model intended for final evaluation and Streamlit deployment.

## Model summaries

```text
model_summary_fold1.txt
model_summary_fold2.txt
model_summary_fold3.txt
final_model_summary.txt
```

contain Keras architecture summaries.

## Test results

```text
test_results.json
```

contains the final independent test metrics.

## Confusion matrix

```text
confusion_matrix.png
```

contains the confusion matrix generated from the independent test split.

---

# Independent Test Evaluation

After cross-validation, the notebook trains a final model using the pooled training and validation data.

The independent test set is then evaluated separately.

The test set is loaded using:

```python
tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

with:

```text
shuffle=False
```

This ensures that predictions can be directly compared with the corresponding test labels.

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-score

The current implementation uses macro averaging for the calculated test precision, recall, and F1 metrics.

---

# Real-Time Streamlit Pipeline

The real-time application is:

```text
app.py
```

The application combines:

```text
OpenCV
    +
MediaPipe Hands
    +
MobileNetV2
    +
Streamlit
    +
psutil
```

The processing sequence is:

```text
Webcam
   ↓
Frame Capture
   ↓
Horizontal Flip
   ↓
MediaPipe Hand Detection
   ↓
Combined Hand Bounding Box
   ↓
Padding
   ↓
Hand Crop
   ↓
Resize to 224 × 224
   ↓
MobileNetV2 Preprocessing
   ↓
Model Inference
   ↓
Predicted Class + Confidence
   ↓
Streamlit Visualization
   ↓
Performance Telemetry
```

---

# Pipeline Requirements

After training is complete, keep the final deployment artifacts in the same directory as `app.py`.

The application requires, at minimum:

```text
app.py
final_model.keras
classes.json
class_indices.json
config.json
```

A recommended project layout is:

```text
project/
│
├── app.py
├── requirements.txt
│
├── final_model.keras
├── classes.json
├── class_indices.json
├── config.json
│
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
│
├── bdsl49_updated_2026/
│   ├── train/
│   ├── val/
│   └── test/
│
└── bdsl49_results/
```

If the model artifacts are stored inside `bdsl49_results/`, either copy the required files to the application directory or update the corresponding paths in `app.py`.

---

# Running the Real-Time Application

From the directory containing `app.py`, execute:

```bash
streamlit run app.py
```

Streamlit will launch the local application.

Open the address displayed by Streamlit in a web browser.

Enable the webcam to start real-time inference.

---

# Webcam Usage

Once the webcam is enabled:

1. Allow browser/system access to the webcam.
2. Position one or two hands in front of the camera.
3. Perform a sign from the BdSL49 class set.
4. MediaPipe detects the hand landmarks.
5. A combined hand bounding box is created.
6. The detected region is cropped.
7. The cropped image is passed to the classifier.
8. The predicted sign and confidence are displayed.

---

# Real-Time Performance Monitoring

The Streamlit application displays runtime telemetry including:

```text
Live FPS
Latency
Current RAM
Peak RAM
```

When the webcam session ends, the application additionally reports:

```text
Total Frames Processed
True Average Latency
True Average Operational FPS
Peak RAM Consumption
```

The pipeline measures the complete frame-processing loop:

```text
Frame capture
+
Hand detection
+
Cropping
+
Preprocessing
+
Model inference
+
Visualization-related processing
```

Therefore, the reported latency should be interpreted as **end-to-end pipeline latency**, rather than neural-network inference time alone.

---

# Reported Results

The following results correspond to the reported **Static Two Hand, MobileNetV2** experiment.

| Model           | Signer Dependency | Split                   | Scaling / Preprocessing        | Cross-Validation | Architecture                        | Features | Augmentation | Total Parameters | Model Size | Train Accuracy | Validation Accuracy | Test Accuracy | Precision | Recall | Weighted F1-Score | Peak RAM (avg.) | FPS (avg.) | End-to-End Latency (avg.) |
| --------------- | ----------------- | ----------------------- | ------------------------------ | ---------------- | ----------------------------------- | -------- | ------------ | ---------------: | ---------- | -------------: | ------------------: | ------------: | --------: | -----: | ----------------: | --------------: | ---------: | ------------------------: |
| Static Two Hand | Dependent         | Train, Test, Validation | MobileNetV2 `preprocess_input` | Yes              | MobileNetV2, first 50 layers frozen | N/A      | Yes          |        2,598,513 | 28.64 MB   |         99.57% |              97.09% |        96.33% |    96.93% | 96.63% |            96.54% |        848.1 MB |      5 FPS |                  231.3 ms |

> **Important:** The values in this table are the reported experimental results supplied with this repository. When reproducing the experiment, hardware, software versions, random seeds, execution environment, and preprocessing consistency should be documented because these factors can affect measured performance.

---

# Understanding the Reported Metrics

## Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

## Precision

Precision measures the proportion of predicted samples for a class that are actually members of that class.

## Recall

Recall measures the proportion of samples belonging to a class that are correctly identified.

## F1-score

F1-score combines precision and recall through their harmonic mean.

## FPS

FPS represents the number of frames processed per second during the real-time pipeline.

## End-to-End Latency

End-to-end latency represents the elapsed time for the complete frame-processing loop rather than only the neural-network forward pass.

## Peak RAM

Peak RAM represents the maximum observed resident memory usage of the running process during the webcam session.

---

# Reproducibility Notes

Several details should be recorded when reproducing or extending this experiment.

Recommended metadata include:

```text
Python version
TensorFlow version
Keras version
CUDA version
cuDNN version
GPU model
GPU memory
CPU model
Operating system
RAM
Dataset version
Git commit
Training random seed
Execution platform
```

For example:

```text
Python:
TensorFlow:
Keras:
CUDA:
cuDNN:
GPU:
GPU Memory:
CPU:
RAM:
OS:
Dataset:
Git Commit:
```

This information is particularly important when comparing real-time FPS, latency, and RAM consumption.

---

# Important Implementation Details

## Number of Classes

The current dataset contains:

```text
49 classes
```

The notebook discovers the class names from the training directory rather than hard-coding them.

## Input Resolution

```text
224 × 224 × 3
```

RGB input.

## Frozen Layers

```text
First 50 MobileNetV2 layers
```

## Transfer Learning

The MobileNetV2 backbone uses ImageNet-pretrained weights.

The classification head is newly initialized for the 49-class BdSL classification problem.

## Data Augmentation

Training augmentation consists of:

```text
Random horizontal flip
Random brightness adjustment
```

No augmentation is applied during independent test evaluation.

---

# Research Extension

Researchers can use this repository as a baseline for further experiments.

## Backbone Comparison

Potential alternatives include:

```text
MobileNetV3
EfficientNet
EfficientNetV2
ResNet50
DenseNet
ConvNeXt
Xception
Vision Transformers
```

## Fine-Tuning Strategies

Compare:

```text
All layers frozen
First 20 layers frozen
First 50 layers frozen
First 75 layers frozen
First 100 layers frozen
No layers frozen
```

## Augmentation Experiments

Investigate:

```text
Rotation
Translation
Zoom
Contrast
Brightness
Horizontal flipping
Random cropping
MixUp
CutMix
```

## Input Resolution

Compare:

```text
160 × 160
192 × 192
224 × 224
256 × 256
```

## Deployment Optimization

Investigate:

```text
TensorFlow Lite
TensorRT
ONNX
Quantization
Pruning
Knowledge distillation
```

## Real-Time Performance

Measure separately:

```text
Hand detection latency
Image preprocessing latency
Model inference latency
Post-processing latency
End-to-end latency
```

This can provide a more detailed understanding of the computational bottlenecks in the complete real-time system.

---

# Reproducing the Complete Workflow

The complete recommended workflow is:

```text
                         ┌──────────────────────┐
                         │      Zenodo Dataset  │
                         │     BdSL49_Updated   │
                         │         _2026        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Extract Dataset    │
                         │ train / val / test   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Training Notebook             │
                    │ MobileNetV2                   │
                    │ First 50 Layers Frozen        │
                    │ 3-Fold Stratified CV          │
                    │ Data Augmentation             │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │      Model Artifacts          │
                    │ best_fold_1.keras              │
                    │ best_fold_2.keras              │
                    │ best_fold_3.keras              │
                    │ final_model.keras              │
                    │ classes.json                   │
                    │ metrics / confusion matrix    │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │       Independent Test        │
                    │ Accuracy / Precision / Recall │
                    │ F1 / Confusion Matrix         │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │       Streamlit + Webcam      │
                    │ OpenCV + MediaPipe + Keras    │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │       Real-Time Output        │
                    │ Sign + Confidence             │
                    │ FPS + Latency + Peak RAM     │
                    └───────────────────────────────┘
```

---

# Kaggle Reproduction

A publicly available Kaggle notebook is provided as a complementary reproduction reference:

**BdSL49 Updated 2026 — MobileNetV2 Cross Validation**

https://www.kaggle.com/code/hassan0008jhh/bdsl49-updated-2026-mobilenetv2-crossvalidation

The Kaggle notebook provides a reference for reproducing the training workflow in a GPU environment.

The supplied GitHub `.ipynb` file is the source implementation and **should not be assumed to contain the original executed output cells**. Users should execute the notebook themselves to reproduce the experiment.

---

# Recommended File Handling for GitHub

Because datasets and trained neural-network artifacts can be large, the recommended repository strategy is:

```text
GitHub
│
├── Source code
├── Training notebook
├── requirements.txt
├── README.md
└── Configuration/documentation
```

and:

```text
Zenodo
└── BdSL49_Updated_2026 dataset
```

Large model files such as:

```text
*.keras
```

may be distributed separately or through an appropriate Git LFS/model-hosting mechanism if they exceed normal GitHub repository limits.

Researchers should not need to download the complete dataset from GitHub.

Instead:

```text
GitHub → Code
Zenodo → Dataset
Kaggle → Reproduction reference
```

---

# Citation

If you use the dataset, please cite the corresponding Zenodo record:

```text
BdSL49_Updated_2026
DOI: 10.5281/zenodo.22930407
```

Dataset DOI:

https://doi.org/10.5281/zenodo.22930407

If this repository or its implementation contributes to your research, please also cite the associated manuscript/publication when the bibliographic information becomes available.

---

# Acknowledgements

This project uses the following open-source technologies:

* TensorFlow / Keras
* MobileNetV2
* scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* MediaPipe
* Streamlit
* psutil

The project also relies on the publicly distributed BdSL49 dataset hosted on Zenodo.

---

# License

Please add the appropriate license for the source code and dataset according to the rights under which each component is distributed.

The dataset license and the source-code license should be treated separately where applicable.

Before redistributing the dataset or trained model files, verify the licensing and redistribution terms of the corresponding dataset and pretrained components.

---

# Contact and Research Use

For questions, reproducibility issues, or problems running the code, please contact:

**Email:** `mehedi3128.mhd@gmail.com`

This repository is intended to support:

* Reproducible research
* Academic experimentation
* Bangla Sign Language recognition research
* Transfer-learning experiments
* Real-time sign language recognition
* Model comparison
* Deployment experiments
* Future dataset and architecture extensions

Researchers are encouraged to document modifications to the training pipeline and report the corresponding hardware, software versions, dataset version, and experimental configuration when publishing new results based on this implementation.

---

# Quick Start

For experienced users:

```bash
# 1. Clone repository
git clone <GITHUB-REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download BdSL49_Updated_2026 from Zenodo
#    https://doi.org/10.5281/zenodo.22930407

# 4. Place/extract the dataset as:
#
#    bdsl49_updated_2026/
#        train/
#        val/
#        test/

# 5. Run:
#    mobilentv2_First50FrozenLayers_cross_validation.ipynb
#
# Recommended: Kaggle GPU / NVIDIA T4 environment

# 6. After training, ensure the final deployment artifacts
#    are available beside app.py:
#
#    final_model.keras
#    classes.json
#    class_indices.json
#    config.json

# 7. Launch the real-time application
streamlit run app.py
```

Then open the Streamlit application, enable the webcam, perform signs from the BdSL49 class set, and observe the predicted class, confidence, FPS, latency, and RAM usage.

---

# Project Summary

```text
Dataset:
BdSL49_Updated_2026

Classes:
49

Task:
Static Bangla Sign Language Classification

Model:
MobileNetV2

Transfer Learning:
ImageNet

Frozen Layers:
First 50 layers

Input:
224 × 224 × 3

Preprocessing:
MobileNetV2 preprocess_input

Augmentation:
Horizontal Flip + Random Brightness

Cross-Validation:
3-Fold Stratified

Optimizer:
Adam

Learning Rate:
1e-4

Loss:
Sparse Categorical Crossentropy

Deployment:
Streamlit + OpenCV + MediaPipe

Deployment Model:
final_model.keras

Reported Test Accuracy:
96.33%

Reported Average FPS:
5

Reported Average End-to-End Latency:
231.3 ms

Reported Average Peak RAM:
848.1 MB

Dataset DOI:
10.5281/zenodo.22930407

Contact:
mehedi3128.mhd@gmail.com
```

