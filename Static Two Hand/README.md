# BdSL49_Updated_2026: MobileNetV2 Cross-Validation & Real-Time Recognition

A reproducible deep-learning pipeline for **static two-hand Bangla Sign Language (BdSL) recognition** using **ImageNet-pretrained MobileNetV2**, 3-fold stratified cross-validation, lightweight data augmentation, and a real-time Streamlit inference pipeline.

The repository is intended for **research reproducibility, baseline comparison, and future extension**.

---

## 1. Dataset

The experiments use the **BdSL49_Updated_2026** dataset containing **49 Bangla Sign Language classes**.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.22930407

Download the dataset from Zenodo before running the training notebook.

After extraction, the required structure is:

```text
project/
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
├── app.py
├── requirements.txt
│
└── bdsl49_updated_2026/
    ├── train/
    │   ├── class_01/
    │   ├── class_02/
    │   └── ...
    │
    ├── val/
    │   ├── class_01/
    │   ├── class_02/
    │   └── ...
    │
    └── test/
        ├── class_01/
        ├── class_02/
        └── ...
````

The `train`, `val`, and `test` directories should each contain the 49 class folders.

The notebook discovers and sorts the class names automatically.

---

## 2. Repository Contents

```text
.
├── README.md
├── requirements.txt
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
├── app.py
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

Large datasets and model files do not need to be committed to GitHub. The dataset is distributed through Zenodo, while the final trained model is also available through Zenodo as described below.

---

## 3. Model

The classifier uses **MobileNetV2 pretrained on ImageNet** with the first 50 backbone layers frozen.

Input:

```text
224 × 224 × 3 RGB image
```

Architecture:

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

Training configuration:

```text
Optimizer:              Adam
Learning rate:          1e-4
Loss:                   Sparse Categorical Crossentropy
Maximum epochs:         20
Cross-validation:       3-fold Stratified
Frozen backbone:        First 50 MobileNetV2 layers
```

---

## 4. Image Preprocessing and Augmentation

Images are resized to:

```text
224 × 224
```

The training pipeline uses the official MobileNetV2 preprocessing:

```python
tensorflow.keras.applications.mobilenet_v2.preprocess_input
```

Training augmentation consists of:

```text
Random horizontal flip
Random brightness adjustment
```

No augmentation is applied during independent test evaluation.

---

## 5. Installation

### Clone the repository

```bash
git clone <GITHUB-REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>
```

### Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies are:

```text
TensorFlow / Keras
scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
OpenCV
MediaPipe
Streamlit
psutil
```

For reproducibility, it is recommended to record the Python version, TensorFlow/Keras version, operating system, GPU, CUDA/cuDNN versions, and dataset version.

---

## 6. Reproducing Training

The main training notebook is:

```text
mobilentv2_First50FrozenLayers_cross_validation.ipynb
```

Before running it, ensure that the notebook can access:

```text
bdsl49_updated_2026/
    train/
    val/
    test/
```

The notebook performs the following workflow:

```text
Dataset
   ↓
Load train / validation / test images
   ↓
Combine train + validation for cross-validation
   ↓
3-fold stratified cross-validation
   ↓
MobileNetV2 transfer learning
   ↓
First 50 layers frozen
   ↓
Data augmentation
   ↓
Best model saved for each fold
   ↓
Final model trained on pooled train + validation data
   ↓
Independent test evaluation
   ↓
Metrics + confusion matrix + model artifacts
```

### Cross-validation

The training/validation data are combined for:

```text
3-fold StratifiedKFold
shuffle=True
random_state=42
```

Each fold starts with a new MobileNetV2 model.

The best checkpoint from each fold is saved as:

```text
best_fold_1.keras
best_fold_2.keras
best_fold_3.keras
```

---

## 7. Final Model and Evaluation

After cross-validation, the final model is trained using the combined training and validation data.

The independent test set is then evaluated separately.

The pipeline reports:

* Accuracy
* Macro precision
* Macro recall
* F1-score
* Confusion matrix

The final deployment model is:

```text
final_model.keras
```

A copy of the final model is available through Zenodo:

[https://doi.org/10.5281/zenodo.22944642](https://doi.org/10.5281/zenodo.22944642)

---

## 8. Generated Artifacts

The training notebook generates:

```text
bdsl49_results/
```

### Configuration and class information

```text
classes.json
class_indices.json
config.json
```

These contain the class names, class-index mappings, image size, number of classes, and relevant preprocessing configuration.

### Cross-validation

```text
cv_results.json
```

Contains the recorded cross-validation metrics.

### Fold models

```text
best_fold_1.keras
best_fold_2.keras
best_fold_3.keras
```

### Final model

```text
final_model.keras
```

This is the model intended for final evaluation and real-time deployment.

### Evaluation

```text
test_results.json
confusion_matrix.png
```

### Architecture summaries

```text
model_summary_fold1.txt
model_summary_fold2.txt
model_summary_fold3.txt
final_model_summary.txt
```

---

## 9. Real-Time Streamlit Pipeline

The real-time application is:

```text
app.py
```

It combines:

```text
OpenCV
MediaPipe Hands
TensorFlow/Keras
MobileNetV2
Streamlit
psutil
```

The inference pipeline is:

```text
Webcam
   ↓
Frame capture
   ↓
Horizontal flip
   ↓
MediaPipe hand detection
   ↓
Combined hand bounding box
   ↓
Crop + padding
   ↓
Resize to 224 × 224
   ↓
MobileNetV2 preprocessing
   ↓
Model prediction
   ↓
Class + confidence
   ↓
Streamlit display
```

The application supports detection of one or two hands and creates a combined bounding box around the detected hand region before classification.

---

## 10. Running the Real-Time Application

Place the required deployment files where `app.py` expects them:

```text
app.py
final_model.keras
classes.json
class_indices.json
config.json
```

If the files are inside `bdsl49_results/`, either copy them beside `app.py` or update the paths in `app.py`.

Then run:

```bash
streamlit run app.py
```

Open the local Streamlit address shown in the terminal and allow webcam access.

The application displays:

```text
Predicted sign
Confidence
FPS
Latency
RAM usage
```

The final session also reports aggregate runtime statistics.

---

## 11. Real-Time Performance

The application measures the complete processing pipeline rather than neural-network inference alone.

Therefore, reported latency includes operations such as:

```text
Frame capture
+ MediaPipe hand detection
+ Cropping
+ Image preprocessing
+ Model inference
+ Output processing
```

The runtime metrics include:

```text
FPS
End-to-end latency
Current RAM
Peak RAM
```

Actual performance depends on the hardware, operating system, TensorFlow version, camera, and execution environment.

---

## 12. Reported Experimental Result

The following values correspond to the reported **Static Two Hand, MobileNetV2, first-50-layers-frozen** experiment:

| Metric                     |    Reported value |
| -------------------------- | ----------------: |
| Classes                    |                49 |
| Architecture               |       MobileNetV2 |
| Frozen layers              |          First 50 |
| Input                      |     224 × 224 × 3 |
| Cross-validation           | 3-fold stratified |
| Train accuracy             |            99.57% |
| Validation accuracy        |            97.09% |
| Test accuracy              |            96.33% |
| Precision                  |            96.93% |
| Recall                     |            96.63% |
| Weighted F1-score          |            96.54% |
| Average peak RAM           |          848.1 MB |
| Average FPS                |                 5 |
| Average end-to-end latency |          231.3 ms |

These are the reported results for the supplied experiment, not guaranteed values for every reproduction. Hardware, software versions, random seeds, and execution environments can affect the results.

---

## 13. Recommended Reproducibility Information

When reproducing or extending this experiment, record:

```text
Python version
TensorFlow / Keras version
CUDA / cuDNN version
GPU model
GPU memory
CPU model
RAM
Operating system
Dataset version
Git commit
Random seed
Execution platform
```

This is particularly important when comparing training results or real-time FPS, latency, and RAM consumption.

---

## 14. Research Extensions

The repository can be used as a baseline for further experiments, including:

```text
Different CNN backbones
Different numbers of frozen layers
Alternative augmentation strategies
Different input resolutions
Fine-tuning strategies
TensorFlow Lite / ONNX deployment
Quantization
Pruning
Knowledge distillation
Detailed real-time profiling
```

Researchers extending the work should document any changes to the dataset split, preprocessing, model architecture, training configuration, and evaluation protocol.

---

## 15. Kaggle Reproduction

A Kaggle notebook is also available as a complementary reference for GPU-based reproduction:

**BdSL49 Updated 2026 — MobileNetV2 Cross Validation**

[https://www.kaggle.com/code/hassan0008jhh/bdsl49-updated-2026-mobilenetv2-crossvalidation](https://www.kaggle.com/code/hassan0008jhh/bdsl49-updated-2026-mobilenetv2-crossvalidation)

The GitHub notebook remains the source implementation. The Kaggle notebook is provided as an additional reproduction environment.

---

## 16. Citation and Licenses

### Dataset

If you use **BdSL49_Updated_2026**, please cite the Zenodo dataset:

```text
BdSL49_Updated_2026
DOI: 10.5281/zenodo.22930407
```

[https://doi.org/10.5281/zenodo.22930407](https://doi.org/10.5281/zenodo.22930407)

### Model artifact

The final trained model is separately available through Zenodo:

[https://doi.org/10.5281/zenodo.22944642](https://doi.org/10.5281/zenodo.22944642)

Please follow the license and attribution requirements specified by the corresponding Zenodo records.

### Source code

The repository source code is released under the **MIT License**.

The dataset and source code should be treated as separate licensed resources. Please check the applicable license terms before redistributing either one.

---

## 17. Quick Start

For users who are already familiar with the workflow:

```bash
# 1. Clone repository
git clone <GITHUB-REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download BdSL49_Updated_2026 from Zenodo
#    https://doi.org/10.5281/zenodo.22930407

# 4. Extract the dataset as:
#
#    bdsl49_updated_2026/
#        train/
#        val/
#        test/

# 5. Run:
#    mobilentv2_First50FrozenLayers_cross_validation.ipynb

# 6. After training, make the deployment artifacts
#    available to app.py:
#
#    final_model.keras
#    classes.json
#    class_indices.json
#    config.json

# 7. Start the real-time application
streamlit run app.py
```

---

## 18. Contact

For reproducibility issues, implementation questions, or problems running the pipeline:

**Mehedi**
**Email:** [mehedi3128.mhd@gmail.com](mailto:mehedi3128.mhd@gmail.com)

When reporting an issue, please include the operating system, Python/TensorFlow version, relevant error message, and the step where the problem occurred.

```
```
