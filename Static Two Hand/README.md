# BdSL49 Updated 2026 — MobileNetV2 with Cross-Validation

Static **two-hand Bangla Sign Language (BdSL)** recognition using **ImageNet-pretrained MobileNetV2**, with the first 50 backbone layers frozen and 3-fold stratified cross-validation.

## Dataset

**BdSL49_Updated_2026**

* 49 static two-hand Bangla Sign Language classes
* Train / validation / test splits
* Input size: `224 × 224 × 3`
* Dataset DOI: https://doi.org/10.5281/zenodo.22930407

Expected dataset structure:

```text
bdsl49_updated_2026/
├── train/
├── val/
└── test/
```

Each split contains the 49 class directories. The notebook automatically discovers and sorts the class names.

---

## Repository Contents

```text
Static Two Hand/
├── mobilentv2_First50FrozenLayers_cross_validation.ipynb
├── app.py
├── readme.md
├── class_indices.json
├── classes.json
├── config.json
└── requirements.txt
```

The trained `final_model.keras` is **not included in GitHub** because it exceeds GitHub's file-size limit.

It is available on Zenodo:

**Final model:** https://doi.org/10.5281/zenodo.22944642

---

## Model Architecture

The classifier uses ImageNet-pretrained MobileNetV2 with the first 50 backbone layers frozen.

```text
Input: 224 × 224 × 3
        ↓
MobileNetV2 (ImageNet pretrained)
        ↓
Global Average Pooling
        ↓
Dense(256, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(49, Softmax)
```

### Training Configuration

* Optimizer: Adam
* Learning rate: `1e-4`
* Loss: Sparse Categorical Crossentropy
* Maximum epochs: `20`
* Cross-validation: 3-fold StratifiedKFold
* Shuffle: `True`
* Random state: `42`
* Frozen backbone layers: First `50`
* Data augmentation:

  * Random horizontal flip
  * Random brightness
* MobileNetV2 `preprocess_input`
* Early stopping and model checkpointing

---

## Installation

Python dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

For training/reproduction, ensure that the dataset is downloaded from Zenodo and that the dataset path used by the notebook matches the extracted directory.

---

## Reproducing the Experiments

Open:

```text
mobilentv2_First50FrozenLayers_cross_validation.ipynb
```

The notebook performs the following:

1. Loads the BdSL49 dataset.
2. Discovers the 49 class labels.
3. Applies MobileNetV2 preprocessing and augmentation.
4. Performs 3-fold stratified cross-validation.
5. Freezes the first 50 MobileNetV2 layers.
6. Trains the classification head.
7. Saves the best model for each fold.
8. Evaluates the final model on the independent test set.
9. Generates evaluation results and confusion matrices.

For reproducibility, record the Python/TensorFlow versions, hardware, operating system, dataset version, random seed, and Git commit used for an experiment.

---

## Final Model

The final trained model is provided separately because its size exceeds GitHub's file-size limit.

**Download:** https://doi.org/10.5281/zenodo.22944642

After downloading, place the model in the project directory:

```text
Static Two Hand/
├── app.py
├── final_model.keras
├── classes.json
├── class_indices.json
├── config.json
└── requirements.txt
```

The JSON files provide the class and configuration information required by the real-time application.

---

## Real-Time Recognition

`app.py` provides a Streamlit-based real-time recognition pipeline using:

* OpenCV
* MediaPipe Hands
* TensorFlow/Keras
* MobileNetV2
* Streamlit
* psutil

The pipeline detects one or two hands, creates a combined bounding box, applies padding, crops the hand region, resizes it to `224 × 224`, applies MobileNetV2 preprocessing, and performs classification.

Run:

```bash
streamlit run app.py
```

Required files:

```text
app.py
final_model.keras
classes.json
class_indices.json
config.json
```

---

## Real-Time Performance

The application reports:

* Predicted class
* Prediction confidence
* FPS
* End-to-end frame-processing latency
* Current RAM usage
* Peak RAM usage

Session-level statistics include total processed frames, average latency, average operational FPS, and peak RAM.

---

## Reported Experimental Results

| Metric                     |            Result |
| -------------------------- | ----------------: |
| Number of classes          |                49 |
| Architecture               |       MobileNetV2 |
| Frozen layers              |          First 50 |
| Input size                 |     224 × 224 × 3 |
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

Latency represents **end-to-end frame-processing latency**, rather than model inference time alone.

---

## Reproducibility

For future experiments, it is recommended to report:

```text
Python version
TensorFlow/Keras version
CUDA/cuDNN version
GPU/CPU
RAM
Operating system
Dataset DOI/version
Git commit
Random seed
Execution platform
```

This information helps future researchers reproduce and compare experiments consistently.

---

## Research Extensions

Possible extensions include:

* Comparing different pretrained CNN backbones
* Testing different numbers of frozen MobileNetV2 layers
* Evaluating alternative augmentation strategies
* Comparing input resolutions
* TFLite or ONNX deployment
* Quantization
* Pruning
* Knowledge distillation
* Detailed inference-time and memory profiling

---

## Kaggle Reproduction

A Kaggle version of the experiment is available here:

https://www.kaggle.com/code/hassan0008jhh/bdsl49-updated-2026-mobilenetv2-crossvalidation

---
## Ablation study availability
EfficientNetV2B0: https://www.kaggle.com/code/hassan0008jhh/bdsl49efficientnetv2b0crossvalidation
Xception: https://www.kaggle.com/code/hassan0008jhh/bdsl49-xception-cross-validation

## Citation and Data Availability

### Dataset

BdSL49_Updated_2026:

https://doi.org/10.5281/zenodo.22930407

### Final Model

Trained `final_model.keras`:

https://doi.org/10.5281/zenodo.22944642

The GitHub repository contains the training notebook, inference application, configuration files, and requirements. The large trained model is distributed separately through Zenodo.

---

## Contact

For questions, reproducibility issues, or research collaboration:

**Mehedi**
Email: `mehedi3128.mhd@gmail.com`
