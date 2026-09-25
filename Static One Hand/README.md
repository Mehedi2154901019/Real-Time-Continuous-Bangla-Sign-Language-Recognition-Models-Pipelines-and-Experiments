#BdSL47_Updated_2026 Dataset, model training,ablation study and real-time pipeline materials
---
The BdSL47_Updated_2026 dataset was used to train static one-handed Bangla Sign Language utilizing Mediapipe hand landmarks and Multi-Layer Perceptron (MLP).

The dataset is available at Zenodo:
https://doi.org/10.5281/zenodo.22937669
It has BdSL47_Updated_2026_csv file that needs to be downloaded and the detailed architecture is described there.

The current repository contains 'model.ipynb' that can directly be pulled and keep in the same directory of the csv folder. The model.ipynb is used for training and it automatically saves the artifacts scaler.pkl, pipeline_config.json and final_model.keras. They can be later utilized for the real-time streamlit pipeline as in app.py.

Also, the ablation study shows which model architecture and features work the best and finally the model.ipynb was selected which can be used for baseline for further study or research. The original dataset 'BdSL47' had only 10 signers which affects the model learning inefficiently because of only similar hand and backgrounds respectively.
The NotoSansBengali-Regular.ttf in this repository is used to show bangla text on live streamlit. The class mapping in app.py was designed as per the training which were saved as label 0-46 in the csv files where 0-9 are digits or sign0-sign9 in the dataset, 10-46 are alphabets pointing to sign00-sign36 of the dataset.
