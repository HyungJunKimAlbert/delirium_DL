## Delirium_DL

---

### Project Outline
    - Task
        - Prediction Delirium State for ICU patietns
    - Aim
        - AUC 0.82
        - PRC 0.6
    - Logging : not used 

---

### Model 

- Architecture:
  - Transformer based LSTM
    - 6 Layers Transforemr Layer
    - 2 bi-LSTM Layers

- Hyperparameters
    - Learning Rate: 1e-4
    - Epoch : 100
    - Batch Size: 32
    - Optimier : Adam
    - Normalization: Batch Normalization

- Deep Learning Framework: 
  - Pytorch (https://pytorch.org/)

---

### Dataset
- EICU-crd Dataset
    - Number of Target
        - Delirium (Case) : 
        - Non-Delirium (Control) : 

- Preprocessing
    - Outlier Removal
    - Integrate EMR and Vital sign Data
    - Standart Scaling

---

### Results

Method                   | AUC     | PRC 
------------------------ | ------- | ------- | 
CRNN                     | 0.8167  | 0.624
ResNet + LSTM            | 0.817   | 0.616
Attentaion + LSTM        | 0.8053  | 0.599 
Tranformer + LSTM        | **0.8251**  | **0.641**

