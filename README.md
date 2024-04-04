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
    - 6 Transforemr Layers
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

- Data Split

    Delirium      | Train  | Valid  | Test   | Total | 
    ------------- | ------ | ------ | ------ | ----- |
    Case          | 669    | 155    | 129    | 953   |
    Control       | 1760   | 365    | 392    | 2517  |
    Incidence     | 0.2754 | 0.2981 | 0.2476 | 3470  |

### Results

Method                   | AUC     | PRC 
------------------------ | ------- | ------- | 
CRNN                     | 0.8167  | 0.624
ResNet + LSTM            | 0.817   | 0.616
Attentaion + LSTM        | 0.8053  | 0.599 
Tranformer + LSTM        | **0.8251**  | **0.641**

