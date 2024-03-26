import numpy as np, pandas as pd
import os
import random
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn




def fix_seed(SEED=42):
    os.environ['SEED'] = str(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)


def get_label(demo):
    # Set case-control patients
    label = pd.read_csv("/home/hjkim/projects/local_dev/delirium/data/label.csv", index_col=0)
    label = label[label.patientunitstayid.isin(demo.patientunitstayid)] # demographics 없는 경우 제외 (Age, Gender, Weight/Height)

    case = label[(label.delirium==1) & (label.charttime>1*24*60)] # onset 24h ~ 7days
    case_plist = list(set(case.patientunitstayid))
    control = label[(label.delirium==0) & (label.charttime>1*24*60) ]    
    control = control[~control.patientunitstayid.isin(case_plist)].reset_index(drop=True)  # delirium positive patients excluded

    case = case.sort_values(by=['patientunitstayid', 'charttime']).groupby(['patientunitstayid']).head(1).reset_index(drop=True)
    control_sampled = control.groupby('patientunitstayid').apply(lambda x: x.sample(min(2, len(x)), random_state=42))
    control_sampled = control_sampled[['charttime', 'delirium']].reset_index()[['patientunitstayid', 'charttime', 'delirium']]
    final_label = pd.concat([case, control_sampled], axis=0).reset_index(drop=True)

    return final_label

def loss_plot(train_losses, valid_losses, dst_path):
    plt.clf()
    # Loss Plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(dst_path + '/loss_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()

def auc_plot(train_auc, valid_auc, dst_path):
    plt.clf()
    # AUC Plot
    plt.plot(train_auc, label='Training AUC')
    plt.plot(valid_auc, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.savefig(dst_path + '/auc_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()


def qSOFA_Score(sys_bp, rr, gcs):
    score = 0

    if (sys_bp < 100) and (sys_bp > 0):
        score +=1
    if (rr > 20):
        score += 1
    if (gcs <= 14) and (gcs > 0):
        score += 1
    return score



class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
