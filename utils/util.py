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
