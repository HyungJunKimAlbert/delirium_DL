import os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')
# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Custom 
from dataset.dataset import Delirium_Dataset
from utils.util import fix_seed, loss_plot, auc_plot, EarlyStopping
from models.model import CNNLSTMModel, CRNN, AttentionLSTM

def train_one_epoch(model, dataloader, optimizer, metric, device):
    model.train()
    output_list = []
    target_list = []

    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        # For weighted Loss
        pos_weight = torch.tensor(y.sum() / len(y))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        x_emr = x['emr'][:,:].to(device)
        x_vitals = x['vitals'][:, :, :].to(device) # [:,144:,:]      # VITALS: HR/RR/SpO2/DBP/SBP/BT
        y = y.to(device).float()
        outputs = model(x_emr, x_vitals) 
        outputs = outputs.float()
        
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        output_list.extend(outputs.detach().cpu().numpy())
        target_list.extend(y.detach().cpu().numpy())

    output_list = torch.tensor(output_list)
    target_list = torch.tensor(target_list)

    final_loss = running_loss / (batch_idx + 1)
    auc = metric(output_list, target_list)

    return final_loss, auc

def valid_one_epoch(model, dataloader, metric, device):
    model.eval()
    output_list = []
    target_list = []
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # For weighted Loss
            pos_weight = torch.tensor(y.sum() / len(y))
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            x_emr = x['emr'][:,:].to(device)
            x_vitals = x['vitals'][:, :, :].to(device) # [:,144:,:]
            y = y.to(device).float()

            outputs = model(x_emr, x_vitals)
            outputs = outputs.float()

            loss = criterion(outputs, y)

            running_loss += loss.item()
            output_list.extend(outputs.cpu().detach().numpy())
            target_list.extend(y.cpu().detach().numpy())
            
        output_list = torch.tensor(output_list)
        target_list = torch.tensor(target_list)

        final_loss = running_loss / (batch_idx + 1)
        auc = metric(output_list, target_list)

    return final_loss, auc


if __name__ == "__main__":
# 0. Set Directory Path
    data_path="/home/hjkim/projects/local_dev/delirium/data"
    dst_path = "/home/hjkim/projects/local_dev/delirium/result_static"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. Dataset
    BATCH_SIZE=64
    # load npy files
    train_data = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
    valid_data = np.load(os.path.join(data_path, "valid.npy"), allow_pickle=True).item()
    test_data = np.load(os.path.join(data_path, "test.npy"), allow_pickle=True).item()
    # create dataset
    train_dataset = Delirium_Dataset(train_data)
    valid_dataset = Delirium_Dataset(valid_data)
    test_dataset = Delirium_Dataset(test_data)
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)
    emr_size = train_data['emr'][:,:].shape[1]
    vital_size = train_data['vitals'][:, :, :].shape[2]
    print(f"TRAIN SIZE: {len(train_dataloader)}, VALID SIZE: {len(valid_dataloader)}, TEST SIZE: {len(test_dataloader)}")
    print("LABEL")
    print(f"TRAIN DELIRIUM: {train_data['label'].sum()}, NON-DELIRIUM :{ len(train_data['label']) - train_data['label'].sum() } \
          RATE: {train_data['label'].sum() / len(train_data['label']) } ")
    print(f"VALID DELIRIUM: {valid_data['label'].sum()}, NON-DELIRIUM :{ len(valid_data['label']) - valid_data['label'].sum() }\
          RATE: {valid_data['label'].sum() / len(valid_data['label']) } ")
    print(f"TEST DELIRIUM: {test_data['label'].sum()}, NON-DELIRIUM :{ len(test_data['label']) - test_data['label'].sum() } \
          RATE: {test_data['label'].sum() / len(test_data['label']) } ")

# 2. Set options (device, Loss, etc ....)
    SEED=42
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    metric = BinaryAUROC()
    fix_seed(SEED)

    model = CNNLSTMModel(emr_size, vital_size).to(device)
    # model = CRNN(emr_size=emr_size, vitals_size=vital_size).to(device)
    # model = AttentionLSTM(emr_size=emr_size, vitals_size=vital_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)
# 3. Training & Validation
    train_loss_list = []
    valid_loss_list = [] 
    train_auc_list = []
    valid_auc_list = []

    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(NUM_EPOCHS):

        train_epoch_loss, train_auc = train_one_epoch(model, train_dataloader, optimizer, metric, device)  # train
        val_epoch_loss, val_auc = valid_one_epoch(model, valid_dataloader, metric, device) # valid
        
        early_stopping(val_auc)

        if early_stopping.early_stop:
            print("Early stopping")
            best_epoch = epoch
            best_model = model.state_dict()
            break

        # Scheduler
        scheduler.step(val_epoch_loss)

        # train results
        train_loss_list.append(train_epoch_loss)
        train_auc_list.append(train_auc)
        # valid_results
        valid_loss_list.append(val_epoch_loss)
        valid_auc_list.append(val_auc)
        
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Plotting
    loss_plot(train_loss_list, valid_loss_list, dst_path=dst_path)
    auc_plot(train_auc_list, valid_auc_list, dst_path=dst_path)
    
    # Testset
    model.load_state_dict(best_model)
    test_loss, test_auc = valid_one_epoch(model, test_dataloader, metric, device) 
    print(f"BEST PERFORMANCE: {best_epoch}")
    print(f"Test Loss: {test_loss}, Test AUROC: {test_auc}")

