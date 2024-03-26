import os
import pandas as pd, numpy as np

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

def train_one_epoch(model, dataloader, criterion, optimizer, metric, device, DATA_LENGTH, LEAD_TIME):
    model.train()
    output_list = []
    target_list = []

    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        
        x_emr = x['emr'][:,:].to(device)
        x_vitals = x['vitals'][:, -DATA_LENGTH:, :].to(device) # VITALS: HR/RR/SpO2/DBP/SBP/BT
        y = y.to(device)
        outputs = model(x_emr, x_vitals).to(torch.float16)

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

def valid_one_epoch(model, dataloader, criterion, metric, device, DATA_LENGTH, LEAD_TIME):
    model.eval()
    output_list = []
    target_list = []
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x_emr = x['emr'][:,:].to(device)
            x_vitals = x['vitals'][:, -DATA_LENGTH:, :].to(device) # [:,144:,:]
            y = y.to(device)

            outputs = model(x_emr, x_vitals).to(torch.float16)

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
    data_path="/home/hjkim/projects/local_dev/delirium/data/dynamic"
    dst_path = "/home/hjkim/projects/local_dev/delirium/result"
    # For 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = '1'
    # torch.cuda.empty_cache()
# 1. Dataset
    BATCH_SIZE=64
    LEAD_TIME = 2       
    DATA_LENGTH = 2 * 12   # 1Hour = 12 vitals
    # load npy files
    train_data = np.load(os.path.join(data_path, "train_" + str(LEAD_TIME) + "h.npy"), allow_pickle=True).item()
    valid_data = np.load(os.path.join(data_path, "valid_" + str(LEAD_TIME) + "h.npy"), allow_pickle=True).item()
    test_data = np.load(os.path.join(data_path, "test_" + str(LEAD_TIME) + "h.npy"), allow_pickle=True).item()
    # create dataset
    train_dataset = Delirium_Dataset(train_data)
    valid_dataset = Delirium_Dataset(valid_data)
    test_dataset = Delirium_Dataset(test_data)
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)
    emr_size = train_data['emr'][:,:].shape[1]
    vital_size = train_data['vitals'][:, -DATA_LENGTH:, :].shape[2]

# 2. Set options (device, Loss, etc ....)
    SEED=42
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-5
    device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    metric = BinaryAUROC()
    fix_seed(SEED)

    model = CNNLSTMModel(emr_size, vital_size).to(device)
    # model = CRNN(emr_size=emr_size, vitals_size=vital_size).to(device)
    # model = AttentionLSTM(emr_size=emr_size, vitals_size=vital_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

# 3. Training & Validation
    train_loss_list = []
    valid_loss_list = [] 
    train_auc_list = []
    valid_auc_list = []

    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(NUM_EPOCHS):

        train_epoch_loss, train_auc = train_one_epoch(model, train_dataloader, criterion, optimizer, metric, device, DATA_LENGTH, LEAD_TIME)  # train
        val_epoch_loss, val_auc = valid_one_epoch(model, valid_dataloader, criterion, metric, device, DATA_LENGTH, LEAD_TIME) # valid
        # Scheduler
        scheduler.step(val_epoch_loss)

        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            best_epoch = epoch
            break

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
    test_loss, test_auc = valid_one_epoch(model, test_dataloader, criterion, metric, device, DATA_LENGTH, LEAD_TIME) 
    print(f"BEST PERFORMANCE: {best_epoch}")
    print(f"Test Loss: {test_loss}, Test AUROC: {test_auc}")

