import os, warnings
import pandas as pd, numpy as np

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
# Custom 
from dataset.dataset import Delirium_Dataset
from utils.util import fix_seed, loss_plot, auc_plot, EarlyStopping
from models.model import CNNLSTMModel, CRNN, AttentionLSTM
warnings.filterwarnings('ignore')

def train_one_epoch(model, dataloader, optimizer, criterion, metric, device, DATA_LENGTH):

    model.train()
    output_list = []
    target_list = []

    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        # For weighted Loss
        # pos_weight = torch.tensor(y.sum() / len(y))
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

def valid_one_epoch(model, dataloader, criterion, metric, device, DATA_LENGTH):
    model.eval()
    output_list = []
    target_list = []
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # pos_weight = torch.tensor(y.sum() / len(y))
            # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = '1'
    # torch.cuda.empty_cache()
# 1. Dataset
    BATCH_SIZE = 32
    LEAD_TIME = 1
    DATA_LENGTH =  2 * 12   # 1Hour = 12 vitals
    # load npy files
    # train_data = np.load(os.path.join(data_path, "train_" + str(LEAD_TIME) + "h_add_labs.npy"), allow_pickle=True).item()
    # valid_data = np.load(os.path.join(data_path, "valid_" + str(LEAD_TIME) + "h_add_labs.npy"), allow_pickle=True).item()
    # test_data = np.load(os.path.join(data_path, "test_" + str(LEAD_TIME) + "h_add_labs.npy"), allow_pickle=True).item()
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
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-4
    device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    metric = BinaryAUROC()
    fix_seed(SEED)

    model = CNNLSTMModel(emr_size, vital_size).to(device)
    # model = CRNN(emr_size=emr_size, vitals_size=vital_size).to(device)
    # model = AttentionLSTM(emr_size=emr_size, vitals_size=vital_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)
    
# 3. Training & Validation
    train_loss_list = []
    valid_loss_list = [] 
    train_auc_list = []
    valid_auc_list = []

    # Early stopping
    # early_stopping = EarlyStopping(patience=10, verbose=True)
    best_val_auc = 0.0

    for epoch in range(NUM_EPOCHS):

        train_epoch_loss, train_auc = train_one_epoch(model, train_dataloader, optimizer, criterion, metric, device, DATA_LENGTH)  # train
        val_epoch_loss, val_auc = valid_one_epoch(model, valid_dataloader, criterion, metric, device, DATA_LENGTH) # valid

        # early_stopping(val_auc)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     best_epoch = epoch
        #     best_model = model.state_dict()
        #     break
        
        if best_val_auc < val_auc:
            best_epoch = epoch
            best_val_auc = val_auc
            best_model = model.state_dict()

        if epoch - best_epoch > 5:
            print("Early Stopping...")
            break

        # Scheduler
        scheduler.step(val_auc)

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
    model.load_state_dict(best_model)   # best model weights
    test_loss, test_auc = valid_one_epoch(model, test_dataloader, criterion, metric, device, DATA_LENGTH) 
    print(f"BEST PERFORMANCE: {best_epoch}")
    print(f"Test Loss: {test_loss}, Test AUROC: {test_auc}")

