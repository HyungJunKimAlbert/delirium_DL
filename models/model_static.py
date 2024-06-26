import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CNNLSTMBlock(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, units, dropout):
        super(CNNLSTMBlock, self).__init__()

        self.cnn = nn.Sequential(
            # 1
            nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_filters),
            nn.Dropout(dropout),
            # 2
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_filters*2),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        self.lstm1 = nn.LSTM(num_filters*2, units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(units*2, units//2, bidirectional=True, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)      
        x = self.cnn(x)             
        x = x.permute(0, 2, 1)      
        x, _ = self.lstm1(x)        
        x = self.dropout1(x)        
        x, _ = self.lstm2(x)        
        x = x[:, -1, :]             
        x = self.dropout2(x)        
        return x
    

class CNNLSTMModel(nn.Module):
    def __init__(self, emr_size, vitals_size, num_filters=256, kernel_size=7, units=256, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        # CRNN Blocks
        self.cnn_lstm_block = nn.ModuleList(
        [CNNLSTMBlock(vitals_size, num_filters, kernel_size, units, dropout)]
        )
        # FC layers
        self.fc1 = nn.Linear((emr_size + units), units)
        self.bn = nn.BatchNorm1d(units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(units, 1)

    def forward(self, x_emr, x_vitals):
        x_emr = x_emr.float()  # Convert input EMR data to Double type
        x_vitals = x_vitals.float()
        hs = [cnn_lstm_block(x_vitals) for cnn_lstm_block in self.cnn_lstm_block]
        h = torch.cat([x_emr] + hs, dim=-1) 
        h = self.fc1(h)                    
        h = self.bn(h)
        h = self.dropout(h)
        h = self.fc2(h)                     

        return h.squeeze()
    

# ResNet + LSTM 
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        x += residual
        x = self.dropout(x)
        x = self.relu(x)
        return x
    
class CRNN(nn.Module):
    def __init__(self, emr_size, vitals_size, num_filters=128, hidden_size=256, dropout_rate=0.4):
        super(CRNN, self).__init__()
        # CNN Layer
        self.resblock1 = ResBlock(vitals_size, num_filters, kernel_size=7)
        self.resblock2 = ResBlock(num_filters, num_filters, kernel_size=7)
        self.resblock3 = ResBlock(num_filters, num_filters, kernel_size=7)
        self.dropout_cnn = nn.Dropout(dropout_rate)
        # RNN Layer
        self.bi_lstm = nn.LSTM(num_filters, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        # FC layer
        self.fc = nn.Linear((emr_size + hidden_size*2), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self,  x_emr, x_vitals):
        x_emr = x_emr.float()  # Convert input EMR data to Double type
        x_vitals = x_vitals.float().permute(0, 2, 1)
        # CNN
        x_vitals = self.resblock1(x_vitals)
        x_vitals = self.resblock2(x_vitals)
        x_vitals = self.resblock3(x_vitals)
        x_vitals = self.dropout_cnn(x_vitals)
        # LSTM Layer
        x_vitals = x_vitals.permute(0, 2, 1) # input shape: (batch_size, 187, 1) 
        x_vitals, _ = self.bi_lstm(x_vitals)
        x_vitals = self.dropout_lstm(x_vitals)
        hs = x_vitals[:, -1, :]
        # Concat EMR + Vitals
        h = torch.cat([x_emr, hs], dim=-1) # torch.Size([32, 148])
        # FC Layer 
        h = self.fc(h)
        h = self.dropout_fc(h)
        h = self.fc2(h)
        h = self.sigmoid(h)           

        return h.squeeze()




# Attention LSTM
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # attn_weights = self.softmax(self.attn(x))
        attn_weights = self.softmax(torch.relu(self.attn(x)))  # relu or tanh
        context = torch.sum(attn_weights * x, dim=1)

        return context, attn_weights


class AttentionLSTM(nn.Module):
    def __init__(self, emr_size, vitals_size, hidden_size=256, num_classes=1, dropout_rate=0.4):
        super(AttentionLSTM, self).__init__()
        # LSTM Layer
        self.bi_lstm = nn.LSTM(vitals_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        # Attention Layer
        self.attention = Attention(hidden_size*2)
        # FC Layer
        self.fc1 = nn.Linear(emr_size + hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_emr, x_vitals):
        x_emr = x_emr.float()  # Convert input EMR data to Double type
        x_vitals = x_vitals.float()
        # LSTM Layer
        lstm_out, _, = self.bi_lstm(x_vitals)
        # lstm_out, _, = self.bi_lstm2(self.dropout(lstm_out))

        # Attention Layer
        context, attn_weights = self.attention(lstm_out)
        # Concat EMR + Vitals
        h = torch.cat([x_emr, context], dim=-1) # torch.Size([32, 148])

        # FC Layer
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.sigmoid(h).squeeze()       

        # return h, attn_weights    # Attention score
        return h
    
    

