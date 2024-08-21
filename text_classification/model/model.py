# model/model.py

import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

class MultiheadAttentionModel(nn.Module):
    def __init__(self, n_classes):
        super(MultiheadAttentionModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)  
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_output = self.dropout(cls_output)  
        return self.fc(cls_output)