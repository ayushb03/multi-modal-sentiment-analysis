# utils/data_loader.py

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from config import Config

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df):
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = TextDataset(
        texts=df['text'].to_numpy(),
        labels=df['annotation'].to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.MAX_LEN
    )
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)