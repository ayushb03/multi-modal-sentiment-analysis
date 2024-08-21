# test.py

import torch
import pandas as pd
from utils.preprocess import load_data, preprocess_data
from utils.data_loader import create_data_loader
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer
from model.model import MultiheadAttentionModel  
from utils.data_loader import create_data_loader   
from config import Config

model = MultiheadAttentionModel(Config.NUM_CLASSES).to(Config.DEVICE)
model.load_state_dict(torch.load('model/model.pth'))


def test_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


test_df = load_data('data/cmu-mosi/label.csv')
test_df = preprocess_data(test_df)

test_loader = create_data_loader(test_df)

true_labels, predicted_labels = test_model(model, test_loader, Config.DEVICE)

# accuracy score
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy:.4f}")

# calssification report
print(classification_report(true_labels, predicted_labels, target_names=["negative", "neutral", "positive"]))