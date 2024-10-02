# model/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from config import Config
from model.model import MultiheadAttentionModel
from utils.data_loader import create_data_loader

def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device).long()  # Ensure input_ids are LongTensor
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).long()  # Ensure labels are LongTensor

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)  # Use loss function directly
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device).long()  # Ensure LongTensor
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).long()  # Ensure LongTensor

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def train_model(df):
    device = Config.DEVICE
    model = MultiheadAttentionModel(Config.NUM_CLASSES).to(device)

    # Freeze BERT layers for initial training
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = create_data_loader(df)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    best_accuracy = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Training loss: {train_loss:.4f}")

        # Evaluate the model after each epoch
        accuracy = evaluate_model(model, train_loader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Checkpoint the model if it improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'model/best_model.pth')
            print("Best model saved!")

        # Step the scheduler
        scheduler.step(train_loss)

    # Unfreeze BERT layers for fine-tuning
    for param in model.bert.parameters():
        param.requires_grad = True
    print("BERT layers unfrozen for fine-tuning.")
