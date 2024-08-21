# config.py

import torch

class Config:
    MAX_LEN = 32  # Maximum sequence length for BERT input
    BATCH_SIZE = 3  # Number of samples processed before the model is updated
    LEARNING_RATE = 2e-5  # Learning rate for the optimizer
    EPOCHS = 14  # Number of training epochs
    MODEL_NAME = 'bert-base-uncased'  # Pretrained BERT model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    NUM_CLASSES = 3  # Number of classes, corresponding to the one-hot encoded labels (Positive, Negative, Neutral)
