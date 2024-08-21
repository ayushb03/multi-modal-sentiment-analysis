# main.py

import pandas as pd
from model.train import train_model
from utils.preprocess import load_data, preprocess_data

def main():
    df = load_data('data/cmu-mosi/label.csv')
    df = preprocess_data(df)

    train_model(df)

if __name__ == "__main__":
    main()