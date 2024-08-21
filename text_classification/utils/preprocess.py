# utils/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def to_lower(text):
    return text.lower()

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df = df.drop(columns=['label_T', 'label_V', 'label_A'], axis=1)

    df['text'] = df['text'].apply(to_lower)

    label_encoder = LabelEncoder()
    df['annotation'] = label_encoder.fit_transform(df['annotation'])
    df = df[['text', 'annotation']]
    
    print(df.head(), end='\n\n')
    print(df.groupby('annotation').count())
    return df