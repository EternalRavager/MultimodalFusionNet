import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from scripts.utils import extract_url_features, clean_text
import scripts.config as config

class MultimodalWebpageDataset(Dataset):
    """
    Custom Dataset class to handle the two modalities:
    1. Textual content (Title + Description) -> Processed via LSTM
    2. Structural features (URL properties) -> Processed via Dense layers
    """
    def __init__(self, df, text_vocab, max_text_len):
        self.texts = df['combined_text'].tolist()
        self.urls = df['url'].tolist()
        self.labels = df['label'].tolist() 
        self.text_vocab = text_vocab
        self.max_text_len = max_text_len

    def text_to_sequence(self, text):
        """Converts text string into a list of integer IDs based on vocabulary."""
        # Default to 1 (<unk>) if word not found
        ids = [self.text_vocab.get(w, 1) for w in str(text).split()]
        
        # Truncate or Pad to fixed length
        ids = ids[:self.max_text_len]
        padding = [0] * (self.max_text_len - len(ids))
        ids += padding
        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 1. Prepare Text Sequence (Title + Desc)
        text_seq = torch.tensor(self.text_to_sequence(self.texts[idx]), dtype=torch.long)
        
        # 2. Extract URL Features
        # We process the URL string purely for its structural signals (https, length, etc.)
        url_feats = torch.tensor(extract_url_features(self.urls[idx]), dtype=torch.float32)
        
        # 3. Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return url_feats, text_seq, label

def build_text_vocab(texts, vocab_size):
    """Creates a mapping from words to unique Integers."""
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())
    
    # Reserve spots for padding (0) and unknown (1)
    most_common = counter.most_common(vocab_size - 2)
    stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    stoi['<pad>'] = 0
    stoi['<unk>'] = 1
    return stoi

def prepare_data():
    """
    Loads raw CSV from Curlie/DMOZ format, handles preprocessing, 
    splits data, and builds vocabulary.
    """
    print(f"Loading data from {config.RAW_DATA_PATH}...")
    
    # Load CSV (No headers as per Curlie.org format)
    # Expected columns: URL, Title, Description, Category ID
    try:
        df = pd.read_csv(config.RAW_DATA_PATH, header=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file at {config.RAW_DATA_PATH}. Check your paths!")

    # Robust column assignment
    if len(df.columns) >= 4:
        # We only take the first 4 columns if there are extras
        df = df.iloc[:, :4]
        df.columns = ['url', 'title', 'description', 'category_id']
    else:
        raise ValueError(f"Dataset format error: Expected at least 4 columns, got {len(df.columns)}")
    
    # Ensure data types are correct
    df['url'] = df['url'].astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)
    
    # --- Feature Engineering ---
    # Combine Title and Description for a rich text signal. 
    # Example: "CNN News Breaking News and Video"
    df['combined_text'] = (df['title'] + " " + df['description']).apply(clean_text)
    
    # Filter out empty rows if any
    df = df[df['combined_text'].str.len() > 0].copy()

    # --- Label Encoding ---
    # Even though category_id is numeric, we encode it to ensure 
    # classes map to 0..N-1 perfectly for the CrossEntropyLoss
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category_id'])
    
    print(f"Detected {len(le.classes_)} unique categories.")
    
    # Remove classes with too few samples (less than 2) to enable stratified splitting
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    df = df[df['label'].isin(valid_labels)].copy()
    
    # Re-encode labels after filtering to ensure continuous 0..N-1 mapping
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category_id'])
    print(f"After filtering rare classes: {len(le.classes_)} categories, {len(df)} samples.")
    
    # --- Splitting ---
    # Stratified Split to maintain class distribution across sets
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_SEED, stratify=train_df['label']
    )
    
    # Build Vocabulary only on training data to prevent data leakage
    text_vocab = build_text_vocab(train_df['combined_text'], config.VOCAB_SIZE)
    
    print(f"Data prepared. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, text_vocab, le