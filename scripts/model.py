import torch
import torch.nn as nn

class MultimodalFusionNet(nn.Module):
    """
    A Hybrid Neural Network that fuses two branches:
    1. MLP Branch: Processes explicit structural features (URL length, dot count, etc.)
    2. LSTM Branch: Processes the semantic sequence of the URL text.
    """
    def __init__(self, num_url_features, vocab_size, embed_dim, num_classes):
        super().__init__()
        
        # --- Branch 1: Structural URL Features (Dense Network) ---
        self.url_fc = nn.Sequential(
            nn.Linear(num_url_features, 64),
            nn.BatchNorm1d(64),  # Normalizing to help with convergence
            nn.ReLU(),
            nn.Dropout(0.3),     # Prevent overfitting on specific features
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # --- Branch 2: Semantic Text Features (LSTM) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM allows the model to see context from both left and right
        self.text_lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3, 
            bidirectional=True
        )
        
        # Post-LSTM processing
        self.text_fc = nn.Sequential(
            nn.Linear(128 * 2, 128),  # *2 because of bidirectional output
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # --- Fusion Layer ---
        # Concatenate the outputs of both branches and classify
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, url_feats, text_seq):
        # 1. Process URL structural data
        url_out = self.url_fc(url_feats)
        
        # 2. Process Text Sequence
        emb = self.embedding(text_seq)
        # We only need the final hidden state (h), not the full sequence output
        _, (h, _) = self.text_lstm(emb)
        
        # Concatenate the final forward and backward hidden states
        # h structure: [num_layers * num_directions, batch, hidden_size]
        # We take the last two layers (forward and backward of the top layer)
        h_concatenated = torch.cat([h[-2], h[-1]], dim=1)
        text_out = self.text_fc(h_concatenated)
        
        # 3. Fuse and Classify
        combined = torch.cat([url_out, text_out], dim=1)
        out = self.fusion(combined)
        return out