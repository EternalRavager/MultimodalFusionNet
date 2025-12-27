import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os

# Local imports
import scripts.config as config
from scripts.dataset import MultimodalWebpageDataset, prepare_data
from scripts.model import MultimodalFusionNet

class FocalLoss(nn.Module):
    """
    Custom Loss function to handle class imbalance.
    It applies a modulating term to the cross entropy loss to focus learning on hard misclassified examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def train_step(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = []
    correct = 0
    total = 0
    
    # Tqdm provides a nice progress bar for the terminal
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for url_feats, text_seq, label in progress_bar:
        url_feats, text_seq, label = url_feats.to(device), text_seq.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(url_feats, text_seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
        preds = output.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return np.mean(running_loss), correct / total

def val_step(model, loader, criterion, device):
    model.eval()
    running_loss = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for url_feats, text_seq, label in loader:
            url_feats, text_seq, label = url_feats.to(device), text_seq.to(device), label.to(device)
            
            output = model(url_feats, text_seq)
            loss = criterion(output, label)
            
            running_loss.append(loss.item())
            preds = output.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
    return np.mean(running_loss), correct / total

def main():
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data
    train_df, val_df, test_df, text_vocab, le = prepare_data()
    
    train_set = MultimodalWebpageDataset(train_df, text_vocab, config.MAX_TEXT_LENGTH)
    val_set = MultimodalWebpageDataset(val_df, text_vocab, config.MAX_TEXT_LENGTH)
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE)
    
    # 2. Handle Class Imbalance
    # We calculate weights inversely proportional to class frequency
    classes = np.unique(train_df['label'])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    # 3. Initialize Model
    # Note: 14 features comes from the utils.extract_url_features function
    model = MultimodalFusionNet(
        num_url_features=14, 
        vocab_size=config.VOCAB_SIZE, 
        embed_dim=config.EMBED_DIM, 
        num_classes=len(le.classes_)
    ).to(device)
    
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 4. Training Loop
    best_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    
    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_step(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(config.MODELS_DIR, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"--> New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= patience:
            print("Early stopping triggered to prevent overfitting.")
            break
            
    # Save artifacts for inference
    with open(os.path.join(config.MODELS_DIR, 'text_vocab.pkl'), 'wb') as f:
        pickle.dump(text_vocab, f)
    with open(os.path.join(config.MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
        
    print("Training complete.")

if __name__ == '__main__':
    main()