import torch
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Local imports
import scripts.config as config
from scripts.dataset import MultimodalWebpageDataset, prepare_data
from scripts.model import MultimodalFusionNet

def main():
    print("Generating visualizations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Artifacts
    with open(os.path.join(config.MODELS_DIR, 'text_vocab.pkl'), 'rb') as f:
        text_vocab = pickle.load(f)
    with open(os.path.join(config.MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
        
    # Get Test Data
    _, _, test_df, _, _ = prepare_data()
    test_set = MultimodalWebpageDataset(test_df, text_vocab, config.MAX_TEXT_LENGTH)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE)
    
    # Load Model
    model = MultimodalFusionNet(
        num_url_features=14,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=len(le.classes_)
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'best_model.pt'), map_location=device))
    model.eval()
    
    # Collect Predictions
    preds, labels = [], []
    with torch.no_grad():
        for url_feats, text_seq, label in test_loader:
            url_feats, text_seq = url_feats.to(device), text_seq.to(device)
            out = model(url_feats, text_seq)
            y_hat = out.argmax(1).cpu().numpy()
            
            preds.extend(y_hat)
            labels.extend(label.numpy())

    # 1. Save Predictions CSV
    pred_df = pd.DataFrame({
        'true_label': le.inverse_transform(labels),
        'pred_label': le.inverse_transform(preds)
    })
    csv_path = os.path.join(config.RESULTS_DIR, 'predictions.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, list(le.classes_), rotation=45, ha='right')
    plt.yticks(tick_marks, list(le.classes_))
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

    # 3. Per-Class Accuracy Plot
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        accs = cm.diagonal() / cm.sum(axis=1)
        accs = np.nan_to_num(accs) # Replace NaNs with 0
        
    plt.figure(figsize=(12, 6))
    bars = plt.bar(list(le.classes_), accs, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    acc_path = os.path.join(config.RESULTS_DIR, 'per_class_accuracy.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy Plot saved to {acc_path}")

if __name__ == '__main__':
    main()