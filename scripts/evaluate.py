import torch
from torch.utils.data import DataLoader
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
import scripts.config as config
from scripts.dataset import MultimodalWebpageDataset, prepare_data
from scripts.model import MultimodalFusionNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")

    # Load artifacts
    vocab_path = os.path.join(config.MODELS_DIR, 'text_vocab.pkl')
    le_path = os.path.join(config.MODELS_DIR, 'label_encoder.pkl')
    model_path = os.path.join(config.MODELS_DIR, 'best_model.pt')
    
    if not os.path.exists(model_path):
        print("Model not found! Run train.py first.")
        return

    with open(vocab_path, 'rb') as f:
        text_vocab = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    # Get test data
    # Note: We re-run prepare_data to get the exact same split, 
    # but in a real production env, we'd save the test split to disk.
    _, _, test_df, _, _ = prepare_data()
    
    test_set = MultimodalWebpageDataset(test_df, text_vocab, config.MAX_TEXT_LENGTH)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE)

    # Initialize model
    model = MultimodalFusionNet(
        num_url_features=14,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=len(le.classes_)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    true_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for url_feats, text_seq, label in test_loader:
            url_feats, text_seq = url_feats.to(device), text_seq.to(device)
            out = model(url_feats, text_seq)
            
            y_hat = out.argmax(1).cpu().numpy()
            preds.extend(y_hat)
            true_labels.extend(label.numpy())

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, preds, target_names=le.classes_))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX (Row=True, Col=Pred)")
    print("="*60)
    print(confusion_matrix(true_labels, preds))

if __name__ == '__main__':
    main()