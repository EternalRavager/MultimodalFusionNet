import torch
import pickle
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.config as config
from scripts.model import MultimodalFusionNet
from scripts.utils import extract_url_features, clean_text

def predict(url, title, description):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Artifacts
    try:
        with open(os.path.join(config.MODELS_DIR, 'text_vocab.pkl'), 'rb') as f:
            text_vocab = pickle.load(f)
        with open(os.path.join(config.MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model artifacts not found. Have you trained the model yet?")
        return

    # Initialize Model
    model = MultimodalFusionNet(
        num_url_features=14,
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_classes=len(le.classes_)
    ).to(device)
    
    model_path = os.path.join(config.MODELS_DIR, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- Preprocess Input ---
    
    # 1. URL Features (Structural)
    url_feats = torch.tensor([extract_url_features(url)], dtype=torch.float32).to(device)
    
    # 2. Text Sequence (Semantic: Title + Description)
    # We combine them just like we did in dataset.py
    raw_text = f"{title} {description}"
    cleaned_text = clean_text(raw_text)
    words = cleaned_text.split()
    
    ids = [text_vocab.get(w, 1) for w in words] # 1 is <unk>
    
    # Pad/Truncate
    if len(ids) > config.MAX_TEXT_LENGTH:
        ids = ids[:config.MAX_TEXT_LENGTH]
    else:
        ids += [0] * (config.MAX_TEXT_LENGTH - len(ids))
        
    text_seq = torch.tensor([ids], dtype=torch.long).to(device)
    
    # --- Inference ---
    with torch.no_grad():
        out = model(url_feats, text_seq)
        probabilities = torch.softmax(out, dim=1)[0]
    
    # --- Results ---
    top_probs, top_indices = torch.topk(probabilities, min(5, len(le.classes_)))
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"URL:   {url}")
    print(f"Title: {title}")
    print("-" * 50)
    
    top_category = le.inverse_transform([top_indices[0].item()])[0]
    print(f"Top Prediction:  >>> {top_category} <<<")
    print(f"Confidence:      {top_probs[0].item()*100:.2f}%")
    
    print("\nAlternative Probabilities:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        category = le.inverse_transform([idx.item()])[0]
        print(f"{i}. {str(category):15s} : {prob.item()*100:6.2f}%")
    print("="*50)

if __name__ == '__main__':
    print("=" * 50)
    print("Webpage Classification Engine")
    print("=" * 50)
    
    # Interactive Input
    url_in = input("\nEnter website URL: ").strip()
    title_in = input("Enter website Title: ").strip()
    desc_in = input("Enter website Description: ").strip()
    
    print("\nProcessing...")
    predict(url_in, title_in, desc_in)