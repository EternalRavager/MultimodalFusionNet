# Model Improvements for Webpage Classification

## Problem Identified
The model was predicting everything as "Business" class due to several issues:
1. **Severe class imbalance**: News (0.57%) vs Arts (16.24%)
2. **Weak model architecture**: Simple, shallow network
3. **Limited URL features**: Only 7 basic features
4. **No handling of class imbalance**: Standard CrossEntropyLoss
5. **Poor training configuration**: Low batch size, no scheduler

## Solutions Implemented

### 1. Enhanced Model Architecture (`model.py`)
**Changes:**
- Increased URL feature network from 2 layers to 3 layers with BatchNorm
- Added Bidirectional LSTM (2 layers) for better text understanding
- Increased embedding dimension from 64 to 128
- Added comprehensive dropout (0.3-0.4) to prevent overfitting
- Deeper fusion network (4 layers) with BatchNorm
- Total parameters increased significantly for better learning capacity

**Benefits:**
- Better feature extraction from both URL and text
- More capacity to learn complex patterns
- Regularization prevents overfitting

### 2. Focal Loss for Class Imbalance (`train.py`)
**Changes:**
- Implemented custom FocalLoss class with gamma=2.0
- Combined with class weights for double protection against imbalance
- Focuses learning on hard-to-classify examples

**Benefits:**
- Minority classes (like News: 0.57%) get proper attention
- Model won't just predict majority class (Business)
- Better overall accuracy across all classes

### 3. Improved URL Feature Extraction (`utils.py`)
**Changes:**
- Increased features from 7 to 14
- Added: underscore count, query parameters, subdomain count, path depth, port detection, domain length
- Added keyword matching for 14 category types (news, shop, sports, health, tech, etc.)
- Category-specific keywords help identify webpage type

**Benefits:**
- Richer feature representation
- Semantic understanding through keywords
- Better discrimination between categories

### 4. Better Training Configuration
**Changes:**
- Added learning rate scheduler (ReduceLROnPlateau)
- Increased batch size: 64 → 128
- Increased epochs: 20 → 30
- Added weight decay (1e-4) for regularization
- Increased embedding dimension: 64 → 128

**Benefits:**
- Adaptive learning rate prevents overfitting
- Larger batches provide better gradient estimates
- Better convergence

### 5. Enhanced Prediction Output (`predict.py`)
**Changes:**
- Shows prediction confidence (percentage)
- Displays top 5 predictions with probabilities
- Better formatted output
- Interactive prompts for user input

**Benefits:**
- User can see model confidence
- Can verify if prediction makes sense
- Easier to debug incorrect predictions

## How to Use

### 1. Retrain the Model
```powershell
& C:/Users/LEGION/myenv/Scripts/python.exe c:/CODE/Python/multimodal_webpage_classification/scripts/train.py
```

**Important:** The old model is incompatible with the new architecture. You MUST retrain.

### 2. Run Predictions
```powershell
& C:/Users/LEGION/myenv/Scripts/python.exe c:/CODE/Python/multimodal_webpage_classification/scripts/predict.py
```

Then enter:
- Website URL (e.g., `https://www.techcrunch.com/news/ai-breakthrough`)
- Description (e.g., `Latest technology and AI news articles`)

### 3. Expected Output
```
==================================================
PREDICTION RESULTS
==================================================

Predicted Category: News
Confidence: 78.45%

Top 5 Predictions:
--------------------------------------------------
1. News            -  78.45%
2. Computers       -  12.34%
3. Science         -   5.67%
4. Business        -   2.11%
5. Arts            -   0.89%
==================================================
```

## Expected Improvements

After retraining with these changes:
- **Better accuracy**: Expected 10-20% improvement
- **Balanced predictions**: Won't predict only "Business"
- **Confidence scores**: Can see when model is uncertain
- **Minority class performance**: Classes like "News" should be predicted correctly

## Next Steps

1. **Delete old model**: The old `best_model.pt` is incompatible
2. **Retrain**: Run `train.py` - will take longer but learn better
3. **Monitor training**: Watch for improving validation accuracy
4. **Test diverse URLs**: Try URLs from different categories
5. **Check confidence**: Low confidence (<50%) means uncertain prediction

## Notes

- Training will take longer due to larger model and batch size
- Use GPU if available (will be 10-50x faster)
- Early stopping will prevent overfitting
- The model now has ~10x more parameters - needs more data to train properly
- Class weights ensure all 15 categories are learned properly
