# Quick Start Guide - Retrain and Test

## IMPORTANT: You MUST Retrain the Model

The old model file is **incompatible** with the new architecture. Follow these steps:

### Step 1: Delete Old Model (Optional but Recommended)
```powershell
Remove-Item "C:\CODE\Python\multimodal_webpage_classification\models\best_model.pt"
```

### Step 2: Start Training
```powershell
cd C:\CODE\Python\multimodal_webpage_classification
& C:/Users/LEGION/myenv/Scripts/python.exe scripts/train.py
```

**What to expect:**
- Training will show progress bars for each epoch
- You'll see Train Loss, Train Acc, Val Loss, Val Acc for each epoch
- Model will save when validation accuracy improves
- Early stopping after 5 epochs with no improvement
- Training may take 30-120 minutes depending on your hardware

**Good Training Signs:**
- Train accuracy increasing (target: >80%)
- Validation accuracy increasing (target: >70%)
- Loss decreasing
- "Saved new best model" messages

**Bad Training Signs:**
- Train accuracy stuck at same value
- All predictions same class
- Loss not decreasing

### Step 3: Test the Model

After training completes, test predictions:

```powershell
& C:/Users/LEGION/myenv/Scripts/python.exe scripts/predict.py
```

**Test Cases:**

1. **News Website**
   - URL: `https://www.nytimes.com/news/breaking`
   - Text: `Breaking news and latest updates`
   - Expected: News category with high confidence

2. **Shopping Website**
   - URL: `https://www.amazon.com/shop/electronics`
   - Text: `Buy electronics and products online`
   - Expected: Shopping category

3. **Sports Website**
   - URL: `https://www.espn.com/nfl/game`
   - Text: `NFL football scores and highlights`
   - Expected: Sports category

4. **Technology Website**
   - URL: `https://www.techcrunch.com/startups`
   - Text: `Latest technology and startup news`
   - Expected: Computers or Business

### Step 4: Evaluate Results

Run full evaluation:
```powershell
& C:/Users/LEGION/myenv/Scripts/python.exe scripts/evaluate.py
```

This will show:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Classification report

### Troubleshooting

**Problem: Still predicting mostly one class**
- Check training accuracy - should be >80%
- Verify data loaded correctly
- May need more epochs (increase in config.py)

**Problem: Training is very slow**
- Reduce BATCH_SIZE in config.py (128 → 64 → 32)
- Reduce VOCAB_SIZE (15000 → 10000)
- Use GPU if available

**Problem: Out of memory**
- Reduce BATCH_SIZE to 32 or 16
- Reduce MAX_TEXT_LENGTH to 256

**Problem: Low accuracy (<60%)**
- Train longer (increase EPOCHS)
- Check if data is balanced after loading
- Verify URL features are extracting correctly

### Expected Performance

After proper training:
- **Overall Accuracy**: 70-85%
- **Business Class**: 75-90% (was 100% before, predicting everything)
- **News Class**: 40-70% (was 0% before, never predicted)
- **Common Classes (Arts, Society)**: 70-85%
- **Rare Classes (Kids, Adult)**: 50-70%

The key improvement is that **all classes should now be predicted**, not just Business!
