# 🕸️ Hybrid Webpage Classifier

A multimodal deep learning project that classifies URLs into categories (Sports, News, Shopping, etc.) by fusing **semantic text analysis** with **structural URL feature extraction**.

## 🚀 Project Overview

Traditional URL classifiers often rely solely on keywords. This project takes a hybrid approach:
1. **Semantic Analysis**: A bidirectional LSTM processes the text within the URL to understand context.
2. **Structural Analysis**: A Dense Neural Network extracts engineering features (e.g., path depth, subdomain count, special character density).

The two streams are fused in a final fully connected layer to make a prediction. This allows the model to detect patterns like `subdomain.site.com/login` (structure) alongside `.../football-scores` (semantic).

## 🛠️ Tech Stack
* **Core**: Python 3.9+
* **Deep Learning**: PyTorch (LSTM + Custom Fusion Network)
* **Data Processing**: Pandas, NumPy, Scikit-Learn
* **Visualization**: Matplotlib

## 🧠 Key Features
* **Multimodal Architecture**: Combines `nn.LSTM` and `nn.Linear` layers.
* **Feature Engineering**: Custom extractor for 14 specific URL characteristics (TLD length, HTTPS usage, etc.).
* **Focal Loss**: Implemented to handle class imbalance in the training dataset.
* **Early Stopping**: Automatically halts training when validation accuracy plateaus to prevent overfitting.

## 📂 Project Structure
```
├── data/ 
│ └── raw/ # Place your CSV dataset here 
├── models/ # Saved .pt models and .pkl artifacts 
├── results/ # Confusion matrices and prediction CSVs 
├── scripts/ │ ├── config.py # Hyperparameters and paths 
│ ├── dataset.py # Data loading and preprocessing 
│ ├── model.py # PyTorch model definition 
│ ├── train.py # Training loop 
│ ├── predict.py # Inference script for testing 
│ └── utils.py # Helper functions 
└── README.md
```

## 🔧 Setup & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib tldextract tqdm
```
2. Prepare Data
Ensure your dataset is at data/raw/URL Classification.csv. It should be a CSV where one column contains URLs and another contains the category label.

3. Training
Train the model. It will automatically save the best version to models/best_model.pt.

```Bash
python scripts/train.py
```
4. Evaluation
Generate confusion matrices and accuracy plots in the results/ folder:

```Bash
python scripts/visualize_results.py
```
5. Prediction
Test the model with your own URLs interactively:

```Bash
python scripts/predict.py
```
---

### 🟢 How to Run and Verify (Execution Guide)

Follow these exact steps to ensure everything works on your machine before the interview.

**Step 1: Setup the Folder Structure**
Create a folder named `multimodal_webpage_classification` and inside it create a `scripts` folder and a `data/raw` folder.
* Put all the python files above into `scripts/`.
* Put your `URL Classification.csv` file into `data/raw/`.

**Step 2: Install Dependencies**
Open your terminal (Command Prompt or PowerShell) and run:
```bash
pip install torch pandas numpy scikit-learn matplotlib tldextract tqdm
```
Step 3: Run Training Run the training script to generate the model files.

```Bash
python scripts/train.py
```
Expected Output: You will see a progress bar for "Training". At the end of every epoch, you should see Train Loss, Val Loss, and Val Acc. Look for the message --> New best model saved.

Step 4: Verify Artifacts Check the models/ folder. You should see three files:

best_model.pt

label_encoder.pkl

text_vocab.pkl

Step 5: Run Prediction Run the interactive predictor:

```Bash
python scripts/predict.py
```
Input: Enter https://www.amazon.com/gp/product/B08 when prompted. Expected Output: The model should output Top Prediction: >>> SHOPPING <<< (or similar category) with a confidence score.