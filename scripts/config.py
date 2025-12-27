import os

# --- Project Structure ---
# Get the absolute path of the project root (one level up from this scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define directories using the project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Assuming the file from curlie.org is saved as 'dataset.csv' or similar
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'URL Classification.csv') 
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Data Splitting ---
TEST_SIZE = 0.2        # Reserved 20% for final testing
VALIDATION_SIZE = 0.1  # 10% of training data used for validation
RANDOM_SEED = 42       # Set seed for reproducibility during presentations

# --- Model & Tokenization Hyperparameters ---
# Max length of 512 is still good for Title + Description
MAX_TEXT_LENGTH = 512  
# Increased vocab size slightly since Title/Descriptions use more diverse language than just URLs
VOCAB_SIZE = 20000     
EMBED_DIM = 128        # Dimension for our dense vector embeddings

# --- Training Hyperparameters ---
EPOCHS = 30            # Max epochs (Early Stopping will likely stop it sooner)
BATCH_SIZE = 128       # Larger batch size stabilizes gradient updates
LEARNING_RATE = 0.001  # Standard starting point for Adam optimizer