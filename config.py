import os

# Directory information
ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, 'models/')
BASELINE_DIR = os.path.join(MODELS_DIR, 'model_baseline')

# Data information
Image_size = 28

# Divsision
Validate_perc = 0.2

# Training information
Max_epochs = 15
Epochs = 10
Batch_size = 32
Use_pretrained = False
