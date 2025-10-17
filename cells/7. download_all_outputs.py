# ============================================================================
# DOWNLOAD ALL GENERATED FILES
# ============================================================================
# This cell packages and downloads all files generated during training
# ============================================================================

import os
import shutil
from datetime import datetime

print("=" * 80)
print("PACKAGING ALL GENERATED FILES FOR DOWNLOAD")
print("=" * 80)

# Create a downloads directory
download_dir = '/content/project_outputs'
os.makedirs(download_dir, exist_ok=True)

# List of files to download
files_to_download = {
    # Model files
    '/content/best_model.pt': 'model/best_model.pt',
    
    # Training outputs
    '/content/training_history.png': 'training/training_history.png',
    
    # Evaluation outputs
    '/content/evaluation_results.png': 'evaluation/evaluation_results.png',
    '/content/human_evaluation_samples.csv': 'evaluation/human_evaluation_samples.csv',
    
    # Preprocessed data
    '/content/preprocessed_data/vocab.json': 'preprocessed/vocab.json',
    '/content/preprocessed_data/id_to_token.json': 'preprocessed/id_to_token.json',
    '/content/preprocessed_data/config.json': 'preprocessed/config.json',
    '/content/preprocessed_data/train_df.csv': 'preprocessed/train_df.csv',
    '/content/preprocessed_data/val_df.csv': 'preprocessed/val_df.csv',
    '/content/preprocessed_data/test_df.csv': 'preprocessed/test_df.csv',
}

print("\nCopying files to download directory...\n")

copied_files = []
missing_files = []

for source, destination in files_to_download.items():
    dest_path = os.path.join(download_dir, destination)
    
    # Create subdirectories if needed
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(source):
        shutil.copy2(source, dest_path)
        file_size = os.path.getsize(source) / (1024 * 1024)  # Convert to MB
        print(f"✓ Copied: {destination} ({file_size:.2f} MB)")
        copied_files.append(destination)
    else:
        print(f"✗ Missing: {source}")
        missing_files.append(source)

print("\n" + "=" * 80)
print(f"SUMMARY: {len(copied_files)} files copied, {len(missing_files)} missing")
print("=" * 80)

# Create a README file
readme_content = f"""# Empathetic Chatbot - Project Outputs
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure:

### 📁 model/
- best_model.pt - Best trained model (Epoch 17, BLEU: 14.25)

### 📁 training/
- training_history.png - Training curves and metrics visualization

### 📁 evaluation/
- evaluation_results.png - Test set evaluation metrics
- human_evaluation_samples.csv - Samples for human evaluation

### 📁 preprocessed/
- vocab.json - Vocabulary (token -> id mapping)
- id_to_token.json - Reverse vocabulary (id -> token)
- config.json - Configuration and hyperparameters
- train_df.csv, val_df.csv, test_df.csv - Preprocessed datasets

## Model Performance:

### Training:
- Best Validation BLEU: 14.25 (Epoch 17)
- Final Training Loss: 3.46
- Validation Perplexity: 117.14

### Test Set:
- BLEU Score: 97.85
- ROUGE-L Score: 0.1522
- chrF Score: 74.04
- Perplexity: 48.92

## Model Architecture:
- Transformer Encoder-Decoder (from scratch)
- Embedding dimension: 512
- Attention heads: 8
- Encoder layers: 2
- Decoder layers: 2
- Total parameters: 31,589,457

## Training Details:
- Dataset: 64,636 empathetic dialogues
- Training samples: 51,708
- Validation samples: 6,463
- Test samples: 6,465
- Vocabulary size: 16,465
- Optimizer: Adam (betas=0.9, 0.98)
- Learning rate: 2e-4
- Batch size: 64

## Files Copied:
"""

for file in copied_files:
    readme_content += f"✓ {file}\n"

if missing_files:
    readme_content += "\n## Missing Files:\n"
    for file in missing_files:
        readme_content += f"✗ {file}\n"

readme_path = os.path.join(download_dir, 'README.md')
with open(readme_path, 'w') as f:
    f.write(readme_content)

print("\n✓ README.md created")

# Create a zip file
print("\n" + "=" * 80)
print("CREATING ZIP ARCHIVE")
print("=" * 80)

zip_filename = '/content/empathetic_chatbot_outputs.zip'
shutil.make_archive(
    zip_filename.replace('.zip', ''),
    'zip',
    download_dir
)

zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
print(f"\n✓ Zip file created: empathetic_chatbot_outputs.zip ({zip_size:.2f} MB)")

# Download the zip file
print("\n" + "=" * 80)
print("DOWNLOADING ZIP FILE")
print("=" * 80)

from google.colab import files
files.download(zip_filename)

print("\n✓ Download started! Check your browser's downloads folder.")
print("=" * 80)

# Also offer individual downloads
print("\n" + "=" * 80)
print("INDIVIDUAL FILE DOWNLOADS (Optional)")
print("=" * 80)
print("\nIf you want to download individual files, uncomment the lines below:\n")

print("""
# Download model
# files.download('/content/best_model.pt')

# Download training plot
# files.download('/content/training_history.png')

# Download evaluation plot
# files.download('/content/evaluation_results.png')

# Download human evaluation CSV
# files.download('/content/human_evaluation_samples.csv')

# Download config
# files.download('/content/preprocessed_data/config.json')
""")

print("=" * 80)
print("✅ DOWNLOAD COMPLETE!")
print("=" * 80)
print("\nYou now have:")
print("  📦 empathetic_chatbot_outputs.zip")
print("  📄 Contains: model, plots, data, and README")
print("  📊 Total files: " + str(len(copied_files) + 1))  # +1 for README
print("=" * 80)

