# ============================================================================
# CELL 1: TASK 1 - PREPROCESSING
# ============================================================================
# Empathetic Conversational Chatbot - Transformer with MultiHead Attention
# 
# This cell handles complete data preprocessing including:
# - Loading and exploring the dataset
# - Text normalization (lowercase, clean whitespace, punctuation)
# - Tokenization and vocabulary building from training data only
# - Adding special tokens (<pad>, <bos>, <eos>, <unk>, <sep>)
# - Train/Val/Test split (80/10/10)
# - Saving preprocessed data for next tasks
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Install and Import Required Libraries
# ----------------------------------------------------------------------------
print("=" * 80)
print("INSTALLING REQUIRED LIBRARIES...")
print("=" * 80)

!pip install pandas numpy matplotlib seaborn tqdm -q

import pandas as pd
import numpy as np
import re
import pickle
import json
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import io

# Set random seed for reproducibility
np.random.seed(42)

print("✓ All libraries imported successfully!\n")


# ----------------------------------------------------------------------------
# STEP 2: Upload Dataset File
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: UPLOAD DATASET")
print("=" * 80)
print("Please upload the emotion-emotion_69k.csv file:")
print("Click 'Choose Files' button that appears below and select your CSV file\n")

from google.colab import files
uploaded = files.upload()

# Get the uploaded file name
uploaded_filename = list(uploaded.keys())[0]
DATASET_PATH = uploaded_filename

print("\n" + "=" * 80)
print(f"✓ File uploaded successfully: {uploaded_filename}")
print(f"✓ File size: {len(uploaded[uploaded_filename]) / 1024 / 1024:.2f} MB")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 3: Load and Explore the Dataset
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 2: LOADING AND EXPLORING DATASET")
print("=" * 80)

# Load the dataset
df = pd.read_csv(DATASET_PATH)

print(f"Total number of rows: {len(df)}")
print(f"Total number of columns: {len(df.columns)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(3))
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDataset shape: {df.shape}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 4: Parse Dialogues to Extract Customer and Agent Text
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 3: PARSING DIALOGUES")
print("=" * 80)

def parse_dialogue(dialogue_text, labels_text):
    """
    Parse the empathetic_dialogues and labels columns to extract customer and agent text.
    
    The dataset structure:
    - empathetic_dialogues: contains "Customer :<customer_text>\nAgent :"
    - labels: contains the agent's reply
    
    Args:
        dialogue_text: String from empathetic_dialogues column
        labels_text: String from labels column (agent reply)
    
    Returns:
        tuple: (customer_utterance, agent_reply)
    """
    if pd.isna(dialogue_text) or pd.isna(labels_text):
        return None, None
    
    # Extract customer utterance from empathetic_dialogues
    # It contains "Customer :<text>\nAgent :"
    customer_utterance = dialogue_text
    
    # Remove "Customer :" prefix if present
    if "Customer :" in customer_utterance:
        customer_utterance = customer_utterance.split("Customer :")[1]
    
    # Remove "\nAgent :" or "Agent :" suffix if present
    if "\nAgent :" in customer_utterance:
        customer_utterance = customer_utterance.split("\nAgent :")[0]
    elif "Agent :" in customer_utterance:
        customer_utterance = customer_utterance.split("Agent :")[0]
    
    customer_utterance = customer_utterance.strip()
    
    # Agent reply is directly from labels column
    agent_reply = str(labels_text).strip()
    
    # Only return valid pairs
    if customer_utterance and agent_reply and agent_reply != 'nan':
        return customer_utterance, agent_reply
    
    return None, None


# Apply parsing to the dataset
parsed_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dialogues"):
    customer_utterance, agent_reply = parse_dialogue(
        row['empathetic_dialogues'], 
        row['labels']
    )
    
    # Only keep rows where both customer and agent text exist
    if customer_utterance and agent_reply:
        parsed_data.append({
            'situation': row['Situation'] if pd.notna(row['Situation']) else '',
            'emotion': row['emotion'] if pd.notna(row['emotion']) else '',
            'customer_utterance': customer_utterance,
            'agent_reply': agent_reply
        })

# Create a new clean dataframe
df_clean = pd.DataFrame(parsed_data)

print(f"\n✓ Total valid dialogues parsed: {len(df_clean)}")
print(f"\nEmotion distribution:")
print(df_clean['emotion'].value_counts())
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 5: Define and Apply Text Normalization
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 4: TEXT NORMALIZATION")
print("=" * 80)

def normalize_text(text):
    """
    Normalize text by:
    - Converting to lowercase
    - Normalizing whitespace (replace multiple spaces with single space)
    - Normalizing punctuation (add spaces around punctuation marks)
    - Expanding contractions
    - Removing extra whitespace
    
    Args:
        text: Input string to normalize
    
    Returns:
        Normalized string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize common contractions (helps with consistency)
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Add spaces around punctuation marks for better tokenization
    text = re.sub(r'([.!?,;:\"\'])', r' \1 ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


# Normalize all text columns
print("Normalizing text fields...")
df_clean['situation_normalized'] = df_clean['situation'].apply(normalize_text)
df_clean['customer_normalized'] = df_clean['customer_utterance'].apply(normalize_text)
df_clean['agent_normalized'] = df_clean['agent_reply'].apply(normalize_text)
df_clean['emotion_normalized'] = df_clean['emotion'].str.lower().str.strip()

print("✓ Text normalization complete!")
print("\nNormalization examples:")
for i in range(2):
    print(f"\nExample {i+1}:")
    print(f"  Original: {df_clean.iloc[i]['customer_utterance']}")
    print(f"  Normalized: {df_clean.iloc[i]['customer_normalized']}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 6: Train/Validation/Test Split (80/10/10)
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 5: DATASET SPLITTING")
print("=" * 80)

# Shuffle the dataset
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split sizes
total_size = len(df_clean)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Split the data
train_df = df_clean[:train_size].reset_index(drop=True)
val_df = df_clean[train_size:train_size+val_size].reset_index(drop=True)
test_df = df_clean[train_size+val_size:].reset_index(drop=True)

print(f"Total samples: {total_size}")
print(f"Training samples: {len(train_df)} ({len(train_df)/total_size*100:.1f}%)")
print(f"Validation samples: {len(val_df)} ({len(val_df)/total_size*100:.1f}%)")
print(f"Test samples: {len(test_df)} ({len(test_df)/total_size*100:.1f}%)")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 7: Define Tokenization Function
# ----------------------------------------------------------------------------
def simple_tokenize(text):
    """
    Simple word-level tokenizer that splits text by whitespace.
    The text should already be normalized before tokenization.
    
    Args:
        text: Input string (should be normalized)
    
    Returns:
        List of tokens (words)
    """
    if not isinstance(text, str):
        return []
    
    # Split by whitespace
    tokens = text.split()
    
    return tokens


# ----------------------------------------------------------------------------
# STEP 8: Build Vocabulary from Training Data ONLY
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 6: BUILDING VOCABULARY (from training data only)")
print("=" * 80)

def build_vocabulary(dataframe, min_freq=2):
    """
    Build vocabulary from the training dataset.
    
    Args:
        dataframe: Training dataframe with normalized text
        min_freq: Minimum frequency for a word to be included in vocabulary
    
    Returns:
        Dictionary mapping tokens to IDs
    """
    # Counter to count word frequencies
    word_counter = Counter()
    
    # Count words from all text fields in training data
    print("Counting word frequencies in training data...")
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Building vocab"):
        # Tokenize and count from all fields
        situation_tokens = simple_tokenize(row['situation_normalized'])
        word_counter.update(situation_tokens)
        
        customer_tokens = simple_tokenize(row['customer_normalized'])
        word_counter.update(customer_tokens)
        
        agent_tokens = simple_tokenize(row['agent_normalized'])
        word_counter.update(agent_tokens)
    
    print(f"\nTotal unique words found: {len(word_counter)}")
    print(f"Total word occurrences: {sum(word_counter.values())}")
    
    # Filter words by minimum frequency
    filtered_words = [word for word, freq in word_counter.items() if freq >= min_freq]
    print(f"Words with frequency >= {min_freq}: {len(filtered_words)}")
    
    # Define special tokens (these are crucial for the model)
    special_tokens = [
        '<pad>',  # Padding token - for making all sequences same length
        '<unk>',  # Unknown token - for words not in vocabulary
        '<bos>',  # Beginning of sequence - marks start of target
        '<eos>',  # End of sequence - marks end of target
        '<sep>',  # Separator token - separates input components
    ]
    
    # Build vocabulary: token -> id mapping
    vocab = {}
    
    # Add special tokens first (they get the lowest IDs: 0, 1, 2, 3, 4)
    for idx, token in enumerate(special_tokens):
        vocab[token] = idx
    
    # Add regular words (sorted alphabetically for consistency)
    for idx, word in enumerate(sorted(filtered_words), start=len(special_tokens)):
        vocab[word] = idx
    
    print(f"\n✓ Vocabulary built successfully!")
    print(f"Final vocabulary size: {len(vocab)}")
    
    return vocab, word_counter


# Build vocabulary from TRAINING data only (important: not from val/test!)
MIN_FREQUENCY = 2  # Words must appear at least 2 times

vocab, word_freq = build_vocabulary(train_df, min_freq=MIN_FREQUENCY)

# Create reverse vocabulary (id -> token) for decoding later
id_to_token = {idx: token for token, idx in vocab.items()}

print("\nSpecial tokens and their IDs:")
special_tokens_list = ['<pad>', '<unk>', '<bos>', '<eos>', '<sep>']
for token in special_tokens_list:
    print(f"  {token}: {vocab[token]}")

print(f"\nMost common words in training data (top 15):")
for word, freq in word_freq.most_common(15):
    print(f"  '{word}': {freq} occurrences")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 9: Token-ID Conversion Functions
# ----------------------------------------------------------------------------
def tokens_to_ids(tokens, vocab):
    """
    Convert list of tokens to list of IDs using vocabulary.
    Unknown tokens are mapped to <unk> token ID.
    
    Args:
        tokens: List of token strings
        vocab: Vocabulary dictionary (token -> id)
    
    Returns:
        List of token IDs
    """
    unk_id = vocab['<unk>']
    return [vocab.get(token, unk_id) for token in tokens]


def ids_to_tokens(ids, id_to_token):
    """
    Convert list of IDs to list of tokens.
    
    Args:
        ids: List of token IDs
        id_to_token: Reverse vocabulary dictionary (id -> token)
    
    Returns:
        List of tokens
    """
    return [id_to_token.get(idx, '<unk>') for idx in ids]


# ----------------------------------------------------------------------------
# STEP 10: Create Input-Output Pairs with Special Tokens
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 7: PREPARING INPUT-OUTPUT PAIRS")
print("=" * 80)

def prepare_input_output(row, vocab):
    """
    Prepare input and output sequences with special tokens.
    
    Input format (X): 
        emotion: {emotion} <sep> situation: {situation} <sep> customer: {customer} <sep>
    
    Target format (Y): 
        <bos> {agent_reply} <eos>
    
    Args:
        row: DataFrame row containing normalized text
        vocab: Vocabulary dictionary
    
    Returns:
        tuple: (input_ids, target_ids)
    """
    # Prepare input text with structure
    input_text = f"emotion : {row['emotion_normalized']} <sep> situation : {row['situation_normalized']} <sep> customer : {row['customer_normalized']} <sep>"
    
    # Tokenize input (handle <sep> token specially)
    input_tokens = []
    for token in input_text.split():
        if token == '<sep>':
            input_tokens.append('<sep>')
        else:
            input_tokens.append(token)
    
    # Convert input tokens to IDs
    input_ids = tokens_to_ids(input_tokens, vocab)
    
    # Prepare target text with <bos> and <eos> markers
    target_text = row['agent_normalized']
    target_tokens = ['<bos>'] + simple_tokenize(target_text) + ['<eos>']
    
    # Convert target tokens to IDs
    target_ids = tokens_to_ids(target_tokens, vocab)
    
    return input_ids, target_ids


# Test on sample examples
print("Input-Output format examples:\n")
for i in range(2):
    row = train_df.iloc[i]
    input_ids, target_ids = prepare_input_output(row, vocab)
    
    print(f"Example {i+1}:")
    print(f"  Emotion: {row['emotion_normalized']}")
    print(f"  Situation: {row['situation_normalized'][:60]}...")
    print(f"  Customer: {row['customer_normalized']}")
    print(f"  Agent: {row['agent_normalized']}")
    print(f"  Input length: {len(input_ids)} tokens")
    print(f"  Target length: {len(target_ids)} tokens")
    print(f"  Target tokens: {ids_to_tokens(target_ids, id_to_token)}")
    print()

print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 11: Process All Datasets (Train/Val/Test)
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 8: PROCESSING ALL DATASETS")
print("=" * 80)

def process_dataset(dataframe, vocab, split_name):
    """
    Process entire dataset to create input-output pairs.
    
    Args:
        dataframe: DataFrame to process
        vocab: Vocabulary dictionary
        split_name: Name of the split (for logging)
    
    Returns:
        List of dictionaries containing input_ids and target_ids
    """
    processed_data = []
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {split_name}"):
        input_ids, target_ids = prepare_input_output(row, vocab)
        
        processed_data.append({
            'input_ids': input_ids,
            'target_ids': target_ids,
            'emotion': row['emotion_normalized'],
            'input_length': len(input_ids),
            'target_length': len(target_ids)
        })
    
    return processed_data


# Process all three splits
train_processed = process_dataset(train_df, vocab, "Training")
val_processed = process_dataset(val_df, vocab, "Validation")
test_processed = process_dataset(test_df, vocab, "Test")

print(f"\n✓ Training samples processed: {len(train_processed)}")
print(f"✓ Validation samples processed: {len(val_processed)}")
print(f"✓ Test samples processed: {len(test_processed)}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 12: Analyze Sequence Lengths
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 9: ANALYZING SEQUENCE LENGTHS")
print("=" * 80)

# Extract sequence lengths from training data
train_input_lengths = [sample['input_length'] for sample in train_processed]
train_target_lengths = [sample['target_length'] for sample in train_processed]

# Calculate statistics
print("\nInput sequences (X) statistics:")
print(f"  Mean length: {np.mean(train_input_lengths):.2f}")
print(f"  Median length: {np.median(train_input_lengths):.2f}")
print(f"  Min length: {np.min(train_input_lengths)}")
print(f"  Max length: {np.max(train_input_lengths)}")
print(f"  95th percentile: {np.percentile(train_input_lengths, 95):.2f}")
print(f"  99th percentile: {np.percentile(train_input_lengths, 99):.2f}")

print("\nTarget sequences (Y) statistics:")
print(f"  Mean length: {np.mean(train_target_lengths):.2f}")
print(f"  Median length: {np.median(train_target_lengths):.2f}")
print(f"  Min length: {np.min(train_target_lengths)}")
print(f"  Max length: {np.max(train_target_lengths)}")
print(f"  95th percentile: {np.percentile(train_target_lengths, 95):.2f}")
print(f"  99th percentile: {np.percentile(train_target_lengths, 99):.2f}")

# Suggest max lengths based on 95th percentile (captures most data while avoiding outliers)
suggested_max_input_len = int(np.percentile(train_input_lengths, 95))
suggested_max_target_len = int(np.percentile(train_target_lengths, 95))

print(f"\n💡 SUGGESTED MAX LENGTHS (based on 95th percentile):")
print(f"  Max input length: {suggested_max_input_len}")
print(f"  Max target length: {suggested_max_target_len}")
print("  (These will be used for padding in the model)")

# Visualize distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_input_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(suggested_max_input_len, color='red', linestyle='--', linewidth=2,
                label=f'95th percentile: {suggested_max_input_len}')
axes[0].set_xlabel('Sequence Length', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Input Sequence Length Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(train_target_lengths, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
axes[1].axvline(suggested_max_target_len, color='red', linestyle='--', linewidth=2,
                label=f'95th percentile: {suggested_max_target_len}')
axes[1].set_xlabel('Sequence Length', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Target Sequence Length Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 13: Save Preprocessed Data and Vocabulary
# ----------------------------------------------------------------------------
print("=" * 80)
print("STEP 10: SAVING PREPROCESSED DATA")
print("=" * 80)

# Create directory to save all preprocessed data
save_dir = '/content/preprocessed_data'
os.makedirs(save_dir, exist_ok=True)

# Save vocabulary (token -> id mapping)
with open(f'{save_dir}/vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

# Save reverse vocabulary (id -> token mapping)
with open(f'{save_dir}/id_to_token.json', 'w') as f:
    json.dump(id_to_token, f, indent=2)

print("✓ Vocabulary saved!")

# Save processed datasets (in pickle format for efficiency)
with open(f'{save_dir}/train_processed.pkl', 'wb') as f:
    pickle.dump(train_processed, f)

with open(f'{save_dir}/val_processed.pkl', 'wb') as f:
    pickle.dump(val_processed, f)

with open(f'{save_dir}/test_processed.pkl', 'wb') as f:
    pickle.dump(test_processed, f)

print("✓ Processed datasets saved!")

# Save original dataframes (for reference and debugging)
train_df.to_csv(f'{save_dir}/train_df.csv', index=False)
val_df.to_csv(f'{save_dir}/val_df.csv', index=False)
test_df.to_csv(f'{save_dir}/test_df.csv', index=False)

print("✓ Original dataframes saved!")

# Save configuration file with all important parameters
config = {
    'vocab_size': len(vocab),
    'min_frequency': MIN_FREQUENCY,
    'train_size': len(train_processed),
    'val_size': len(val_processed),
    'test_size': len(test_processed),
    'suggested_max_input_len': suggested_max_input_len,
    'suggested_max_target_len': suggested_max_target_len,
    'special_tokens': {
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'sep_token': '<sep>',
        'pad_id': vocab['<pad>'],
        'unk_id': vocab['<unk>'],
        'bos_id': vocab['<bos>'],
        'eos_id': vocab['<eos>'],
        'sep_id': vocab['<sep>']
    },
    'input_format': 'emotion: {emotion} <sep> situation: {situation} <sep> customer: {customer} <sep>',
    'target_format': '<bos> {agent_reply} <eos>'
}

with open(f'{save_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Configuration saved!")

print(f"\nAll files saved to: {save_dir}")
print("\nSaved files:")
print("  📄 vocab.json - Vocabulary (token -> id)")
print("  📄 id_to_token.json - Reverse vocabulary (id -> token)")
print("  📄 train_processed.pkl - Processed training data")
print("  📄 val_processed.pkl - Processed validation data")
print("  📄 test_processed.pkl - Processed test data")
print("  📄 train_df.csv - Original training dataframe")
print("  📄 val_df.csv - Original validation dataframe")
print("  📄 test_df.csv - Original test dataframe")
print("  📄 config.json - All configuration parameters")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------------------
print("=" * 80)
print("✅ TASK 1 COMPLETE - PREPROCESSING SUMMARY")
print("=" * 80)

print("\n📊 DATASET STATISTICS:")
print(f"   • Total samples: {len(df_clean)}")
print(f"   • Training: {len(train_processed)} samples (80%)")
print(f"   • Validation: {len(val_processed)} samples (10%)")
print(f"   • Test: {len(test_processed)} samples (10%)")

print("\n📚 VOCABULARY:")
print(f"   • Vocabulary size: {len(vocab)}")
print(f"   • Special tokens: <pad>, <unk>, <bos>, <eos>, <sep>")
print(f"   • Minimum word frequency: {MIN_FREQUENCY}")

print("\n📏 SEQUENCE LENGTHS:")
print(f"   • Suggested max input length: {suggested_max_input_len}")
print(f"   • Suggested max target length: {suggested_max_target_len}")

print("\n🔤 TEXT NORMALIZATION:")
print("   ✓ Converted to lowercase")
print("   ✓ Normalized whitespace")
print("   ✓ Normalized punctuation")
print("   ✓ Expanded contractions")

print("\n📝 DATA FORMAT:")
print("   • Input (X): emotion: {emotion} <sep> situation: {situation} <sep> customer: {customer} <sep>")
print("   • Target (Y): <bos> {agent_reply} <eos>")

print("\n" + "=" * 80)
print("🎉 ALL PREPROCESSING COMPLETE!")
print("=" * 80)
print("✅ Data is ready for Task 2: Model Architecture (Transformer)")
print("=" * 80)

# Quick verification - test loading saved files
print("\n🔍 Quick verification - testing saved files...")
with open(f'{save_dir}/vocab.json', 'r') as f:
    loaded_vocab = json.load(f)
print(f"✓ Successfully loaded vocabulary: {len(loaded_vocab)} tokens")

with open(f'{save_dir}/train_processed.pkl', 'rb') as f:
    loaded_train = pickle.load(f)
print(f"✓ Successfully loaded training data: {len(loaded_train)} samples")

print("\n" + "=" * 80)
print("All files saved and verified successfully!")
print("You are now ready to proceed to Task 2!")
print("=" * 80)
