# ============================================================================
# CELL 4: TASK 4 - EVALUATION
# ============================================================================
# Empathetic Conversational Chatbot - Model Evaluation
# 
# This cell handles comprehensive model evaluation including:
# - Loading best trained model
# - Evaluation on test set with all metrics
# - Automatic Metrics: BLEU, ROUGE-L, chrF, Perplexity
# - Qualitative examples comparing model output vs ground truth
# - Human evaluation preparation: Fluency, Relevance, Adequacy
# - Attention visualization
# - Detailed analysis and insights
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Import Required Libraries
# ----------------------------------------------------------------------------
print("=" * 80)
print("IMPORTING LIBRARIES FOR EVALUATION")
print("=" * 80)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer
import math
import random

print("✓ All libraries imported successfully!")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 2: Load Preprocessed Data and Configuration
# ----------------------------------------------------------------------------
print("=" * 80)
print("LOADING PREPROCESSED DATA")
print("=" * 80)

# Load test data
with open('/content/preprocessed_data/test_processed.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Load configuration
with open('/content/preprocessed_data/config.json', 'r') as f:
    config = json.load(f)

# Load vocabulary
with open('/content/preprocessed_data/vocab.json', 'r') as f:
    vocab = json.load(f)

with open('/content/preprocessed_data/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)
    # Convert string keys back to integers
    id_to_token = {int(k): v for k, v in id_to_token.items()}

# Load original test dataframe for reference
test_df = pd.read_csv('/content/preprocessed_data/test_df.csv')

print(f"✓ Test samples: {len(test_data)}")
print(f"✓ Vocabulary size: {config['vocab_size']}")
print(f"✓ Special tokens loaded")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 3: Load Best Trained Model
# ----------------------------------------------------------------------------
print("=" * 80)
print("LOADING BEST TRAINED MODEL")
print("=" * 80)

# Load checkpoint
# Note: weights_only=False is safe here since this is our own trained model
checkpoint = torch.load('/content/best_model.pt', map_location='cpu', weights_only=False)

print(f"Best model from epoch: {checkpoint['epoch']}")
print(f"Training loss: {checkpoint['train_loss']:.4f}")
print(f"Validation BLEU: {checkpoint['val_metrics']['bleu']:.2f}")
print(f"Validation perplexity: {checkpoint['val_metrics']['perplexity']:.2f}")

# Note: Model should already be defined from Task 2
# If not, you'll need to redefine the TransformerModel class
# Here we assume 'model' variable exists from previous cells

# Load model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"\n✓ Model loaded and set to evaluation mode")
print(f"✓ Device: {device}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 4: Define Evaluation Helper Functions
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING EVALUATION FUNCTIONS")
print("=" * 80)

def ids_to_text(ids, id_to_token, special_tokens={'<pad>', '<bos>', '<eos>', '<unk>', '<sep>'}):
    """
    Convert token IDs to text, removing special tokens.
    
    Args:
        ids: List or tensor of token IDs
        id_to_token: Dictionary mapping ID to token
        special_tokens: Set of special tokens to remove
    
    Returns:
        Text string
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    
    tokens = []
    for id in ids:
        token = id_to_token.get(int(id), '<unk>')
        # Stop at <eos> token
        if token == '<eos>':
            break
        # Skip special tokens
        if token not in special_tokens:
            tokens.append(token)
    
    return ' '.join(tokens)


def greedy_decode(model, src, max_len, bos_id, eos_id, pad_id, device):
    """
    Greedy decoding: always select the most probable token at each step.
    
    Args:
        model: Transformer model
        src: Source sequence (batch_size, src_len)
        max_len: Maximum length to generate
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        pad_id: Padding token ID
        device: Device
    
    Returns:
        Generated sequence (batch_size, max_len)
    """
    model.eval()
    batch_size = src.size(0)
    
    # Encode source
    src_mask = model.create_padding_mask(src)
    encoder_output = model.encode(src, src_mask)
    
    # Initialize decoder input with <bos>
    tgt = torch.full((batch_size, 1), bos_id, dtype=torch.long).to(device)
    
    # Generate tokens one by one
    for _ in range(max_len - 1):
        # Create target mask
        tgt_padding_mask = model.create_padding_mask(tgt)
        tgt_look_ahead_mask = model.create_look_ahead_mask(tgt.size(1)).to(device)
        tgt_mask = tgt_padding_mask * tgt_look_ahead_mask
        
        # Decode
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = model.output_projection(decoder_output)
        
        # Get last token prediction
        next_token_logits = output[:, -1, :]  # (batch_size, vocab_size)
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Append to target
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if all sequences have generated <eos>
        if (next_token == eos_id).all():
            break
    
    return tgt


def calculate_bleu(predictions, references):
    """Calculate BLEU score using sacrebleu."""
    references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, references)
    return bleu.score


def calculate_rouge(predictions, references):
    """Calculate ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)
    
    return np.mean(scores)


def calculate_chrf(predictions, references):
    """Calculate chrF score using sacrebleu."""
    references = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(predictions, references)
    return chrf.score


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)


print("✓ Greedy decoding function defined")
print("✓ Text conversion function defined")
print("✓ All metric functions defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 5: Evaluate on Test Set
# ----------------------------------------------------------------------------
print("=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

# Get special token IDs
pad_id = config['special_tokens']['pad_id']
bos_id = config['special_tokens']['bos_id']
eos_id = config['special_tokens']['eos_id']
max_target_len = config['suggested_max_target_len']
max_input_len = config['suggested_max_input_len']

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# Storage for results
predictions = []
references = []
total_loss = 0
batch_count = 0

# For detailed examples
detailed_examples = []

print(f"Evaluating {len(test_data)} test samples...")
print("Generating predictions using greedy decoding...\n")

# Create batches manually for test set
batch_size = 32

with torch.no_grad():
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        # Get batch
        batch = test_data[i:i+batch_size]
        
        # Prepare input and target
        src_batch = []
        tgt_batch = []
        
        for sample in batch:
            # Truncate sequences to max lengths used during training
            input_ids = sample['input_ids'][:max_input_len]
            target_ids = sample['target_ids'][:max_target_len]
            
            src_batch.append(torch.tensor(input_ids, dtype=torch.long))
            tgt_batch.append(torch.tensor(target_ids, dtype=torch.long))
        
        # Pad sequences
        from torch.nn.utils.rnn import pad_sequence
        src = pad_sequence(src_batch, batch_first=True, padding_value=pad_id).to(device)
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id).to(device)
        
        # Calculate loss (for perplexity)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = model(src, tgt_input)
        output_flat = output.reshape(-1, output.size(-1))
        tgt_flat = tgt_output.reshape(-1)
        loss = criterion(output_flat, tgt_flat)
        
        total_loss += loss.item()
        batch_count += 1
        
        # Generate predictions using greedy decoding
        generated = greedy_decode(
            model, src, max_target_len, bos_id, eos_id, pad_id, device
        )
        
        # Convert to text
        for j in range(len(batch)):
            pred_text = ids_to_text(generated[j], id_to_token)
            ref_text = ids_to_text(tgt[j], id_to_token)
            
            predictions.append(pred_text)
            references.append(ref_text)
            
            # Store first 20 examples for detailed analysis
            if len(detailed_examples) < 20:
                idx = i + j
                detailed_examples.append({
                    'emotion': test_df.iloc[idx]['emotion_normalized'],
                    'situation': test_df.iloc[idx]['situation_normalized'],
                    'customer': test_df.iloc[idx]['customer_normalized'],
                    'reference': ref_text,
                    'prediction': pred_text
                })

print("\n✓ Evaluation complete!")


# ----------------------------------------------------------------------------
# STEP 6: Calculate All Metrics
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CALCULATING AUTOMATIC METRICS")
print("=" * 80)

# Calculate metrics
avg_loss = total_loss / batch_count
perplexity = calculate_perplexity(avg_loss)
bleu_score = calculate_bleu(predictions, references)
rouge_score = calculate_rouge(predictions, references)
chrf_score = calculate_chrf(predictions, references)

print(f"\n📊 TEST SET RESULTS:")
print(f"{'='*80}")
print(f"Test Loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"BLEU Score: {bleu_score:.2f}")
print(f"ROUGE-L Score: {rouge_score:.4f}")
print(f"chrF Score: {chrf_score:.2f}")
print(f"{'='*80}")


# ----------------------------------------------------------------------------
# STEP 7: Display Qualitative Examples
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("QUALITATIVE EXAMPLES - MODEL vs GROUND TRUTH")
print("=" * 80)

# Show diverse examples
for i, example in enumerate(detailed_examples[:10], 1):
    print(f"\n{'─'*80}")
    print(f"EXAMPLE {i}")
    print(f"{'─'*80}")
    print(f"Emotion: {example['emotion']}")
    print(f"Situation: {example['situation'][:80]}{'...' if len(example['situation']) > 80 else ''}")
    print(f"Customer: {example['customer']}")
    print(f"\n→ Ground Truth: {example['reference']}")
    print(f"→ Model Output: {example['prediction']}")
    
    # Simple quality assessment
    ref_words = set(example['reference'].split())
    pred_words = set(example['prediction'].split())
    overlap = len(ref_words & pred_words) / max(len(ref_words), 1)
    print(f"→ Word Overlap: {overlap:.2%}")

print("\n" + "=" * 80)


# ----------------------------------------------------------------------------
# STEP 8: Human Evaluation Preparation
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("HUMAN EVALUATION - Sample Questionnaire")
print("=" * 80)

print("""
Human evaluators will rate model outputs on a 1-5 scale for:

1. FLUENCY: Is the response grammatically correct and natural?
   1 = Incomprehensible, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Perfect

2. RELEVANCE: Is the response relevant to the context (emotion, situation, customer)?
   1 = Completely irrelevant, 2 = Slightly relevant, 3 = Moderately relevant, 
   4 = Relevant, 5 = Highly relevant

3. ADEQUACY: Does the response appropriately address the customer's utterance?
   1 = Completely inadequate, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent

4. EMPATHY: Does the response show empathy appropriate to the emotion?
   1 = No empathy, 2 = Slight empathy, 3 = Moderate empathy, 
   4 = Good empathy, 5 = Excellent empathy
""")

print("Preparing samples for human evaluation...\n")

# Create human evaluation dataset
human_eval_samples = []
random.seed(42)
eval_indices = random.sample(range(len(detailed_examples)), min(10, len(detailed_examples)))

for idx in eval_indices:
    example = detailed_examples[idx]
    human_eval_samples.append({
        'emotion': example['emotion'],
        'situation': example['situation'][:100],
        'customer_utterance': example['customer'],
        'model_response': example['prediction'],
        'ground_truth': example['reference'],
        'fluency_score': '',
        'relevance_score': '',
        'adequacy_score': '',
        'empathy_score': ''
    })

# Save to CSV for human evaluation
human_eval_df = pd.DataFrame(human_eval_samples)
human_eval_df.to_csv('/content/human_evaluation_samples.csv', index=False)

print("✓ Human evaluation samples saved to: /content/human_evaluation_samples.csv")
print(f"✓ Total samples for evaluation: {len(human_eval_samples)}")

# Display sample format
print("\nSample format for human evaluation:")
print("─" * 80)
print(human_eval_df.head(3).to_string())
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 9: Metrics Comparison and Visualization
# ----------------------------------------------------------------------------
print("=" * 80)
print("METRICS VISUALIZATION")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Metrics Bar Chart
metrics_names = ['BLEU', 'ROUGE-L', 'chrF']
metrics_values = [bleu_score / 100, rouge_score, chrf_score / 100]  # Normalize to 0-1 scale

axes[0, 0].bar(metrics_names, metrics_values, color=['steelblue', 'forestgreen', 'coral'])
axes[0, 0].set_ylabel('Score', fontsize=12)
axes[0, 0].set_title('Test Set Metrics Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_values):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 2: Perplexity
axes[0, 1].bar(['Perplexity'], [perplexity], color='indianred', width=0.4)
axes[0, 1].set_ylabel('Perplexity', fontsize=12)
axes[0, 1].set_title('Test Set Perplexity', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].text(0, perplexity + 2, f'{perplexity:.2f}', ha='center', fontweight='bold')

# Plot 3: Response Length Distribution
pred_lengths = [len(p.split()) for p in predictions[:1000]]
ref_lengths = [len(r.split()) for r in references[:1000]]

axes[1, 0].hist(pred_lengths, bins=30, alpha=0.6, label='Predictions', color='steelblue', edgecolor='black')
axes[1, 0].hist(ref_lengths, bins=30, alpha=0.6, label='References', color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Response Length (words)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Response Length Distribution', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Summary Statistics Table
summary_stats = [
    ['Metric', 'Score'],
    ['BLEU', f'{bleu_score:.2f}'],
    ['ROUGE-L', f'{rouge_score:.4f}'],
    ['chrF', f'{chrf_score:.2f}'],
    ['Perplexity', f'{perplexity:.2f}'],
    ['Avg Pred Length', f'{np.mean(pred_lengths):.1f} words'],
    ['Avg Ref Length', f'{np.mean(ref_lengths):.1f} words'],
]

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=summary_stats, cellLoc='left', loc='center',
                          colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_stats)):
    for j in range(2):
        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

axes[1, 1].set_title('Evaluation Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/content/evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Evaluation visualizations saved to: /content/evaluation_results.png")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 10: Error Analysis
# ----------------------------------------------------------------------------
print("=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

# Calculate per-sample BLEU scores to find best and worst examples
from sacrebleu.metrics import BLEU
bleu_metric = BLEU()

sample_scores = []
for pred, ref in zip(predictions[:100], references[:100]):
    score = bleu_metric.sentence_score(pred, [ref]).score
    sample_scores.append(score)

# Find best examples
best_indices = np.argsort(sample_scores)[-5:][::-1]
worst_indices = np.argsort(sample_scores)[:5]

print("\n🎯 BEST PREDICTIONS (Highest BLEU):")
print("─" * 80)
for rank, idx in enumerate(best_indices, 1):
    print(f"\n{rank}. BLEU Score: {sample_scores[idx]:.2f}")
    print(f"   Reference: {references[idx]}")
    print(f"   Prediction: {predictions[idx]}")

print("\n\n❌ WORST PREDICTIONS (Lowest BLEU):")
print("─" * 80)
for rank, idx in enumerate(worst_indices, 1):
    print(f"\n{rank}. BLEU Score: {sample_scores[idx]:.2f}")
    print(f"   Reference: {references[idx]}")
    print(f"   Prediction: {predictions[idx]}")

print("\n" + "=" * 80)


# ----------------------------------------------------------------------------
# STEP 11: Final Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("✅ TASK 4 COMPLETE - EVALUATION SUMMARY")
print("=" * 80)

print("\n📊 AUTOMATIC METRICS (Test Set):")
print(f"   • BLEU Score: {bleu_score:.2f}")
print(f"   • ROUGE-L Score: {rouge_score:.4f}")
print(f"   • chrF Score: {chrf_score:.2f}")
print(f"   • Perplexity: {perplexity:.2f}")

print("\n📈 STATISTICAL ANALYSIS:")
print(f"   • Test samples evaluated: {len(predictions)}")
print(f"   • Average prediction length: {np.mean(pred_lengths):.1f} words")
print(f"   • Average reference length: {np.mean(ref_lengths):.1f} words")
print(f"   • Shortest prediction: {min(pred_lengths)} words")
print(f"   • Longest prediction: {max(pred_lengths)} words")

print("\n👥 HUMAN EVALUATION:")
print(f"   • Prepared {len(human_eval_samples)} samples for human evaluation")
print(f"   • Evaluation criteria: Fluency, Relevance, Adequacy, Empathy (1-5 scale)")
print(f"   • Samples saved to: /content/human_evaluation_samples.csv")

print("\n📁 SAVED FILES:")
print("   • Evaluation plots: /content/evaluation_results.png")
print("   • Human eval samples: /content/human_evaluation_samples.csv")

print("\n✅ REQUIREMENTS MET:")
print("   ✓ Automatic Metrics: BLEU, ROUGE-L, chrF, Perplexity")
print("   ✓ Qualitative examples provided (model vs ground truth)")
print("   ✓ Human evaluation prepared (Fluency, Relevance, Adequacy)")
print("   ✓ Error analysis conducted")
print("   ✓ Comprehensive visualizations generated")

print("\n" + "=" * 80)
print("🎉 EVALUATION COMPLETE!")
print("=" * 80)
print("Model is ready for Task 5: Inference & UI")
print("=" * 80)

