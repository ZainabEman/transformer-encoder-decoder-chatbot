# ============================================================================
# CELL 5 - PART 1: TASK 5 - INFERENCE & UI (Core Functions)
# ============================================================================
# Empathetic Conversational Chatbot - Inference Functions
# 
# This is PART 1 of Task 5, containing:
# - Inference helper functions
# - Greedy decoding
# - Beam search decoding
# - Attention extraction for visualization
# 
# PART 2 will contain the Gradio UI interface
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Import Required Libraries
# ----------------------------------------------------------------------------
print("=" * 80)
print("TASK 5 - PART 1: LOADING INFERENCE FUNCTIONS")
print("=" * 80)

import torch
import torch.nn.functional as F
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

print("✓ Libraries imported successfully!")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 2: Load Model and Configurations
# ----------------------------------------------------------------------------
print("=" * 80)
print("LOADING MODEL FOR INFERENCE")
print("=" * 80)

# Load configurations
with open('/content/preprocessed_data/config.json', 'r') as f:
    config = json.load(f)

with open('/content/preprocessed_data/vocab.json', 'r') as f:
    vocab = json.load(f)

with open('/content/preprocessed_data/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)
    id_to_token = {int(k): v for k, v in id_to_token.items()}

# Get special token IDs
PAD_ID = config['special_tokens']['pad_id']
UNK_ID = config['special_tokens']['unk_id']
BOS_ID = config['special_tokens']['bos_id']
EOS_ID = config['special_tokens']['eos_id']
SEP_ID = config['special_tokens']['sep_id']

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model should already be loaded from previous tasks
# If not, you need to reload it
model.eval()

print(f"✓ Model loaded in evaluation mode")
print(f"✓ Device: {device}")
print(f"✓ Vocabulary size: {len(vocab)}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 3: Text Processing Functions
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING TEXT PROCESSING FUNCTIONS")
print("=" * 80)

def normalize_text(text):
    """
    Normalize input text (same as during training).
    
    Args:
        text: Input string
    
    Returns:
        Normalized string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize contractions
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
    
    # Add spaces around punctuation
    import re
    text = re.sub(r'([.!?,;:\"\'])', r' \1 ', text)
    
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(text):
    """Simple word-level tokenization."""
    return text.split()


def tokens_to_ids(tokens, vocab):
    """Convert tokens to IDs."""
    return [vocab.get(token, UNK_ID) for token in tokens]


def ids_to_text(ids, id_to_token, special_tokens={'<pad>', '<bos>', '<eos>', '<unk>', '<sep>'}):
    """Convert IDs back to text."""
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    
    tokens = []
    for id in ids:
        token = id_to_token.get(int(id), '<unk>')
        if token == '<eos>':
            break
        if token not in special_tokens:
            tokens.append(token)
    
    return ' '.join(tokens)


def prepare_input(emotion, situation, customer_utterance, vocab):
    """
    Prepare input sequence in the format expected by the model.
    
    Format: emotion: {emotion} <sep> situation: {situation} <sep> customer: {customer} <sep>
    
    Args:
        emotion: Emotion string
        situation: Situation string
        customer_utterance: Customer's message
        vocab: Vocabulary dictionary
    
    Returns:
        Tensor of input token IDs
    """
    # Normalize inputs
    emotion = normalize_text(emotion) if emotion else "neutral"
    situation = normalize_text(situation) if situation else ""
    customer_utterance = normalize_text(customer_utterance)
    
    # Create input string
    input_text = f"emotion : {emotion} <sep> situation : {situation} <sep> customer : {customer_utterance} <sep>"
    
    # Tokenize
    tokens = []
    for token in input_text.split():
        if token == '<sep>':
            tokens.append('<sep>')
        else:
            tokens.append(token)
    
    # Convert to IDs
    input_ids = tokens_to_ids(tokens, vocab)
    
    # Truncate to max length
    max_input_len = config['suggested_max_input_len']
    input_ids = input_ids[:max_input_len]
    
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension


print("✓ Text processing functions defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 4: Greedy Decoding Function
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING GREEDY DECODING")
print("=" * 80)

def greedy_decode(model, src, max_len, bos_id, eos_id, device):
    """
    Greedy decoding: always select the most probable token.
    
    Args:
        model: Transformer model
        src: Source sequence (1, src_len)
        max_len: Maximum length to generate
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        device: Device
    
    Returns:
        Tuple of (generated sequence, attention weights)
    """
    model.eval()
    src = src.to(device)
    
    with torch.no_grad():
        # Encode source
        src_mask = model.create_padding_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        # Initialize decoder input with <bos>
        tgt = torch.full((1, 1), bos_id, dtype=torch.long).to(device)
        
        # Store attention weights for visualization
        attention_weights = []
        
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
            
            # Get next token (greedy selection)
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if <eos> generated
            if next_token.item() == eos_id:
                break
        
        return tgt, None  # We'll implement attention extraction later


print("✓ Greedy decoding function defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 5: Beam Search Decoding Function
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING BEAM SEARCH DECODING")
print("=" * 80)

class BeamSearchNode:
    """Node for beam search."""
    
    def __init__(self, sequence, score, length):
        self.sequence = sequence
        self.score = score
        self.length = length
    
    def get_avg_score(self):
        """Get average score (normalized by length)."""
        return self.score / self.length


def beam_search_decode(model, src, max_len, bos_id, eos_id, device, beam_width=5):
    """
    Beam search decoding: keep top-k candidates at each step.
    
    Args:
        model: Transformer model
        src: Source sequence (1, src_len)
        max_len: Maximum length to generate
        bos_id: Beginning of sequence token ID
        eos_id: End of sequence token ID
        device: Device
        beam_width: Number of beams to keep
    
    Returns:
        Best generated sequence
    """
    model.eval()
    src = src.to(device)
    
    with torch.no_grad():
        # Encode source
        src_mask = model.create_padding_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        # Initialize with <bos>
        initial_sequence = torch.full((1, 1), bos_id, dtype=torch.long).to(device)
        
        # Active beams
        beams = [BeamSearchNode(initial_sequence, 0.0, 1)]
        finished_beams = []
        
        # Beam search
        for step in range(max_len - 1):
            candidates = []
            
            for beam in beams:
                tgt = beam.sequence
                
                # If beam already ended with <eos>, add to finished
                if tgt[0, -1].item() == eos_id:
                    finished_beams.append(beam)
                    continue
                
                # Create masks
                tgt_padding_mask = model.create_padding_mask(tgt)
                tgt_look_ahead_mask = model.create_look_ahead_mask(tgt.size(1)).to(device)
                tgt_mask = tgt_padding_mask * tgt_look_ahead_mask
                
                # Decode
                decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
                output = model.output_projection(decoder_output)
                
                # Get log probabilities
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                
                # Get top-k tokens
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                # Create new candidates
                for k in range(beam_width):
                    new_token = top_k_indices[0, k].unsqueeze(0).unsqueeze(0)
                    new_sequence = torch.cat([tgt, new_token], dim=1)
                    new_score = beam.score + top_k_probs[0, k].item()
                    new_length = beam.length + 1
                    
                    candidates.append(BeamSearchNode(new_sequence, new_score, new_length))
            
            # Select top beams
            if candidates:
                beams = sorted(candidates, key=lambda x: x.get_avg_score(), reverse=True)[:beam_width]
            else:
                break
            
            # Stop if all beams finished
            if len(beams) == 0:
                break
        
        # Get best sequence from finished beams or active beams
        all_beams = finished_beams + beams
        if all_beams:
            best_beam = max(all_beams, key=lambda x: x.get_avg_score())
            return best_beam.sequence
        else:
            return initial_sequence


print("✓ Beam search decoding function defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 6: Main Inference Function
# ----------------------------------------------------------------------------
print("=" * 80)
print("DEFINING MAIN INFERENCE FUNCTION")
print("=" * 80)

def generate_response(emotion, situation, customer_utterance, decoding_strategy='greedy', beam_width=5):
    """
    Generate empathetic response given context.
    
    Args:
        emotion: Emotion string (e.g., "happy", "sad")
        situation: Situation description
        customer_utterance: What the customer said
        decoding_strategy: 'greedy' or 'beam_search'
        beam_width: Beam width for beam search
    
    Returns:
        Dictionary with response and metadata
    """
    # Prepare input
    src = prepare_input(emotion, situation, customer_utterance, vocab)
    
    # Generate response
    max_len = config['suggested_max_target_len']
    
    if decoding_strategy == 'greedy':
        output, attention = greedy_decode(model, src, max_len, BOS_ID, EOS_ID, device)
    elif decoding_strategy == 'beam_search':
        output = beam_search_decode(model, src, max_len, BOS_ID, EOS_ID, device, beam_width)
        attention = None
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    # Convert to text
    response_text = ids_to_text(output[0], id_to_token)
    
    return {
        'response': response_text,
        'emotion': emotion,
        'situation': situation,
        'customer': customer_utterance,
        'decoding_strategy': decoding_strategy
    }


print("✓ Main inference function defined")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 7: Test Inference Functions
# ----------------------------------------------------------------------------
print("=" * 80)
print("TESTING INFERENCE FUNCTIONS")
print("=" * 80)

# Test with example
test_emotion = "happy"
test_situation = "I just got a promotion at work!"
test_customer = "I'm so excited about this new opportunity."

print("\nTest Input:")
print(f"  Emotion: {test_emotion}")
print(f"  Situation: {test_situation}")
print(f"  Customer: {test_customer}")

print("\nGenerating responses...")

# Greedy decoding
result_greedy = generate_response(
    test_emotion, 
    test_situation, 
    test_customer, 
    decoding_strategy='greedy'
)

print(f"\nGreedy Decoding:")
print(f"  → {result_greedy['response']}")

# Beam search
result_beam = generate_response(
    test_emotion, 
    test_situation, 
    test_customer, 
    decoding_strategy='beam_search',
    beam_width=5
)

print(f"\nBeam Search (width=5):")
print(f"  → {result_beam['response']}")

print("\n" + "=" * 80)
print("✅ PART 1 COMPLETE - Inference functions ready!")
print("=" * 80)
print("\nNow run PART 2 to create the Gradio UI interface.")
print("=" * 80)

