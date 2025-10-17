# ============================================================================
# CELL 2: TASK 2 - MODEL ARCHITECTURE (Transformer from Scratch)
# ============================================================================
# Empathetic Conversational Chatbot - Transformer with MultiHead Attention
# 
# This cell implements a complete Transformer encoder-decoder model from scratch:
# - Positional Encoding (for position information in sequences)
# - Multi-Head Attention (core attention mechanism)
# - Feed-Forward Networks (position-wise transformation)
# - Encoder Layer (self-attention + feed-forward)
# - Decoder Layer (self-attention + cross-attention + feed-forward)
# - Complete Transformer Model (encoder + decoder)
# 
# All components built from scratch using PyTorch (no pretrained weights)
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: Install and Import Required Libraries
# ----------------------------------------------------------------------------
print("=" * 80)
print("INSTALLING PYTORCH AND REQUIRED LIBRARIES...")
print("=" * 80)

!pip install torch torchvision torchaudio -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import pickle

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ Device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 2: Load Preprocessed Data and Configuration
# ----------------------------------------------------------------------------
print("=" * 80)
print("LOADING PREPROCESSED DATA")
print("=" * 80)

# Load configuration from Task 1
with open('/content/preprocessed_data/config.json', 'r') as f:
    config = json.load(f)

# Load vocabulary
with open('/content/preprocessed_data/vocab.json', 'r') as f:
    vocab = json.load(f)

with open('/content/preprocessed_data/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)
    # Convert string keys back to integers
    id_to_token = {int(k): v for k, v in id_to_token.items()}

print(f"✓ Vocabulary size: {config['vocab_size']}")
print(f"✓ Training samples: {config['train_size']}")
print(f"✓ Validation samples: {config['val_size']}")
print(f"✓ Test samples: {config['test_size']}")
print(f"✓ Max input length: {config['suggested_max_input_len']}")
print(f"✓ Max target length: {config['suggested_max_target_len']}")
print("\nSpecial token IDs:")
for token_name, token_id in config['special_tokens'].items():
    if token_name.endswith('_id'):
        print(f"  {token_name}: {token_id}")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 3: Define Model Hyperparameters
# ----------------------------------------------------------------------------
print("=" * 80)
print("MODEL HYPERPARAMETERS")
print("=" * 80)

# Model architecture hyperparameters (as per task requirements)
MODEL_CONFIG = {
    # Vocabulary and special tokens
    'vocab_size': config['vocab_size'],
    'pad_token_id': config['special_tokens']['pad_id'],
    'bos_token_id': config['special_tokens']['bos_id'],
    'eos_token_id': config['special_tokens']['eos_id'],
    
    # Sequence lengths
    'max_input_len': config['suggested_max_input_len'],    # 79
    'max_target_len': config['suggested_max_target_len'],  # 34
    
    # Model dimensions
    'd_model': 512,              # Embedding dimension (256 or 512 as suggested)
    'n_heads': 8,                # Number of attention heads (2 as minimum, using 8 for better performance)
    'd_ff': 2048,                # Feed-forward dimension (typically 4 * d_model)
    
    # Number of layers
    'n_encoder_layers': 2,       # Number of encoder layers
    'n_decoder_layers': 2,       # Number of decoder layers
    
    # Regularization
    'dropout': 0.1,              # Dropout rate (0.1-0.3 as suggested)
    
    # Training
    'device': device
}

print("Model Configuration:")
for key, value in MODEL_CONFIG.items():
    print(f"  {key}: {value}")
print("=" * 80 + "\n")


# ============================================================================
# COMPONENT 1: POSITIONAL ENCODING
# ============================================================================
# Positional Encoding adds position information to embeddings
# since Transformers don't have inherent position awareness like RNNs
# Uses sine and cosine functions of different frequencies
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding to inject position information into embeddings.
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        - pos = position in sequence
        - i = dimension index
        - d_model = embedding dimension
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the formula
        # This creates [10000^(0/d_model), 10000^(2/d_model), ..., 10000^(d_model/d_model)]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Add positional encoding to embeddings
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# COMPONENT 2: MULTI-HEAD ATTENTION
# ============================================================================
# Multi-Head Attention allows the model to attend to different parts
# of the sequence from different representation subspaces
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Steps:
    1. Project Q, K, V into multiple heads
    2. Compute scaled dot-product attention for each head
    3. Concatenate all heads
    4. Project back to d_model dimension
    
    Attention formula:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V (one for each)
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (n_heads, d_k).
        Reshape from (batch_size, seq_len, d_model) 
               to (batch_size, n_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch_size, n_heads, seq_len_q, d_k)
            K: Key tensor (batch_size, n_heads, seq_len_k, d_k)
            V: Value tensor (batch_size, n_heads, seq_len_v, d_k)
            mask: Optional mask tensor
        
        Returns:
            Attention output and attention weights
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        # Mask is used to prevent attention to certain positions
        # (e.g., padding tokens, future tokens in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # Shape: (batch_size, n_heads, seq_len_q, d_k)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor
        
        Returns:
            Output tensor (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, n_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, n_heads, seq_len_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, n_heads, seq_len_v, d_k)
        
        # 3. Apply scaled dot-product attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        # (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Reshape to (batch_size, seq_len_q, d_model)
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        
        return output, attention_weights


# ============================================================================
# COMPONENT 3: FEED-FORWARD NETWORK
# ============================================================================
# Position-wise Feed-Forward Network
# Applied to each position separately and identically
# Consists of two linear transformations with ReLU activation
# ============================================================================

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Architecture:
        FFN(x) = max(0, x * W1 + b1) * W2 + b2
    
    Or simply:
        Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super(PositionWiseFeedForward, self).__init__()
        
        # First linear layer: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Apply first linear transformation and ReLU
        x = F.relu(self.linear1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply second linear transformation
        x = self.linear2(x)
        
        return x


# ============================================================================
# COMPONENT 4: ENCODER LAYER
# ============================================================================
# Single Encoder Layer consisting of:
# 1. Multi-Head Self-Attention
# 2. Add & Norm (residual connection + layer normalization)
# 3. Feed-Forward Network
# 4. Add & Norm (residual connection + layer normalization)
# ============================================================================

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Architecture:
        x -> Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm -> output
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (applied before attention and FFN - Pre-LN variant)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        # Pre-LN: Norm -> Attention -> Residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x


# ============================================================================
# COMPONENT 5: DECODER LAYER
# ============================================================================
# Single Decoder Layer consisting of:
# 1. Masked Multi-Head Self-Attention (looks only at previous positions)
# 2. Add & Norm
# 3. Multi-Head Cross-Attention (attends to encoder output)
# 4. Add & Norm
# 5. Feed-Forward Network
# 6. Add & Norm
# ============================================================================

class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Architecture:
        x -> Masked Self-Attention -> Add & Norm 
          -> Cross-Attention (with encoder) -> Add & Norm 
          -> Feed-Forward -> Add & Norm -> output
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention (attends to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source (encoder) mask
            tgt_mask: Target (decoder) mask (prevents looking at future tokens)
        
        Returns:
            Output tensor (batch_size, tgt_seq_len, d_model)
        """
        # 1. Masked self-attention (decoder attends to its own previous outputs)
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # 2. Cross-attention (decoder attends to encoder output)
        # Query from decoder, Key and Value from encoder
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        # 3. Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm3(x)  # Layer normalization
        
        return x


# ============================================================================
# COMPONENT 6: COMPLETE TRANSFORMER MODEL
# ============================================================================
# Full Transformer encoder-decoder architecture
# Combines all components into a complete model
# ============================================================================

class Transformer(nn.Module):
    """
    Complete Transformer Model for sequence-to-sequence tasks.
    
    Architecture:
        Input -> Embedding + Positional Encoding -> Encoder 
              -> Decoder <- Target Embedding + Positional Encoding
              -> Linear -> Output
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary with model configuration
        """
        super(Transformer, self).__init__()
        
        # Store configuration
        self.config = config
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.pad_token_id = config['pad_token_id']
        
        # Embedding layers (shared between encoder and decoder for efficiency)
        # We use the same embedding for both source and target
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, 
                                     padding_idx=self.pad_token_id)
        
        # Positional encoding
        max_len = max(config['max_input_len'], config['max_target_len']) + 10
        self.positional_encoding = PositionalEncoding(
            self.d_model, 
            max_len=max_len, 
            dropout=config['dropout']
        )
        
        # Encoder layers (stack of N encoder layers)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                config['d_model'],
                config['n_heads'],
                config['d_ff'],
                config['dropout']
            ) for _ in range(config['n_encoder_layers'])
        ])
        
        # Decoder layers (stack of N decoder layers)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                config['d_model'],
                config['n_heads'],
                config['d_ff'],
                config['dropout']
            ) for _ in range(config['n_decoder_layers'])
        ])
        
        # Final linear layer to project to vocabulary size
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights.
        Uses Xavier uniform initialization for better gradient flow.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """
        Create mask for padding tokens.
        
        Args:
            seq: Input sequence (batch_size, seq_len)
        
        Returns:
            Mask tensor where 1 = valid token, 0 = padding token
        """
        # Create mask: 1 for non-padding tokens, 0 for padding tokens
        mask = (seq != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, size):
        """
        Create mask to prevent decoder from looking at future tokens.
        This is crucial for autoregressive generation.
        
        Args:
            size: Sequence length
        
        Returns:
            Lower triangular mask
        """
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            src_mask: Source mask
        
        Returns:
            Encoder output (batch_size, src_seq_len, d_model)
        """
        # Embed source tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence (batch_size, tgt_seq_len)
            encoder_output: Output from encoder
            src_mask: Source mask
            tgt_mask: Target mask
        
        Returns:
            Decoder output (batch_size, tgt_seq_len, d_model)
        """
        # Embed target tokens and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """
        Forward pass of the Transformer.
        
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
        
        Returns:
            Output logits (batch_size, tgt_seq_len, vocab_size)
        """
        # Create masks
        src_mask = self.create_padding_mask(src)
        
        # For target, we need both padding mask and look-ahead mask
        # We combine them using element-wise multiplication (both are 0/1 masks)
        tgt_padding_mask = self.create_padding_mask(tgt)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_padding_mask * tgt_look_ahead_mask  # Element-wise multiplication for combining masks
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        
        return output


# ----------------------------------------------------------------------------
# STEP 4: Initialize the Model
# ----------------------------------------------------------------------------
print("=" * 80)
print("INITIALIZING TRANSFORMER MODEL")
print("=" * 80)

# Create the model
model = Transformer(MODEL_CONFIG).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✓ Model created successfully!")
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

print(f"\nModel Architecture:")
print(f"  Encoder layers: {MODEL_CONFIG['n_encoder_layers']}")
print(f"  Decoder layers: {MODEL_CONFIG['n_decoder_layers']}")
print(f"  Attention heads: {MODEL_CONFIG['n_heads']}")
print(f"  Embedding dimension: {MODEL_CONFIG['d_model']}")
print(f"  Feed-forward dimension: {MODEL_CONFIG['d_ff']}")
print(f"  Dropout: {MODEL_CONFIG['dropout']}")

print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 5: Test the Model with Sample Data
# ----------------------------------------------------------------------------
print("=" * 80)
print("TESTING MODEL WITH SAMPLE DATA")
print("=" * 80)

# Create sample batch
batch_size = 4
src_seq_len = 20
tgt_seq_len = 10

# Random input (simulating tokenized sequences)
sample_src = torch.randint(0, MODEL_CONFIG['vocab_size'], (batch_size, src_seq_len)).to(device)
sample_tgt = torch.randint(0, MODEL_CONFIG['vocab_size'], (batch_size, tgt_seq_len)).to(device)

print(f"Sample input shapes:")
print(f"  Source: {sample_src.shape}")
print(f"  Target: {sample_tgt.shape}")

# Forward pass
with torch.no_grad():
    output = model(sample_src, sample_tgt)

print(f"\nModel output shape: {output.shape}")
print(f"Expected shape: (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, vocab_size={MODEL_CONFIG['vocab_size']})")

# Verify output shape
assert output.shape == (batch_size, tgt_seq_len, MODEL_CONFIG['vocab_size']), "Output shape mismatch!"

print("\n✓ Model test passed successfully!")
print("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# STEP 6: Display Model Summary
# ----------------------------------------------------------------------------
print("=" * 80)
print("DETAILED MODEL SUMMARY")
print("=" * 80)

print("\nModel Components:")
print("  1. Embedding Layer")
print(f"     - Input: Token IDs (vocab_size={MODEL_CONFIG['vocab_size']})")
print(f"     - Output: Embeddings (d_model={MODEL_CONFIG['d_model']})")

print("\n  2. Positional Encoding")
print(f"     - Adds position information to embeddings")
print(f"     - Max length: {max(MODEL_CONFIG['max_input_len'], MODEL_CONFIG['max_target_len']) + 10}")

print(f"\n  3. Encoder (x{MODEL_CONFIG['n_encoder_layers']} layers)")
print(f"     Each layer contains:")
print(f"     - Multi-Head Self-Attention ({MODEL_CONFIG['n_heads']} heads)")
print(f"     - Feed-Forward Network (d_ff={MODEL_CONFIG['d_ff']})")
print(f"     - Layer Normalization (x2)")
print(f"     - Residual Connections (x2)")

print(f"\n  4. Decoder (x{MODEL_CONFIG['n_decoder_layers']} layers)")
print(f"     Each layer contains:")
print(f"     - Masked Multi-Head Self-Attention ({MODEL_CONFIG['n_heads']} heads)")
print(f"     - Multi-Head Cross-Attention ({MODEL_CONFIG['n_heads']} heads)")
print(f"     - Feed-Forward Network (d_ff={MODEL_CONFIG['d_ff']})")
print(f"     - Layer Normalization (x3)")
print(f"     - Residual Connections (x3)")

print(f"\n  5. Output Projection")
print(f"     - Projects decoder output to vocabulary size")
print(f"     - Output: Logits over vocabulary")

print("\n" + "=" * 80)
print("✅ TASK 2 COMPLETE - MODEL ARCHITECTURE")
print("=" * 80)
print("✓ Transformer encoder-decoder built from scratch")
print("✓ Multi-head attention implemented")
print("✓ Positional encoding implemented")
print("✓ Feed-forward networks implemented")
print("✓ Layer normalization and residual connections implemented")
print("✓ All model weights randomly initialized (no pretrained weights)")
print("\nModel is ready for Task 3: Training")
print("=" * 80)

