# Hugging Face Spaces Deployment Guide

## Complete Guide to Deploying Your Empathetic Chatbot on Hugging Face Spaces

This guide will walk you through deploying your trained chatbot model to Hugging Face Spaces for permanent, free hosting with a public URL.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Files Overview](#files-overview)
3. [Hugging Face Setup](#hugging-face-setup)
4. [File Preparation](#file-preparation)
5. [Creating the Deployment Files](#creating-the-deployment-files)
6. [Uploading to Hugging Face](#uploading-to-hugging-face)
7. [Testing Your Deployment](#testing-your-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### What You Need

1. **Hugging Face Account** (free)
   - Sign up at: https://huggingface.co/join

2. **Your Model Files** (already prepared)
   - Located in: `empathetic_chatbot_outputs/`
   - Total size: ~390 MB

3. **Git** (for uploading)
   - Install from: https://git-scm.com/downloads

4. **Git LFS** (for large files like the model)
   - Install from: https://git-lfs.github.com/

---

## Files Overview

### Current Files in Your Package

```
empathetic_chatbot_outputs/
├── model/
│   └── best_model.pt                  (361 MB) - Your trained model
├── preprocessed/
│   ├── vocab.json                     (0.3 MB) - Vocabulary
│   ├── id_to_token.json               (0.3 MB) - Reverse vocabulary
│   ├── config.json                    (0.01 MB) - Configuration
│   ├── train_df.csv                   (23 MB) - Training data
│   ├── val_df.csv                     (2.9 MB) - Validation data
│   └── test_df.csv                    (2.9 MB) - Test data
├── training/
│   └── training_history.png           (0.34 MB) - Training curves
├── evaluation/
│   ├── evaluation_results.png         (0.13 MB) - Evaluation plots
│   └── human_evaluation_samples.csv   (0.01 MB) - Evaluation samples
└── README.md                          - Project documentation
```

### Files We'll Create

```
Additional files needed for deployment:
├── app.py                    - Gradio application
├── requirements.txt          - Python dependencies
├── .gitattributes           - Git LFS configuration
└── README.md                - Hugging Face Space README
```

---

## Hugging Face Setup

### Step 1: Create Account

1. Go to https://huggingface.co/join
2. Sign up with email or GitHub
3. Verify your email

### Step 2: Get Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `deployment-token`
4. Type: `Write`
5. Click "Generate token"
6. **Copy and save the token** (you'll need it later)

### Step 3: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in details:
   - **Name**: `empathetic-chatbot`
   - **License**: `MIT`
   - **SDK**: `Gradio`
   - **Hardware**: `CPU basic` (free)
   - **Visibility**: `Public`
4. Click "Create Space"

---

## File Preparation

### Step 1: Create Project Directory

On your local machine:

```bash
# Create a new directory for the Hugging Face Space
mkdir huggingface-deployment
cd huggingface-deployment
```

### Step 2: Copy Your Files

Copy files from `empathetic_chatbot_outputs/` to the new directory:

```bash
# Copy model files
mkdir -p model
cp path/to/empathetic_chatbot_outputs/model/best_model.pt model/

# Copy preprocessed data
mkdir -p preprocessed
cp path/to/empathetic_chatbot_outputs/preprocessed/vocab.json preprocessed/
cp path/to/empathetic_chatbot_outputs/preprocessed/id_to_token.json preprocessed/
cp path/to/empathetic_chatbot_outputs/preprocessed/config.json preprocessed/

# Copy visualizations (optional, for README)
mkdir -p images
cp path/to/empathetic_chatbot_outputs/training/training_history.png images/
cp path/to/empathetic_chatbot_outputs/evaluation/evaluation_results.png images/
```

**Note**: We don't need train_df.csv, val_df.csv, test_df.csv for deployment (they're only for training).

---

## Creating the Deployment Files

### File 1: app.py (Main Application)

Create `app.py` in your deployment directory:

```python
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import math
from datetime import datetime

# Load configuration and vocabulary
with open('preprocessed/config.json', 'r') as f:
    config = json.load(f)

with open('preprocessed/vocab.json', 'r') as f:
    vocab = json.load(f)

with open('preprocessed/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)
    id_to_token = {int(k): v for k, v in id_to_token.items()}

# Special token IDs
PAD_ID = config['special_tokens']['pad_id']
BOS_ID = config['special_tokens']['bos_id']
EOS_ID = config['special_tokens']['eos_id']
UNK_ID = config['special_tokens']['unk_id']
SEP_ID = config['special_tokens']['sep_id']

# Model Architecture (same as training)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_encoder_layers=2, 
                 n_decoder_layers=2, d_ff=2048, dropout=0.1, pad_token_id=0, 
                 max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def create_padding_mask(self, seq):
        mask = (seq != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask
    
    def encode(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        src_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_padding_mask * tgt_look_ahead_mask
        
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_projection(decoder_output)
        
        return output

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(
    vocab_size=config['vocab_size'],
    d_model=512,
    n_heads=8,
    n_encoder_layers=2,
    n_decoder_layers=2,
    d_ff=2048,
    dropout=0.1,
    pad_token_id=PAD_ID,
    max_len=200
)

# Load trained weights
checkpoint = torch.load('model/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Text processing functions
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am"
    }
    for c, e in contractions.items():
        text = text.replace(c, e)
    text = re.sub(r'([.!?,;:\"\'])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    return text.split()

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, UNK_ID) for token in tokens]

def ids_to_text(ids, id_to_token):
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    tokens = []
    for id in ids:
        token = id_to_token.get(int(id), '<unk>')
        if token == '<eos>':
            break
        if token not in {'<pad>', '<bos>', '<eos>', '<unk>', '<sep>'}:
            tokens.append(token)
    return ' '.join(tokens)

def prepare_input(emotion, situation, customer_utterance, vocab):
    emotion = normalize_text(emotion) if emotion else "neutral"
    situation = normalize_text(situation) if situation else ""
    customer_utterance = normalize_text(customer_utterance)
    
    input_text = f"emotion : {emotion} <sep> situation : {situation} <sep> customer : {customer_utterance} <sep>"
    
    tokens = []
    for token in input_text.split():
        if token == '<sep>':
            tokens.append('<sep>')
        else:
            tokens.append(token)
    
    input_ids = tokens_to_ids(tokens, vocab)
    max_input_len = config['suggested_max_input_len']
    input_ids = input_ids[:max_input_len]
    
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

def greedy_decode(model, src, max_len, bos_id, eos_id, device):
    model.eval()
    src = src.to(device)
    
    with torch.no_grad():
        src_mask = model.create_padding_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        tgt = torch.full((1, 1), bos_id, dtype=torch.long).to(device)
        
        for _ in range(max_len - 1):
            tgt_padding_mask = model.create_padding_mask(tgt)
            tgt_look_ahead_mask = model.create_look_ahead_mask(tgt.size(1)).to(device)
            tgt_mask = tgt_padding_mask * tgt_look_ahead_mask
            
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_projection(decoder_output)
            
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == eos_id:
                break
        
        return tgt

def generate_response(customer_message, emotion, situation):
    if not customer_message.strip():
        return "Please enter a message."
    
    src = prepare_input(emotion, situation, customer_message, vocab)
    max_len = config['suggested_max_target_len']
    
    output = greedy_decode(model, src, max_len, BOS_ID, EOS_ID, device)
    response_text = ids_to_text(output[0], id_to_token)
    
    return response_text

# Conversation history
conversation_history = []

def chatbot_interface(customer_message, emotion, situation):
    if not customer_message.strip():
        return "Please enter a message.", conversation_history
    
    response = generate_response(customer_message, emotion, situation)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    conversation_history.append({
        'time': timestamp,
        'emotion': emotion,
        'customer': customer_message,
        'agent': response
    })
    
    history_text = ""
    for i, turn in enumerate(conversation_history, 1):
        history_text += f"**Turn {i}** ({turn['time']}) - *{turn['emotion']}*\n"
        history_text += f"👤 Customer: {turn['customer']}\n"
        history_text += f"🤖 Agent: {turn['agent']}\n\n"
    
    return response, history_text

def clear_history():
    conversation_history.clear()
    return "", "Conversation history cleared."

# Emotion options
emotions = [
    "happy", "sad", "angry", "anxious", "excited", "grateful", "surprised",
    "disappointed", "proud", "afraid", "annoyed", "caring", "confident",
    "content", "disgusted", "embarrassed", "faithful", "furious", "guilty",
    "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
    "prepared", "sentimental", "terrified", "trusting", "devastated",
    "anticipating", "apprehensive", "ashamed"
]

# Create Gradio interface
with gr.Blocks(title="Empathetic Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Empathetic Conversational Chatbot
    
    This chatbot generates empathetic responses based on emotional context.
    Built with a Transformer architecture trained on 64,636 empathetic dialogues.
    
    **Model Performance**: BLEU 97.85 | ROUGE-L 0.15 | chrF 74.04
    """)
    
    with gr.Row():
        with gr.Column():
            customer_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=3
            )
            
            with gr.Row():
                emotion_input = gr.Dropdown(
                    choices=emotions,
                    label="Emotion",
                    value="happy"
                )
                
                situation_input = gr.Textbox(
                    label="Situation (Optional)",
                    placeholder="Describe the situation...",
                    lines=2
                )
            
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History")
            
            agent_output = gr.Textbox(
                label="Agent Response",
                lines=3,
                interactive=False
            )
        
        with gr.Column():
            history_output = gr.Markdown(label="Conversation History")
    
    gr.Examples(
        examples=[
            ["I just got a promotion at work!", "excited", "I've been working really hard"],
            ["I'm feeling really down today.", "sad", "Things haven't been going well"],
            ["I can't believe this happened!", "surprised", "This was totally unexpected"],
        ],
        inputs=[customer_input, emotion_input, situation_input]
    )
    
    send_btn.click(
        chatbot_interface,
        inputs=[customer_input, emotion_input, situation_input],
        outputs=[agent_output, history_output]
    )
    
    clear_btn.click(
        clear_history,
        inputs=[],
        outputs=[customer_input, history_output]
    )

if __name__ == "__main__":
    demo.launch()
```

### File 2: requirements.txt

Create `requirements.txt`:

```txt
torch>=2.0.0
gradio>=4.0.0
numpy>=1.24.0
```

### File 3: .gitattributes

Create `.gitattributes` for Git LFS:

```
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
```

### File 4: README.md (for Hugging Face)

Create `README.md`:

```markdown
---
title: Empathetic Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Empathetic Conversational Chatbot

An emotion-aware chatbot built with a Transformer architecture from scratch.

## Model Details

- **Architecture**: Transformer Encoder-Decoder
- **Parameters**: 31.6M
- **Training Data**: 64,636 empathetic dialogues
- **Performance**: BLEU 97.85

## How to Use

1. Enter your message
2. Select the emotion that describes your context
3. Optionally add a situation description
4. Click "Send" to get an empathetic response

## Features

- 33 different emotions supported
- Context-aware responses
- Conversation history tracking
- Real-time generation

## Model Performance

| Metric | Score |
|--------|-------|
| BLEU | 97.85 |
| ROUGE-L | 0.1522 |
| chrF | 74.04 |
| Perplexity | 48.92 |

## Technical Details

- Built from scratch in PyTorch
- No pre-trained weights
- Transformer with 8 attention heads
- 2 encoder and 2 decoder layers
```

---

## Uploading to Hugging Face

### Method 1: Using Git (Recommended)

**Step 1: Initialize Git**

```bash
cd huggingface-deployment
git init
git lfs install
```

**Step 2: Add Remote**

Replace `YOUR_USERNAME` with your Hugging Face username:

```bash
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/empathetic-chatbot
```

**Step 3: Track Large Files**

```bash
git lfs track "*.pt"
git lfs track "*.pth"
```

**Step 4: Add Files**

```bash
git add .gitattributes
git add app.py requirements.txt README.md
git add preprocessed/
git add model/best_model.pt
git add images/ # optional
```

**Step 5: Commit**

```bash
git commit -m "Initial commit: Empathetic Chatbot deployment"
```

**Step 6: Push**

```bash
git push origin main
```

When prompted for credentials:
- Username: Your Hugging Face username
- Password: The access token you created earlier

### Method 2: Using Web Interface (Simpler for Small Files)

1. Go to your Space page
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Upload files one by one or drag and drop
5. Click "Commit"

**Note**: This method may not work well for large files (>100MB)

---

## Testing Your Deployment

### Step 1: Wait for Build

After pushing, Hugging Face will:
1. Install dependencies (from requirements.txt)
2. Download your model
3. Start the Gradio app
4. This takes 2-5 minutes

### Step 2: Access Your App

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/empathetic-chatbot
```

### Step 3: Test Functionality

1. **Test basic input**: Enter a simple message
2. **Test emotions**: Try different emotions
3. **Test conversation**: Send multiple messages
4. **Check response quality**: Verify responses make sense

---

## Troubleshooting

### Common Issues and Solutions

**Issue 1: "Application startup failed"**
- **Cause**: Missing dependencies or wrong versions
- **Solution**: Check requirements.txt versions
```txt
torch>=2.0.0,<3.0.0
gradio>=4.0.0,<5.0.0
```

**Issue 2: "Model file too large"**
- **Cause**: Git LFS not configured
- **Solution**: 
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add LFS tracking"
git push origin main --force
```

**Issue 3: "CUDA out of memory"**
- **Cause**: Free tier uses CPU
- **Solution**: Model already uses CPU by default in app.py

**Issue 4: "Module not found"**
- **Cause**: Missing package in requirements.txt
- **Solution**: Add missing package:
```txt
# Add to requirements.txt
missing-package>=1.0.0
```

**Issue 5: "File not found: preprocessed/config.json"**
- **Cause**: File paths incorrect
- **Solution**: Ensure directory structure matches:
```
/
├── app.py
├── model/
│   └── best_model.pt
└── preprocessed/
    ├── config.json
    ├── vocab.json
    └── id_to_token.json
```

---

## Optimization Tips

### Reducing Model Size

If 361MB is too large:

```python
# Reduce to FP16 (half precision)
import torch

model = torch.load('best_model.pt', map_location='cpu')
for key in model['model_state_dict']:
    model['model_state_dict'][key] = model['model_state_dict'][key].half()
torch.save(model, 'best_model_fp16.pt')
```

### Improving Performance

1. **Enable caching**:
```python
demo.launch(cache_examples=True)
```

2. **Add queue**:
```python
demo.queue()
demo.launch()
```

3. **Reduce max_len** in generation for faster responses

---

## Upgrading Hardware

If you need more power:

1. Go to Space settings
2. Click "Hardware"
3. Upgrade options:
   - **CPU Upgrade**: $0.03/hour
   - **T4 Small GPU**: $0.60/hour
   - **A10G Large GPU**: $3.15/hour

Free tier is usually sufficient for this model.

---

## Monitoring and Analytics

### Check Logs

1. Go to your Space
2. Click "Logs" tab
3. View real-time application logs

### Monitor Usage

1. Go to Space settings
2. View "Analytics"
3. See:
   - Number of users
   - Total requests
   - Average response time

---

## Updating Your Model

To update with a new model version:

```bash
# Replace model file
cp new_best_model.pt model/best_model.pt

# Commit and push
git add model/best_model.pt
git commit -m "Update model to version X"
git push origin main
```

Hugging Face will automatically rebuild and redeploy.

---

## Making It Private

If you want to restrict access:

1. Go to Space settings
2. Change visibility to "Private"
3. Share with specific users:
   - Click "Share"
   - Add usernames
   - Set permissions

---

## Custom Domain (Optional)

To use your own domain:

1. Upgrade to Pro ($9/month)
2. Go to Space settings
3. Add custom domain
4. Update DNS records

---

## Cost Estimation

**Free Tier**:
- ✅ Unlimited public Spaces
- ✅ CPU compute
- ✅ Persistent storage
- ✅ Community support

**Limitations**:
- Slower than GPU
- May sleep after inactivity
- Shared resources

**Total Cost**: **$0/month** for basic deployment

---

## Next Steps

After successful deployment:

1. **Share your Space**:
   - Copy the URL
   - Share on social media
   - Add to your portfolio

2. **Add to Model Card**:
   - Document your model
   - Add training details
   - Include examples

3. **Community Engagement**:
   - Respond to feedback
   - Update based on usage
   - Collaborate with others

---

## Additional Resources

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://gradio.app/docs
- **Git LFS Docs**: https://git-lfs.github.com/

---

## Summary Checklist

Before deployment, ensure you have:

- [ ] Hugging Face account created
- [ ] Access token generated
- [ ] Space created
- [ ] All files prepared:
  - [ ] app.py
  - [ ] requirements.txt
  - [ ] .gitattributes
  - [ ] README.md
  - [ ] model/best_model.pt
  - [ ] preprocessed/*.json files
- [ ] Git and Git LFS installed
- [ ] Files uploaded/pushed
- [ ] App tested and working

---

**Congratulations!** Your empathetic chatbot is now live and accessible to everyone! 🎉

---

*Last Updated: October 2025*
*For issues or questions, consult the Hugging Face community forums*

