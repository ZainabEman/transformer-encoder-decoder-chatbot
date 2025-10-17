# Quick Start Guide

## For GitHub Repository

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (for training)
- GPU recommended (for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/empathetic-chatbot.git
cd empathetic-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

#### Option 1: Google Colab (Recommended)

1. Open Google Colab: https://colab.research.google.com/
2. Upload the dataset: `emotion-emotion_69k.csv`
3. Copy and paste code from `cells/` directory in order:
   - `1. task1_preprocessing.py`
   - `2. task2_model_architecture.py`
   - `3. task3_training.py` (This will take ~40 minutes)
   - `4. task4_evaluation.py`
   - `5. task5_inference_part1.py`
   - `6. task5_inference_part2.py`
   - `7. download_all_outputs.py`

#### Option 2: Local Execution

Each cell can be run as a standalone Python script:

```bash
# Preprocessing
python "cells/1. task1_preprocessing.py"

# Model Architecture
python "cells/2. task2_model_architecture.py"

# Training (requires preprocessed data)
python "cells/3. task3_training.py"

# Evaluation (requires trained model)
python "cells/4. task4_evaluation.py"

# Inference
python "cells/5. task5_inference_part1.py"
python "cells/6. task5_inference_part2.py"
```

### Dataset

Download the dataset from one of these sources:
- [Original Source] (provide link)
- [Kaggle] (provide link)
- Or use your own emotion-dialogue dataset

### Project Structure

```
.
├── cells/              # Training code (7 tasks)
├── images/             # Generated visualizations
├── README.md           # Project overview
├── BLOG.md            # Detailed technical blog
├── requirements.txt    # Dependencies
└── LICENSE            # MIT License
```

### Expected Outputs

After running all cells, you should have:
- `preprocessed_data/` - Processed dataset
- `best_model.pt` - Trained model (121 MB)
- `training_history.png` - Training curves
- `evaluation_results.png` - Performance metrics
- `data_preprocessing.png` - Data statistics

### Troubleshooting

**OutOfMemoryError during training:**
- Reduce batch size in task 3 (line ~20): `BATCH_SIZE = 32`
- Use Google Colab with GPU enabled

**Module not found:**
```bash
pip install [missing-module]
```

**Dataset upload fails:**
- Check file size (should be ~16.7 MB)
- Ensure filename is exactly `emotion-emotion_69k.csv`

### Next Steps

1. ✅ Train your model
2. ✅ Evaluate performance
3. ✅ Test inference locally
4. 🚀 Deploy to Hugging Face Spaces (see `HUGGINGFACE_DEPLOYMENT_GUIDE.md`)

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### License

MIT License - see LICENSE file for details

### Citation

If you use this project, please cite:

```bibtex
@misc{empathetic-chatbot-2025,
  author = {Your Name},
  title = {Empathetic Conversational Chatbot with Transformer},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/empathetic-chatbot}
}
```

---

**Need Help?** Check out:
- `README.md` - Comprehensive project documentation
- `BLOG.md` - Technical deep dive
- `REQUIREMENTS_CHECKLIST.md` - Task verification

