# Quick Start Guide

## TL;DR

Train a Sparse Autoencoder on Llama-3.1-8B activations using FAST methodology:

```bash
# Option 1: Run complete pipeline
python run_pipeline.py

# Option 2: Run steps individually
python extract_activations.py  # ~30 min on GPU
python train_sae.py            # ~15 min on GPU
python interpret_sae.py        # Optional, requires OpenAI API key
```

## Requirements

```bash
# Install dependencies
pip install torch transformers numpy tqdm

# HypotheSAEs should already be in ./HypotheSAEs/
# Dataset should be: cognitive_actions_7k_final_1759233061.jsonl
```

## What Gets Created

```
SAE_train/
├── activations/
│   └── activations_layer12.npz          # ~500MB - LLM activations
├── checkpoints/
│   └── cognitive_actions/
│       └── SAE_M=256_K=8.pt            # ~30MB - Trained SAE
└── interpretations.csv                  # Optional - Feature descriptions
```

## Configuration

Edit scripts to customize:

### extract_activations.py

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_IDX = 12        # Which layer (0-31 for Llama-3.1-8B)
MAX_LENGTH = 512      # Max tokens per example
```

### train_sae.py

```python
M = 256               # Total SAE features
K = 8                 # Active features per token
USE_MATRYOSHKA = True # Multi-granularity features
N_EPOCHS = 100        # Training epochs
```

## Choosing Hyperparameters

### Dataset Size → M, K

| Dataset Size | Recommended M | Recommended K |
|--------------|---------------|---------------|
| ~1,000 | 64 | 4 |
| **~7,000** (yours) | **256** | **8** |
| ~10,000 | 256 | 8 |
| ~100,000 | 1024 | 8 |

### Which Layer?

- **Early layers** (0-10): Syntax, basic patterns
- **Middle layers** (11-20): Semantic features ← **Recommended**
- **Late layers** (21-31): Task-specific features

For Llama-3.1-8B with 32 layers:
- Layer 12: General semantic features (default)
- Layer 18: Richer semantics
- Layer 25: Task-specific reasoning

## Expected Output

### Step 1: Activation Extraction

```
Loading model: meta-llama/Llama-3.1-8B-Instruct
Extracting from layer 12
Hidden dimension: 4096
Processing 6975 examples sequentially...
Extracted activations from 6975 examples
Sequence lengths: min=45, max=512, mean=187.3

Flattened shape: (1,306,428, 4096)
Total tokens (excl. padding): 1,306,428
```

### Step 2: SAE Training

```
Train set: 1,175,785 tokens
Val set: 130,643 tokens

Training SAE...
Epoch 100/100: train_loss=0.0234 val_loss=0.0241 dead_ratio=0.023

Val MSE: 0.0234 (normalized: 0.867)
Active features/token: 8.12
Dead neurons: 2.3%
```

### Step 3: Interpretation (Optional)

```
Neuron 42 (15.3% active): describes reconsidering a previous decision

Top activating examples:
1. "I'd fiercely advocated for maintaining the established visual style..."
2. "I have to publish this data now, get it out there before Ramirez does..."
3. "Initially, I felt an almost giddy gratitude after uncovering..."
```

## Troubleshooting

### OOM (Out of Memory)

**During extraction:**
```python
# Reduce max_length
MAX_LENGTH = 256  # Instead of 512
```

**During training:**
```python
# Reduce batch size
BATCH_SIZE = 256  # Instead of 512
```

### "No CUDA device"

Both scripts work on CPU, but slower:
- Extraction: ~2 hours instead of 30 min
- Training: ~1 hour instead of 15 min

### "Model not found in HF cache"

```python
# Specify cache directory
CACHE_DIR = "/path/to/huggingface/cache"
```

Or download model first:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir="/your/cache/dir"
)
```

### High dead neuron ratio (> 10%)

Possible fixes:
```python
# Increase aux_k (dead neuron revival)
aux_k = 32  # Default is 2*K=16

# Reduce M (fewer total features)
M = 128  # Instead of 256
```

## What to Do Next

1. **Analyze features**: Look at `interpretations.csv`
2. **Adjust M/K**: If features too coarse/fine, change M
3. **Try different layers**: Extract from layer 18 or 25
4. **Use features**: Apply to downstream tasks

## Key Files Reference

| File | Purpose | Size | Time to Create |
|------|---------|------|----------------|
| `extract_activations.py` | Extract LLM activations | - | 30 min |
| `activations_layer12.npz` | Stored activations | ~500MB | - |
| `train_sae.py` | Train SAE | - | 15 min |
| `SAE_M=256_K=8.pt` | Trained model | ~30MB | - |
| `interpret_sae.py` | Get feature descriptions | - | 10 min |
| `interpretations.csv` | Feature descriptions | ~50KB | - |

## Advanced Usage

### Use Matryoshka SAE

```python
# In train_sae.py
USE_MATRYOSHKA = True
MATRYOSHKA_PREFIXES = [64, 256]

# Trains SAE with two granularities:
# - First 64 features: Coarse patterns
# - All 256 features: Fine patterns
```

### Use Batch Top-K

```python
# In train_sae.py
USE_BATCH_TOPK = True

# Allows variable sparsity per token
# Better for variable-complexity examples
```

### Extract from Multiple Layers

```bash
# Modify extract_activations.py to loop over layers
for LAYER_IDX in [12, 18, 25]:
    extract_activations(layer_idx=LAYER_IDX)
```

## Performance Benchmarks

Hardware: NVIDIA A100 (40GB)

| Step | Time | Memory |
|------|------|--------|
| Extraction (7K examples) | 28 min | 16GB VRAM |
| Training (100 epochs) | 12 min | 4GB VRAM |
| Interpretation (20 features) | 8 min | API calls |

Hardware: CPU only (16GB RAM)

| Step | Time | Memory |
|------|------|--------|
| Extraction | 2h 15min | 8GB RAM |
| Training | 45 min | 4GB RAM |

## Further Reading

- `README_FAST.md` - Detailed methodology explanation
- `FAST_vs_HypotheSAEs.md` - Implementation comparison
- `padding_seq.md` - Original research on padding vs concatenation
- HypotheSAEs docs: https://github.com/rmovva/hypothesaes
