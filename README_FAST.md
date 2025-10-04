# FAST-style SAE Training on LLM Activations

This implementation adapts the FAST (Finetuning-aligned Sequential Training) methodology for training Sparse Autoencoders on LLM internal activations using the HypotheSAEs library.

## Overview

**FAST Methodology** (from paper "Training Superior Sparse Autoencoders for Instruct Models"):
- Processes each data instance **independently** (no concatenation)
- Preserves **semantic integrity** of individual examples
- Designed for **instruct models** (fine-tuned LLMs)
- Extracts activations from LLM hidden layers sequentially

**HypotheSAEs Library**:
- Top-K Sparse Autoencoder implementation
- Matryoshka SAEs for multi-granularity features
- Dead neuron revival with auxiliary loss
- Batch Top-K for adaptive sparsity

## Key Differences from Standard HypotheSAEs Usage

| Aspect | Standard HypotheSAEs | This Implementation (FAST-style) |
|--------|---------------------|----------------------------------|
| Input | External embeddings (SentenceTransformers/OpenAI) | LLM internal activations |
| Processing | Batch-based on pre-computed embeddings | Sequential extraction per example |
| Context | Fixed embedding per text | Layer activations (seq_len × hidden_dim) |
| Padding | Not needed (fixed-size embeddings) | Variable-length sequences, padded then flattened |

## Dataset

- **File**: `cognitive_actions_7k_final_1759233061.jsonl`
- **Size**: 6,975 examples
- **Format**: JSONL with `text` field containing cognitive action descriptions
- **Recommended SAE params**: M=256, K=8 (per HypotheSAEs recommendations for ~7K examples)

## Pipeline

### Step 1: Extract LLM Activations

```bash
python extract_activations.py
```

This script:
1. Loads `meta-llama/Llama-3.1-8B-Instruct` from local HuggingFace cache
2. Processes each example **independently** through the model
3. Extracts activations from layer 12 (middle layer)
4. Handles variable-length sequences with padding
5. Saves flattened activations (excluding padding positions)

**Output**: `activations/activations_layer12.npz`

**Key Parameters** (edit in script):
- `MODEL_NAME`: Default is "meta-llama/Llama-3.1-8B-Instruct"
- `LAYER_IDX`: 12 (middle of 32 layers)
- `MAX_LENGTH`: 512 tokens (reasonable for cognitive action examples)

### Step 2: Train SAE

```bash
python train_sae.py
```

This script:
1. Loads extracted activations
2. Splits into train/val (90/10)
3. Trains SAE using HypotheSAEs library
4. Saves checkpoint to `checkpoints/cognitive_actions/`
5. Reports reconstruction MSE and sparsity metrics

**Key Parameters** (edit in script):
- `M = 256`: Total SAE features
- `K = 8`: Active features per token
- `USE_MATRYOSHKA = True`: Multi-granularity features [64, 256]
- `USE_BATCH_TOPK = False`: Adaptive sparsity
- `N_EPOCHS = 100`: Training epochs
- `PATIENCE = 5`: Early stopping patience

## Expected Results

Based on the FAST paper and HypotheSAEs benchmarks:

**Reconstruction Quality**:
- Normalized MSE should be < 1.0 (better than baseline)
- Lower MSE indicates better reconstruction

**Sparsity**:
- Active features/token should be ~K (8 in default config)
- Dead neuron ratio should be low (< 5%)

**Interpretability**:
- Features can be interpreted using HypotheSAEs' `interpret_sae()` function
- Each neuron should capture a distinct cognitive pattern

## Advantages of FAST Approach

1. **Semantic Integrity**: Each example processed independently, no artificial boundaries
2. **No Concatenation Artifacts**: Avoids mixing contexts from different examples
3. **Aligned with Instruct Models**: Matches how instruct models are fine-tuned
4. **Better Feature Quality**: Paper shows 21.1% high-quality features vs 7-10% for baselines

## Files Created

```
SAE_train/
├── extract_activations.py      # Step 1: Extract LLM activations
├── train_sae.py                # Step 2: Train SAE
├── activations/
│   └── activations_layer12.npz # Extracted activations (created by Step 1)
└── checkpoints/
    └── cognitive_actions/
        └── SAE_M=256_K=8.pt    # Trained SAE (created by Step 2)
```

## Hardware Requirements

**Step 1 (Activation Extraction)**:
- GPU with ~16GB VRAM (for Llama-3.1-8B in fp16)
- ~30 minutes for 7K examples (sequential processing)

**Step 2 (SAE Training)**:
- GPU recommended but not required
- ~10-20 minutes on GPU for 100 epochs

## Interpreting SAE Features (Optional Step 3)

After training, you can interpret the SAE neurons:

```python
from hypothesaes.quickstart import interpret_sae
from hypothesaes.sae import load_model
import numpy as np

# Load trained SAE
sae = load_model("checkpoints/cognitive_actions/SAE_M=256_K=8.pt")

# Load activations
data = np.load("activations/activations_layer12.npz")
activations = data['activations']

# Load original texts
import json
texts = []
with open("cognitive_actions_7k_final_1759233061.jsonl") as f:
    for line in f:
        texts.append(json.loads(line)['text'])

# Interpret top neurons
results = interpret_sae(
    texts=texts,
    embeddings=activations,
    sae=sae,
    n_top_neurons=20,  # Interpret 20 most prevalent neurons
    interpreter_model="gpt-4.1",  # Requires OPENAI_KEY_SAE env var
    print_examples_n=3,
)
```

## References

1. **FAST Paper**: Li et al. (2025) "Training Superior Sparse Autoencoders for Instruct Models" [arXiv:2506.07691]
2. **HypotheSAEs**: Movva et al. (2025) "Sparse Autoencoders for Hypothesis Generation" [arXiv:2502.04382]
3. **Matryoshka SAEs**: Bussmann et al. (2025) [arXiv:2503.17547]

## Notes

- The FAST methodology is primarily about **how activations are extracted** (sequentially, no concatenation)
- HypotheSAEs library handles the SAE training with Top-K sparsity
- Padding is used for batching during training, but padding positions are excluded
- Variable-length sequences are preserved until training (unlike Block Training which forces fixed lengths)
