# FAST vs HypotheSAEs: Implementation Comparison

## Overview

This document explains how we've adapted the FAST (Finetuning-aligned Sequential Training) methodology to work with the HypotheSAEs library.

## FAST Paper Methodology

**Paper**: "Training Superior Sparse Autoencoders for Instruct Models" (Li et al., 2025)

### Core Innovation: Sequential Processing

```
┌─────────────────────────────────────────────────────────┐
│ FAST: Process each example independently                │
│                                                          │
│  Example 1 → LLM → Activations₁ (variable length)       │
│  Example 2 → LLM → Activations₂ (variable length)       │
│  Example 3 → LLM → Activations₃ (variable length)       │
│     ...                                                  │
│  Example N → LLM → Activations_N (variable length)      │
│                                                          │
│  All activations → Mixing Buffer → SAE Training         │
└─────────────────────────────────────────────────────────┘
```

**vs Block Training (BT) - The Baseline:**

```
┌─────────────────────────────────────────────────────────┐
│ Block Training: Concatenate then split into blocks      │
│                                                          │
│  [Ex1 + Ex2 + Ex3 + ...] → Split into 2048-token blocks │
│                                                          │
│  Block 1: [Ex1_part1 | Ex2_part1 | Ex2_part2]           │
│  Block 2: [Ex2_part3 | Ex3_part1 | Ex4_part1]           │
│     ...                                                  │
│                                                          │
│  Each block → LLM → Activations → SAE Training          │
└─────────────────────────────────────────────────────────┘
```

### Why FAST is Better

1. **Semantic Integrity**: No artificial boundaries splitting examples
2. **No Discontinuity**: Avoids mixing contexts from different examples
3. **Aligned with Instruct Models**: Matches fine-tuning methodology
4. **Better Reconstruction**: Paper shows MSE of 0.65 vs 5.20 (Block Training)
5. **Better Interpretability**: 21.1% high-quality features vs 7-10% (baselines)

## HypotheSAEs Library

**Paper**: "Sparse Autoencoders for Hypothesis Generation" (Movva et al., 2025)

### Original Design

HypotheSAEs was designed for:
- **Input**: Text → External embeddings (SentenceTransformers, OpenAI)
- **Purpose**: Hypothesis generation from text datasets
- **Usage**: Pre-computed fixed-size embeddings → SAE training

### SAE Architecture

The library implements:
- **Top-K SAE**: Selects K most active features per example
- **Matryoshka SAE**: Multi-granularity features with nested prefixes
- **Dead Neuron Revival**: Auxiliary loss to revive inactive neurons
- **Batch Top-K**: Adaptive sparsity across batch

## Our Implementation: FAST + HypotheSAEs

### Key Adaptations

| Component | FAST Paper | Our Implementation |
|-----------|------------|-------------------|
| **Input** | LLM activations (residual stream) | ✅ Same - use LLM activations |
| **Processing** | Sequential per-example | ✅ Same - `extract_activations.py` |
| **Variable Length** | Handled via truncation (8192) | ✅ Pad then exclude padding |
| **Mixing Buffer** | Shuffle activations before training | ✅ Shuffle during train/val split |
| **SAE Type** | Standard ReLU & JumpReLU | ✅ Use HypotheSAEs Top-K |
| **Sparsity** | L1 (Standard) or L0 (JumpReLU) | ✅ Top-K (similar to L0) |

### Two-Stage Pipeline

#### Stage 1: Activation Extraction (FAST approach)

```python
# extract_activations.py
for each example in dataset:
    # Process independently (FAST)
    tokens = tokenize(example)
    hidden_states = model(tokens).hidden_states
    activations = hidden_states[layer_idx]  # (seq_len, hidden_dim)

    # Store with actual length
    all_activations.append(activations)
    seq_lengths.append(len(tokens))

# Pad to same length for batching
padded_activations, mask = pad_activations(all_activations)

# Flatten, excluding padding positions
flattened = activations[mask]  # (total_real_tokens, hidden_dim)
```

**Key Insight**: We extract activations sequentially (FAST), then flatten for SAE training. This preserves semantic integrity during extraction while allowing batch training.

#### Stage 2: SAE Training (HypotheSAEs)

```python
# train_sae.py
from hypothesaes.quickstart import train_sae

sae = train_sae(
    embeddings=flattened_activations,  # From Stage 1
    M=256,  # Total features
    K=8,    # Active features
    matryoshka_prefix_lengths=[64, 256],  # Multi-granularity
    # ... other params
)
```

## Differences from Pure FAST

### What We Kept from FAST

✅ **Sequential extraction**: Each example processed independently
✅ **No concatenation**: Examples never mixed during extraction
✅ **Variable lengths**: Original lengths preserved until training
✅ **LLM activations**: Use internal representations, not embeddings

### What We Adapted

🔄 **SAE Architecture**: Use Top-K instead of ReLU/JumpReLU
- Top-K is conceptually similar to JumpReLU (both enforce sparsity)
- HypotheSAEs provides robust Top-K implementation

🔄 **Mixing Buffer**: Simplified to train/val shuffle
- FAST uses producer-consumer pattern during training
- We shuffle once during dataset split (simpler, equivalent effect)

🔄 **Batching**: Pad then exclude padding
- FAST uses dynamic batching with truncation
- We pad for PyTorch DataLoader, then mask out padding

## Implementation Details

### Padding Strategy

```python
# FAST approach: Dynamic batching, truncate to 8192
# Our approach: Pad to max_length, track valid positions

def flatten_activations(padded, mask, exclude_padding=True):
    if exclude_padding:
        # Only use real token positions
        return padded[mask]  # (n_real_tokens, hidden_dim)
    else:
        # Include padding (not recommended)
        return padded.reshape(-1, hidden_dim)
```

**Why this works**: SAE sees only real activations, not padding artifacts.

### Token-level vs Example-level

FAST extracts **token-level** activations, which is what we do:

```
Example 1: "I reconsidered my decision..."
  → 50 tokens → 50 activation vectors

Example 2: "The scientist questioned..."
  → 30 tokens → 30 activation vectors

Total tokens: 80 activation vectors for SAE training
```

For **interpretation**, we pool back to example-level:

```python
# interpret_sae.py
example_activations = []
for example_tokens in grouped_by_example:
    pooled = example_tokens.mean(axis=0)  # or max pooling
    example_activations.append(pooled)
```

## Recommended Hyperparameters

Based on FAST paper and HypotheSAEs recommendations:

### For ~7K examples (your dataset)

```python
M = 256          # Total features
K = 8            # Active per token
LAYER = 12       # Middle layer (Llama-3.1-8B has 32 layers)
MAX_LENGTH = 512 # Truncation length
BATCH_SIZE = 512
LR = 5e-4
EPOCHS = 100
```

### Optional Enhancements

**Matryoshka Prefixes**: Multi-granularity features
```python
matryoshka_prefix_lengths = [64, 256]
```
- First 64 features: Coarse-grained patterns
- Next 192 features: Fine-grained patterns

**Batch Top-K**: Adaptive sparsity
```python
batch_topk = True
```
- More flexible than fixed K per example
- Better for variable-complexity inputs

## Expected Results

### Reconstruction Quality

| Method | Normalized MSE | Notes |
|--------|---------------|-------|
| FAST (paper) | 0.65 | Best performance |
| Block Training (F) | 1.51 | Uses finetuning data |
| Block Training (P) | 5.20 | Uses pretraining data |
| **Our implementation** | **< 1.0** | Should match FAST-style |

### Interpretability

| Method | High-Quality Features (score 4-5) |
|--------|-----------------------------------|
| FAST (paper) | 21.1% |
| Block Training (F) | 10.2% |
| Block Training (P) | 7.0% |
| **Our implementation** | **~15-20%** (estimated) |

## Validation

To verify FAST-style implementation:

1. **Check reconstruction**: Normalized MSE < 1.0
2. **Check sparsity**: Active features ≈ K per token
3. **Check dead neurons**: < 5% dead after training
4. **Interpret features**: Should capture cognitive patterns

## Comparison Summary

| Aspect | FAST Paper | HypotheSAEs | Our Implementation |
|--------|-----------|-------------|-------------------|
| **Input** | LLM activations | External embeddings | LLM activations ✅ |
| **Extraction** | Sequential | Batch pre-computed | Sequential ✅ |
| **Context** | Variable length | Fixed size | Variable → Padded ✅ |
| **SAE** | ReLU/JumpReLU | Top-K | Top-K ✅ |
| **Sparsity** | L1/L0 loss | Top-K selection | Top-K selection ✅ |
| **Dead Neurons** | Not addressed | Auxiliary loss | Auxiliary loss ✅ |
| **Matryoshka** | No | Yes | Optional ✅ |

**Conclusion**: We successfully combine FAST's sequential extraction methodology with HypotheSAEs' robust SAE training and interpretation tools.

## References

1. Li, J. et al. (2025). "Training Superior Sparse Autoencoders for Instruct Models." arXiv:2506.07691
2. Movva, R. et al. (2025). "Sparse Autoencoders for Hypothesis Generation." arXiv:2502.04382
3. Bussmann et al. (2025). "Matryoshka SAEs." arXiv:2503.17547
