"""
Train Sparse Autoencoder on LLM activations using FAST methodology.

This script trains an SAE on activations extracted sequentially from individual
examples, following the FAST (Finetuning-aligned Sequential Training) approach.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add HypotheSAEs to path
sys.path.insert(0, str(Path(__file__).parent / "HypotheSAEs"))

from hypothesaes.quickstart import train_sae
from hypothesaes.sae import load_model


def load_activations(activations_path: str):
    """Load activations from .npz file."""
    print(f"Loading activations from {activations_path}")
    data = np.load(activations_path, allow_pickle=True)

    activations = data['activations']  # (n_tokens, hidden_dim)
    metadata = data['metadata'].item() if 'metadata' in data else {}

    print(f"Loaded activations shape: {activations.shape}")
    print(f"Metadata: {metadata}")

    return activations, metadata


def split_train_val(activations: np.ndarray, val_ratio: float = 0.1):
    """Split activations into train and validation sets."""
    n_total = activations.shape[0]
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    # Shuffle
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_activations = activations[train_indices]
    val_activations = activations[val_indices]

    print(f"Train set: {train_activations.shape[0]:,} tokens")
    print(f"Val set: {val_activations.shape[0]:,} tokens")

    return train_activations, val_activations


def main():
    """Main training pipeline."""
    # Configuration
    ACTIVATIONS_PATH = "activations/activations_layer12.npz"
    CHECKPOINT_DIR = "checkpoints/cognitive_actions"

    # SAE hyperparameters
    # For ~7K examples, use M=256, K=8 (per HypotheSAEs README)
    M = 256  # Total number of SAE features
    K = 8    # Active features per example

    # Optional: Use Matryoshka prefixes for multi-granularity features
    USE_MATRYOSHKA = True
    MATRYOSHKA_PREFIXES = [64, 256] if USE_MATRYOSHKA else None

    # Optional: Use Batch Top-K for richer features
    USE_BATCH_TOPK = False

    # Training parameters
    N_EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 5e-4
    PATIENCE = 5  # Early stopping patience
    VAL_RATIO = 0.1

    print("="*60)
    print("FAST-style SAE Training")
    print("="*60)

    # Load activations
    activations, metadata = load_activations(ACTIVATIONS_PATH)

    # Split into train/val
    print("\nSplitting data...")
    train_activations, val_activations = split_train_val(activations, val_ratio=VAL_RATIO)

    # Display configuration
    print("\n" + "="*60)
    print("SAE Configuration")
    print("="*60)
    print(f"M (total features): {M}")
    print(f"K (active features): {K}")
    print(f"Matryoshka: {USE_MATRYOSHKA}")
    if USE_MATRYOSHKA:
        print(f"  Prefixes: {MATRYOSHKA_PREFIXES}")
    print(f"Batch Top-K: {USE_BATCH_TOPK}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Patience: {PATIENCE}")

    # Train SAE
    print("\n" + "="*60)
    print("Training SAE")
    print("="*60)

    sae = train_sae(
        embeddings=train_activations,
        M=M,
        K=K,
        matryoshka_prefix_lengths=MATRYOSHKA_PREFIXES,
        batch_topk=USE_BATCH_TOPK,
        checkpoint_dir=CHECKPOINT_DIR,
        overwrite_checkpoint=False,  # Load existing if available
        val_embeddings=val_activations,
        # Optional SAE parameters
        aux_k=None,  # Default: 2*K for dead neuron revival
        multi_k=None,  # Optional: use for secondary reconstruction
        dead_neuron_threshold_steps=256,
        # Training parameters
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        aux_coef=1/32,  # Coefficient for auxiliary loss (dead neuron revival)
        multi_coef=0.0,  # Coefficient for multi-k loss
        patience=PATIENCE,
        clip_grad=1.0,
        show_progress=True,
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_activations_sae = sae.get_activations(val_activations, show_progress=True)

    # Compute sparsity statistics
    sparsity = (val_activations_sae != 0).mean()
    active_per_example = (val_activations_sae != 0).sum(axis=1).mean()

    print(f"\nSparsity statistics:")
    print(f"  Overall sparsity: {sparsity:.4f}")
    print(f"  Active features per token: {active_per_example:.2f} (target: {K})")

    # Compute reconstruction error
    print("\nComputing reconstruction error...")
    val_tensor = torch.tensor(val_activations, dtype=torch.float).to(sae.device)
    with torch.no_grad():
        recon, info = sae(val_tensor)
        mse = torch.nn.functional.mse_loss(recon, val_tensor).item()
        # Normalized MSE (as used in the paper)
        baseline_mse = torch.nn.functional.mse_loss(
            val_tensor.mean(dim=0, keepdim=True).expand_as(val_tensor),
            val_tensor
        ).item()
        normalized_mse = mse / baseline_mse

    print(f"  MSE: {mse:.4f}")
    print(f"  Normalized MSE: {normalized_mse:.4f}")

    # Check for dead neurons
    dead_neurons = (sae.steps_since_activation > sae.dead_neuron_threshold_steps).sum().item()
    dead_ratio = dead_neurons / sae.m_total_neurons

    print(f"\nDead neurons: {dead_neurons}/{sae.m_total_neurons} ({dead_ratio:.2%})")

    print("\n" + "="*60)
    print("SAE Training Summary")
    print("="*60)
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Model: M={M}, K={K}")
    print(f"Val MSE: {mse:.4f} (normalized: {normalized_mse:.4f})")
    print(f"Active features/token: {active_per_example:.2f}")
    print(f"Dead neurons: {dead_ratio:.2%}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    main()
